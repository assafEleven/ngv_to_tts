# ================================================================
# 📦 Full-Featured Trust & Safety Investigation UI (VERIFIED_TABLES ONLY)
# ================================================================

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import traceback
import re
import os
import openai
import json
import threading
import concurrent.futures

# --- Import VERIFIED_TABLES from Cell 2 ---
try:
    import __main__
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
    if VERIFIED_TABLES is None:
        raise RuntimeError("❌ VERIFIED_TABLES not found - run Cell 2 to verify tables.")
except Exception as e:
    VERIFIED_TABLES = None
    print(f"❌ VERIFIED_TABLES not found: {e}")

# --- OpenAI API Key Setup ---
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('⚠️  Set your OpenAI API key in the OPENAI_API_KEY environment variable.')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Helper: Get allowed table options for dropdowns ---
def get_allowed_table_options():
    if VERIFIED_TABLES:
        return [(f"{v['table_id']} ({k})", v['table_id']) for k, v in VERIFIED_TABLES.items() if v.get('accessible', True)]
    return []

# --- Helper: Get agent options (by table or logical agent) ---
def get_agent_options():
    # For demo, use table names as agent names; in production, use your agent registry
    return [(k, k) for k in VERIFIED_TABLES.keys()]

# --- Helper: Show schema summary for allowed tables ---
def show_allowed_schema_summary():
    if not VERIFIED_TABLES:
        print("❌ VERIFIED_TABLES not found - run Cell 2.")
        return
    print("📊 ALLOWED TABLES SCHEMA SUMMARY\n" + "="*60)
    for name, info in VERIFIED_TABLES.items():
        print(f"• {name}: {info['table_id']}")
        schema = info.get('schema', {})
        if schema and 'columns' in schema:
            print(f"   Columns: {', '.join(schema['columns'].keys())}")
        else:
            print("   Columns: (unknown)")
        print(f"   Description: {info.get('description', '')}")
        print()

# --- Helper: Get schema summary for prompt ---
def get_schema_for_prompt():
    schema_lines = []
    for name, info in VERIFIED_TABLES.items():
        schema = info.get('schema', {})
        columns = list(schema.get('columns', {}).keys()) if schema else []
        schema_lines.append(f"{name}: {info['table_id']}\n  Columns: {', '.join(columns)}")
    return '\n'.join(schema_lines)

def fix_timestamp_week_interval(sql: str) -> str:
    # Replace TIMESTAMP(timestamp) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY))
    # with timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    sql = re.sub(
        r"TIMESTAMP\s*\(\s*timestamp\s*\)\s*>=\s*TIMESTAMP\s*\(\s*DATE_SUB\s*\(\s*CURRENT_DATE\s*\(\s*\)\s*,\s*INTERVAL\s*7\s*DAY\s*\)\s*\)",
        "timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)",
        sql,
        flags=re.IGNORECASE
    )
    sql = re.sub(
        r"timestamp\s*>=\s*TIMESTAMP\s*\(\s*DATE_SUB\s*\(\s*CURRENT_DATE\s*\(\s*\)\s*,\s*INTERVAL\s*7\s*DAY\s*\)\s*\)",
        "timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)",
        sql,
        flags=re.IGNORECASE
    )
    # Lowercase text for LIKE
    sql = re.sub(
        r"text\s+LIKE\s+'%([^']+)%'",
        lambda m: f"LOWER(text) LIKE '%{m.group(1).lower()}%'",
        sql,
        flags=re.IGNORECASE
    )
    # Also fix INTERVAL 1 WEEK to INTERVAL 7 DAY
    sql = re.sub(r'INTERVAL\\s+1\\s+WEEK', 'INTERVAL 7 DAY', sql, flags=re.IGNORECASE)
    return sql

# --- OpenAI-powered NL Investigation ---
def run_natural_language_investigation(nl_query, days_back=7, limit=100):
    if not VERIFIED_TABLES:
        raise RuntimeError("❌ VERIFIED_TABLES not found - run Cell 2.")
    schema_for_prompt = get_schema_for_prompt()
    prompt = f"""
You are a Trust & Safety threat intelligence investigation assistant.

Your mission is to investigate abuse, fraud, or suspicious behavior using natural language queries from the investigator. You must use only the following tables and columns:
{schema_for_prompt}

The investigator's query is:
{nl_query}

Your responsibilities include:
1. **Translating queries into SQL**:
   - Only use tables and columns from the allowed list.
   - If multiple tables are required, construct proper explicit joins using allowed columns.
   - Output a complete SELECT statement, not just WHERE clauses.
   - Use safe BigQuery date handling:
     - For TIMESTAMP filtering: use `INTERVAL N DAY` (e.g. `INTERVAL 7 DAY`)
     - For DATE filtering: `INTERVAL N WEEK` is allowed.
   - Always alias tables meaningfully when joining.

2. **Explaining your logic**:
   - Clearly explain why you chose those tables and columns.
   - Describe the investigative intent (e.g. identifying TTS fraud, fingerprint clusters, Stripe abuse).
   - If applicable, identify the potential abuse type (e.g. financial fraud, spam/scams, ToS circumvention).

3. **Determining Abuse Risk**:
   - After reviewing data (via investigator-run queries), help interpret patterns or red flags.
   - If abuse is likely, name the abuse category and justify the determination.
   - Use caution: do not falsely accuse — label clearly if you are uncertain.

4. **Maintaining context and asking clarifying questions**:
   - If the query is ambiguous, ask a concise follow-up question to clarify (e.g., "Do you want to limit this to only refunded Stripe charges?").
   - Remember prior turns in the investigation to build context.

5. **Supporting DataFrame exploration**:
   - You may refer to rows or fields from previously returned DataFrames if the investigator requests.
   - You may ask to "show me column X from the last result" or "filter for only email addresses seen more than once."
   - If the investigator asks about the DataFrame or variable name, respond: 'The DataFrame is named df in the Python code.'

Your output must follow this strict JSON format:

```json
{{
  "sql": "...",
  "explanation": "...",
  "abuse_type": "...",  // Optional. One of: scam, fraud, child_safety, spam, policy_violation, unknown
  "follow_up_question": "..."  // Optional. Only if clarification is needed.
}}
Important:

Do not include any other fields or extra text.

Do not hallucinate tables or columns — use only what is provided.

If no SQL is needed, omit the "sql" field but still provide an explanation or follow-up.

Be strategic, concise, and investigative in tone. This is a real-world abuse investigation.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
        except Exception:
            # Try to extract JSON block from the output
            match = re.search(r'({.*})', content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except Exception as e2:
                    raise RuntimeError(f"❌ OpenAI output is not valid JSON. Raw output:\n{content}")
            else:
                raise RuntimeError(f"❌ OpenAI output is not valid JSON. Raw output:\n{content}")
        sql = parsed.get('sql')
        if sql:
            sql = fix_timestamp_week_interval(sql)
        explanation = parsed.get('explanation', '')
        abuse_type = parsed.get('abuse_type', '')
        follow_up_question = parsed.get('follow_up_question', '')
        # --- Allow explanation-only responses for meta-questions ---
        return sql, explanation, abuse_type, follow_up_question
    except Exception as e:
        raise RuntimeError(f"❌ OpenAI API or query failed: {e}")

# --- Structured Query Execution ---
def run_structured_query(table_key, where_clause, limit=100):
    if not VERIFIED_TABLES or table_key not in VERIFIED_TABLES:
        raise RuntimeError(f"❌ Table '{table_key}' not found in VERIFIED_TABLES.")
    table_info = VERIFIED_TABLES[table_key]
    table_id = table_info['table_id']
    query = f"SELECT * FROM `{table_id}` WHERE {where_clause} LIMIT {limit}"
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        df = client.query(query).result().to_dataframe()
        return df, query
    except Exception as e:
        raise RuntimeError(f"❌ Query failed: {e}\nQuery: {query}")

# --- UI Class ---
class InvestigationUI:
    def __init__(self):
        self.conversation = []  # Store conversation history for NL agent
        self.stop_requested = False
        self.stop_button = widgets.Button(description='Stop Agent', button_style='danger')
        self.stop_button.on_click(self.on_stop_agent)
        self.status_label = widgets.Label(value='Status: Idle')
        self.tabs = widgets.Tab()
        # --- NL Investigation Tab ---
        self.nl_query = widgets.Textarea(
            value='',
            placeholder='Describe your investigation (e.g., "Find all scam activity in the last 7 days")',
            description='NL Query:',
            layout=widgets.Layout(width='700px', height='60px')
        )
        self.nl_days = widgets.BoundedIntText(value=7, min=1, max=365, description='Days Back:')
        self.nl_limit = widgets.BoundedIntText(value=100, min=1, max=1000, description='Limit:')
        self.nl_run = widgets.Button(description='Run NL Investigation', button_style='primary')
        self.nl_run.on_click(self.on_run_nl)
        self.nl_output = widgets.Output()
        self.nl_log = widgets.Output()
        self.nl_box = widgets.VBox([
            self.status_label,
            self.nl_query,
            widgets.HBox([self.nl_days, self.nl_limit, self.nl_run]),
            self.nl_output,
            widgets.HTML('<b>Execution Log:</b>'),
            self.nl_log,
            self.stop_button
        ])
        # --- Structured Query Tab ---
        self.sq_table = widgets.Dropdown(options=get_agent_options(), description='Table:')
        self.sq_where = widgets.Text(value='1=1', description='WHERE:', layout=widgets.Layout(width='400px'))
        self.sq_limit = widgets.BoundedIntText(value=100, min=1, max=1000, description='Limit:')
        self.sq_run = widgets.Button(description='Run Structured Query', button_style='success')
        self.sq_run.on_click(self.on_run_sq)
        self.sq_output = widgets.Output()
        self.sq_log = widgets.Output()
        self.sq_box = widgets.VBox([
            self.sq_table,
            self.sq_where,
            self.sq_limit,
            self.sq_run,
            self.sq_output,
            widgets.HTML('<b>Execution Log:</b>'),
            self.sq_log
        ])
        # --- Schema Summary Tab ---
        self.schema_output = widgets.Output()
        self.schema_box = widgets.VBox([
            widgets.Button(description='Show Schema Summary', button_style='info',
                           layout=widgets.Layout(width='200px'),
                           tooltip='Show allowed tables'),
            self.schema_output
        ])
        self.schema_box.children[0].on_click(self.on_show_schema)
        # --- Tabs Setup ---
        self.tabs.children = [self.nl_box, self.sq_box, self.schema_box]
        self.tabs.set_title(0, 'Natural Language')
        self.tabs.set_title(1, 'Structured Query')
        self.tabs.set_title(2, 'Schema Summary')
    def get_system_prompt(self, nl_query):
        schema_for_prompt = get_schema_for_prompt()
        return f"""
IMPORTANT: You must always respond with a single valid JSON object as specified below. Never output plain text, markdown, or code blocks. If you cannot answer, return a JSON object with only an 'explanation' field and a 'suggested_query' field suggesting an alternate query the investigator could try.

- If the investigator asks about the DataFrame or variable name, always answer: "The DataFrame is named df in the Python code."
- Example Q&A:
  Q: What is the name of the DataFrame for this result?
  A: The DataFrame is named df in the Python code.

Your mission is to investigate abuse, fraud, or suspicious behavior using natural language queries from the investigator. You must use only the following tables and columns:
{schema_for_prompt}

The investigator's query is:
{nl_query}

Your responsibilities include:
1. **Translating queries into SQL**:
   - Only use tables and columns from the allowed list.
   - If multiple tables are required, construct proper explicit joins using allowed columns.
   - Output a complete SELECT statement, not just WHERE clauses.
   - Use safe BigQuery date handling:
     - For TIMESTAMP filtering: use `INTERVAL N DAY` (e.g. `INTERVAL 7 DAY`)
     - For DATE filtering: `INTERVAL N WEEK` is allowed.
   - Always alias tables meaningfully when joining.

2. **Explaining your logic**:
   - Clearly explain why you chose those tables and columns.
   - Describe the investigative intent (e.g. identifying TTS fraud, fingerprint clusters, Stripe abuse).
   - If applicable, identify the potential abuse type (e.g. financial fraud, spam/scams, ToS circumvention).

3. **Determining Abuse Risk**:
   - After reviewing data (via investigator-run queries), help interpret patterns or red flags.
   - If abuse is likely, name the abuse category and justify the determination.
   - Use caution: do not falsely accuse — label clearly if you are uncertain.

4. **Maintaining context and asking clarifying questions**:
   - If the query is ambiguous, ask a concise follow-up question to clarify (e.g., "Do you want to limit this to only refunded Stripe charges?").
   - Remember prior turns in the investigation to build context.

5. **Supporting DataFrame exploration**:
   - You may refer to rows or fields from previously returned DataFrames if the investigator requests.
   - You may ask to "show me column X from the last result" or "filter for only email addresses seen more than once."
   - If the investigator asks about the DataFrame or variable name, respond: 'The DataFrame is named df in the Python code.'

Your output must follow this strict JSON format:

```json
{{
  "sql": "...",
  "explanation": "...",
  "abuse_type": "...",  // Optional. One of: scam, fraud, child_safety, spam, policy_violation, unknown
  "follow_up_question": "...",  // Optional. Only if clarification is needed.
  "suggested_query": "..."  // Optional. Suggest an alternate query if the original is not answerable.
}}
Important:

Do not include any other fields or extra text.

Do not hallucinate tables or columns — use only what is provided.

If no SQL is needed, omit the "sql" field but still provide an explanation or follow-up.

Be strategic, concise, and investigative in tone. This is a real-world abuse investigation.
"""

    def run_nl_agent_with_memory(self, nl_query, days_back=7, limit=100):
        if not VERIFIED_TABLES:
            raise RuntimeError("❌ VERIFIED_TABLES not found - run Cell 2.")
        # On first run, add system prompt
        if not self.conversation:
            system_prompt = self.get_system_prompt(nl_query)
            self.conversation.append({"role": "system", "content": system_prompt})
        # Add user query
        self.conversation.append({"role": "user", "content": nl_query})
        try:
            def openai_call():
                return client.chat.completions.create(
                    model="gpt-4",
                    messages=self.conversation
                )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(openai_call)
                try:
                    response = future.result(timeout=30)  # 30 second timeout
                except concurrent.futures.TimeoutError:
                    self.status_label.value = "Status: OpenAI API call timed out."
                    print("❌ OpenAI API call timed out after 30 seconds.")
                    raise RuntimeError("OpenAI API call timed out.")
            content = response.choices[0].message.content
            # Save agent response to conversation
            self.conversation.append({"role": "assistant", "content": content})
            try:
                parsed = json.loads(content)
            except Exception:
                # Try to extract JSON block from the output
                match = re.search(r'({.*})', content, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception as e2:
                        raise RuntimeError(f"❌ OpenAI output is not valid JSON. Raw output:\n{content}")
                else:
                    raise RuntimeError(f"❌ OpenAI output is not valid JSON. Raw output:\n{content}")
            sql = parsed.get('sql')
            if sql:
                sql = fix_timestamp_week_interval(sql)
            explanation = parsed.get('explanation', '')
            abuse_type = parsed.get('abuse_type', '')
            follow_up_question = parsed.get('follow_up_question', '')
            return sql, explanation, abuse_type, follow_up_question
        except openai.error.OpenAIError as oe:
            self.status_label.value = f"Status: OpenAI API error: {oe}"
            print(f"❌ OpenAI API error: {oe}")
            raise
        except Exception as e:
            self.status_label.value = f"Status: Error: {e}"
            print(f"❌ Error: {e}")
            raise

    def display_conversation(self):
        print("\n--- Conversation History ---")
        for msg in self.conversation:
            if msg['role'] == 'user':
                print(f"User: {msg['content']}")
            elif msg['role'] == 'assistant':
                print(f"Agent: {msg['content']}")
        print("---------------------------\n")

    def on_stop_agent(self, b=None):
        print("⏹️ Stop requested.")
        self.stop_requested = True

    def extract_df_name(self, query):
        m = re.search(r'save (?:the )?df as ([A-Za-z_][A-Za-z0-9_]*)', query, re.IGNORECASE)
        if m:
            return m.group(1)
        return 'df'

    def on_run_nl(self, b=None):
        query = self.nl_query.value.strip()
        if not query:
            self.status_label.value = "Status: Please enter a natural language query."
            print("❌ Please enter a natural language query.")
            return
        print(f"[NL] Running: {query}")
        self.status_label.value = "Status: Waiting for OpenAI..."
        self.stop_requested = False
        df_name = self.extract_df_name(query)
        def run_agent():
            try:
                print("[DEBUG] Entered run_agent")
                if self.stop_requested:
                    self.status_label.value = "Status: Stopped by user (before OpenAI call)."
                    print("⏹️ Agent stopped by user (before OpenAI call).")
                    return
                print("[DEBUG] Calling OpenAI agent...")
                sql, explanation, abuse_type, follow_up_question = self.run_nl_agent_with_memory(query, days_back=self.nl_days.value, limit=self.nl_limit.value)
                print("[DEBUG] OpenAI agent call complete.")
                if self.stop_requested:
                    self.status_label.value = "Status: Stopped by user (after OpenAI call, before BigQuery)."
                    print("⏹️ Agent stopped by user (after OpenAI call, before BigQuery).")
                    return
                with self.nl_output:
                    clear_output()
                    self.display_conversation()
                    if sql:
                        from google.cloud import bigquery
                        bq_client = bigquery.Client()
                        self.status_label.value = f"Status: Running BigQuery for {df_name}..."
                        print(f"[DEBUG] About to run BigQuery for {df_name}...")
                        df = bq_client.query(sql).result().to_dataframe()
                        print(f"[DEBUG] BigQuery complete. DataFrame shape: {df.shape}")
                        globals()[df_name] = df
                        if self.stop_requested:
                            self.status_label.value = f"Status: Stopped by user (after BigQuery, before display)."
                            print("⏹️ Agent stopped by user (after BigQuery, before display).")
                            return
                        display(df)
                        print(f"✅ DataFrame saved as '{df_name}'")
                        print(f"✅ SQL: {sql}")
                        print(f"Explanation: {explanation}")
                        print(f"Abuse Type: {abuse_type}")
                        print(f"Follow-up Question: {follow_up_question}")
                        self.status_label.value = f"Status: Complete. DataFrame saved as '{df_name}'"
                    else:
                        # Use a scrollable Textarea for long explanations
                        explanation_widget = widgets.Textarea(
                            value=explanation or '',
                            layout=widgets.Layout(width='100%', height='120px'),
                            disabled=True
                        )
                        display(explanation_widget)
                        print(f"Abuse Type: {abuse_type}")
                        print(f"Follow-up Question: {follow_up_question}")
                        self.status_label.value = "Status: Complete (no SQL)."
            except Exception as e:
                self.status_label.value = f"Status: Error: {e}"
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                msg = str(e)
                if 'did not return a SQL statement or follow-up question' in msg or 'not valid JSON' in msg:
                    import re, json, ipywidgets as widgets
                    m = re.search(r'Raw output:\n([\s\S]+)', msg)
                    if m:
                        raw = m.group(1).strip()
                        # Try to extract a suggested query if present
                        suggested_query = None
                        sq_match = re.search(r'Suggested query: ([^\n]+)', raw)
                        if sq_match:
                            suggested_query = sq_match.group(1).strip()
                        explanation_widget = widgets.Textarea(
                            value=raw,
                            layout=widgets.Layout(width='100%', height='120px'),
                            disabled=True
                        )
                        display(explanation_widget)
                        if suggested_query:
                            print(f"Suggested Query: {suggested_query}")
                        self.status_label.value = "Status: Complete (explanation only)."
                        return
                print(f"[DEBUG] run_agent finished.")
        # Run synchronously for debugging
        run_agent()

    def on_run_sq(self, _):
        with self.sq_output:
            clear_output()
        with self.sq_log:
            clear_output()
            try:
                table = self.sq_table.value
                where = self.sq_where.value.strip()
                limit = self.sq_limit.value
                if not table:
                    print("❌ Please select a table.")
                    return
                print(f"[Structured] Table: {table}, WHERE: {where}, LIMIT: {limit}")
                df, sql = run_structured_query(table, where, limit)
                with self.sq_output:
                    clear_output()
                    display(df)
                print(f"✅ SQL: {sql}")
                print(f"Rows: {len(df)}")
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()
    def on_show_schema(self, _):
        with self.schema_output:
            clear_output()
            show_allowed_schema_summary()
    def display(self):
        display(self.tabs)

print("🚨 TRUST & SAFETY UI: Only VERIFIED_TABLES are available. No auto-discovery. No simulation. All code is production-ready.")
ui = InvestigationUI()
ui.display() 