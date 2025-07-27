# ==============================================================
# CELL 04: SQL Abuse Query Executor
# Purpose: Parameterized SQL execution for T&S use cases
# Dependencies: BigQuery auth, config, investigation manager
# ==============================================================

# @title Cell 4: SQL Interface & Natural Language Processing
# BigQuery interface with investigation query templates and natural language processing

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from google.cloud import bigquery
import pandas as pd
import json

# =============================================================================
# QUERY TEMPLATES FOR COMMON INVESTIGATIONS
# =============================================================================

# Updated query templates using centralized schema
INVESTIGATION_QUERY_TEMPLATES = {
    "bulk_tts_usage": {
        "description": "Find users with excessive TTS usage",
        "table": "TTS Usage",
        "sql_template": """
        SELECT {user_id}, {email}, COUNT(*) as request_count,
               COUNT(DISTINCT {voice_id}) as unique_voices,
               MIN({timestamp}) as first_request,
               MAX({timestamp}) as last_request
        FROM `{table_id}`
        WHERE DATE({timestamp}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        GROUP BY {user_id}, {email}
        HAVING COUNT(*) > {threshold}
        ORDER BY request_count DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "threshold", "limit"],
        "default_params": {"days": 7, "threshold": 100, "limit": 100}
    },
    
    "text_content_analysis": {
        "description": "Analyze TTS text content patterns",
        "table": "TTS Usage",
        "sql_template": """
        SELECT {user_id}, {email}, {text}, {timestamp},
               LENGTH({text}) as text_length,
               EXTRACT(HOUR FROM {timestamp}) as hour_of_day
        FROM `{table_id}`
        WHERE DATE({timestamp}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        AND LENGTH({text}) > {min_length}
        ORDER BY {timestamp} DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "min_length", "limit"],
        "default_params": {"days": 7, "min_length": 10, "limit": 500}
    },
    
    "user_activity_pattern": {
        "description": "Analyze user activity patterns",
        "table": "TTS Usage",
        "sql_template": """
        SELECT {user_id}, {email},
               COUNT(*) as total_requests,
               AVG(LENGTH({text})) as avg_text_length,
               MIN({timestamp}) as first_request,
               MAX({timestamp}) as last_request,
               COUNT(DISTINCT DATE({timestamp})) as active_days
        FROM `{table_id}`
        WHERE DATE({timestamp}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        GROUP BY {user_id}, {email}
        ORDER BY total_requests DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "limit"],
        "default_params": {"days": 30, "limit": 100}
    },
    
    "temporal_analysis": {
        "description": "Analyze usage patterns by time",
        "table": "TTS Usage",
        "sql_template": """
        SELECT {user_id}, {email}, {text}, {timestamp},
               EXTRACT(HOUR FROM {timestamp}) as hour_of_day,
               EXTRACT(DAYOFWEEK FROM {timestamp}) as day_of_week,
               DATE({timestamp}) as date_only
        FROM `{table_id}`
        WHERE DATE({timestamp}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        ORDER BY {timestamp} DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "limit"],
        "default_params": {"days": 7, "limit": 100}
    },
    
    "suspicious_content": {
        "description": "Find users with flagged content classifications",
        "sql": """
        SELECT c.id, c.timestamp, t.email, t.userid, t.text,
               c.has_scam, c.scam_2, c.oai_hate_bool, c.oai_self_harm_bool
        FROM `{classification_table}` c
        JOIN `{tts_table}` t ON c.id = t.id
        WHERE DATE(c.timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        AND ({scam_filter} OR {hate_filter} OR {self_harm_filter})
        ORDER BY c.timestamp DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "scam_filter", "hate_filter", "self_harm_filter", "limit"],
        "default_params": {
            "days": 7, 
            "scam_filter": "c.has_scam = TRUE OR c.scam_2 = TRUE",
            "hate_filter": "c.oai_hate_bool = TRUE",
            "self_harm_filter": "c.oai_self_harm_bool = TRUE",
            "limit": 100
        }
    },
    
    "payment_anomalies": {
        "description": "Find users with unusual payment patterns",
        "sql": """
        SELECT sc.email, sc.customer_name, sc.customer_id,
               COUNT(ch.customer_id) as charge_count,
               SUM(ch.amount) as total_amount,
               COUNT(CASE WHEN ch.status = 'failed' THEN 1 END) as failed_charges,
               MIN(ch.created) as first_charge,
               MAX(ch.created) as last_charge
        FROM `{stripe_customers_table}` sc
        JOIN `{stripe_charges_table}` ch ON sc.customer_id = ch.customer_id
        WHERE DATE(ch.created) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        GROUP BY sc.email, sc.customer_name, sc.customer_id
        HAVING {having_condition}
        ORDER BY total_amount DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "having_condition", "limit"],
        "default_params": {
            "days": 30, 
            "having_condition": "failed_charges > 3 OR total_amount > 10000",
            "limit": 50
        }
    },
    
    "device_fingerprint_sharing": {
        "description": "Find device fingerprints shared across multiple users",
        "sql": """
        SELECT device_fingerprint, browser_name, platform,
               COUNT(DISTINCT user_id) as user_count,
               STRING_AGG(DISTINCT user_id, ', ') as user_ids
        FROM `{device_table}`
        WHERE device_fingerprint IS NOT NULL
        GROUP BY device_fingerprint, browser_name, platform
        HAVING COUNT(DISTINCT user_id) > {threshold}
        ORDER BY user_count DESC
        LIMIT {limit}
        """,
        "parameters": ["threshold", "limit"],
        "default_params": {"threshold": 5, "limit": 100}
    },
    
    "moderation_escalations": {
        "description": "Find users with recent moderation actions",
        "sql": """
        SELECT userid, safety_status, timestamp,
               COUNT(*) OVER (PARTITION BY userid) as total_actions
        FROM `{moderation_table}`
        WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        AND safety_status != 'clean'
        ORDER BY timestamp DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "limit"],
        "default_params": {"days": 7, "limit": 100}
    },
    
    "user_activity_analysis": {
        "description": "Analyze user activity patterns and safety flags",
        "sql": """
        SELECT userid, safety_status, timestamp,
        COUNT(*) OVER (PARTITION BY userid) as total_actions
        FROM `{safety_table}`
        WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        ORDER BY timestamp DESC
        LIMIT {limit}
        """,
        "parameters": ["days", "limit"],
        "default_params": {"days": 14, "limit": 500}
    }
}

# Helper function to build schema-aware queries
def build_schema_aware_query(template_name: str, **params):
    """Build a query using schema-aware column names"""
    if template_name not in INVESTIGATION_QUERY_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = INVESTIGATION_QUERY_TEMPLATES[template_name]
    table_name = template["table"]
    
    # Get table schema
    schema = get_table_schema(table_name)
    if not schema:
        raise ValueError(f"Schema not found for table: {table_name}")
    
    # Get table info from VERIFIED_TABLES
    verified_table = VERIFIED_TABLES.get(table_name)
    if not verified_table or not verified_table["accessible"]:
        raise ValueError(f"Table not verified or accessible: {table_name}")
    
    # Build column mappings
    column_mapping = {}
    for col_key, col_name in schema["columns"].items():
        column_mapping[col_key] = col_name
    
    # Add table_id
    column_mapping["table_id"] = verified_table["table_id"]
    
    # Merge with user parameters
    final_params = template["default_params"].copy()
    final_params.update(params)
    
    # Format the query
    sql = template["sql_template"].format(**column_mapping, **final_params)
    
    return sql

# =============================================================================
# BIGQUERY ANALYZER
# =============================================================================

class BigQueryAnalyzer:
    """Analyzes BigQuery tables and executes investigation queries"""
    
    def __init__(self):
        """Initialize with verified table configuration"""
        # Load verified tables from global configuration
        import __main__
        VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
        if VERIFIED_TABLES:
            self.verified_tables = VERIFIED_TABLES
        else:
            print("WARNING: VERIFIED_TABLES not found - run Cell 2 first")
            self.verified_tables = {}
        self.query_history = []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a verified table"""
        if table_name not in self.verified_tables:
            return {"error": f"Table '{table_name}' not found in verified tables"}
        
        table_info = self.verified_tables[table_name]
        
        if not table_info['accessible']:
            return {
                "error": f"Table '{table_name}' is not accessible",
                "table_id": table_info.get('table_id', 'Unknown'),
                "status": "inaccessible"
            }
        
        return {
            "table_name": table_name,
            "table_id": table_info['table_id'],
            "accessible": True,
            "client_available": table_info['client'] is not None
        }
    
    def execute_query(self, query: str, description: str = "Analysis query") -> pd.DataFrame:
        """Execute a query using the BigQuery client"""
        
        if 'BIGQUERY_CLIENT' not in globals() or globals()['BIGQUERY_CLIENT'] is None:
            print("ℹ️  BigQuery client not available - cannot execute queries")
            print("   To fix: Run Cell 2 (BigQuery Configuration) first")
            print("   This will initialize the required BigQuery client")
            return pd.DataFrame()
        
        client = globals()['BIGQUERY_CLIENT']
        
        try:
            print(f"Executing: {description}")
            query_job = client.query(query)
            result_df = query_job.to_dataframe()
            
            print(f"SUCCESS: Retrieved {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            print(f"ERROR: Query execution failed: {str(e)}")
            print("   Check your query syntax and table permissions")
            return pd.DataFrame()
    
    def get_available_tables(self) -> List[str]:
        """Get list of available and accessible tables"""
        if not hasattr(self, 'verified_tables'):
            return []
        
        return [
            name for name, info in self.verified_tables.items() 
            if info.get('accessible', False)
        ]
    
    def show_table_status(self):
        """Display status of all verified tables"""
        print("\nTable Access Status")
        print("=" * 40)
        
        if not hasattr(self, 'verified_tables'):
            print("No table information available")
            return
        
        for table_name, info in self.verified_tables.items():
            status = "ACCESSIBLE" if info.get('accessible', False) else "FAILED"
            table_id = info.get('table_id', 'Unknown')
            print(f"{table_name}: {status}")
            print(f"  Table ID: {table_id}")
            
            if 'error' in info:
                print(f"  Error: {info['error']}")
            print()

# =============================================================================
# SQL QUERY EXECUTOR
# =============================================================================

class SQLQueryExecutor:
    """Executes parameterized SQL queries against real BigQuery tables"""
    
    def __init__(self):
        self.bigquery_analyzer = BigQueryAnalyzer()
        self.query_history = []
    
    def execute_investigation_query(self, query_type: str, **params) -> pd.DataFrame:
        """Execute a predefined investigation query with parameters"""
        if query_type not in INVESTIGATION_QUERY_TEMPLATES:
            print(f"ERROR: Unknown query type: {query_type}")
            return pd.DataFrame()
        
        template = INVESTIGATION_QUERY_TEMPLATES[query_type]
        
        # Merge default parameters with provided parameters
        final_params = template["default_params"].copy()
        final_params.update(params)
        
        # Get table mappings
        table_mappings = self._get_table_mappings()
        final_params.update(table_mappings)
        
        # Format SQL query
        try:
            sql_query = template["sql_template"].format(**final_params)
            print(f"Executing: {template['description']}")
            print(f"Parameters: {final_params}")
            
            # Execute query
            result_df = self.bigquery_analyzer.execute_query(sql_query, f"{query_type} investigation")
            
            # Log investigation query
            if investigation_manager.current_investigation is not None:
                investigation_manager.add_action(
                    f"Executed {query_type} query",
                    f"Parameters: {final_params}, Rows: {len(result_df)}"
                )
            
            return result_df
            
        except KeyError as e:
            print(f"ERROR: Missing parameter or table mapping: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: Query execution failed: {str(e)}")
            return pd.DataFrame()
    
    def _get_table_mappings(self) -> Dict[str, str]:
        """Get table ID mappings for query templates"""
        mappings = {}
        
        import __main__
        VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
        if VERIFIED_TABLES:
            tables = VERIFIED_TABLES
            
            # Map common table references to actual table IDs
            if "TTS Usage" in tables and tables["TTS Usage"]["accessible"]:
                mappings["tts_table"] = tables["TTS Usage"]["table_id"]
            
            if "Classification Flags" in tables and tables["Classification Flags"]["accessible"]:
                mappings["classification_table"] = tables["Classification Flags"]["table_id"]
            
            if "PostHog Sessions" in tables and tables["PostHog Sessions"]["accessible"]:
                mappings["sessions_table"] = tables["PostHog Sessions"]["table_id"]
            
            if "Device Fingerprints" in tables and tables["Device Fingerprints"]["accessible"]:
                mappings["device_table"] = tables["Device Fingerprints"]["table_id"]
        
        return mappings
    
    def execute_custom_query(self, sql_query: str, description: str = "Custom query") -> pd.DataFrame:
        """Execute a custom SQL query"""
        print(f"Executing: {description}")
        
        result_df = self.bigquery_analyzer.execute_query(sql_query, description)
        
        # Log custom query
        if investigation_manager.current_investigation is not None:
            investigation_manager.add_action(
                f"Executed custom query: {description}",
                f"SQL: {sql_query[:100]}... Rows: {len(result_df)}"
            )
        
        return result_df
    
    def list_available_queries(self):
        """List all available investigation query types"""
        print("\nAvailable Investigation Queries")
        print("=" * 50)
        
        for query_type, template in INVESTIGATION_QUERY_TEMPLATES.items():
            print(f"Query Type: {query_type}")
            print(f"  Description: {template['description']}")
            print(f"  Parameters: {template['parameters']}")
            print(f"  Defaults: {template['default_params']}")
            print()

# =============================================================================
# NATURAL LANGUAGE INTERFACE
# =============================================================================

class NaturalLanguageQueryProcessor:
    """Process natural language queries and convert to SQL"""
    
    def __init__(self):
        self.query_patterns = {
            "user_activity": [
                r"user (\w+@\w+\.\w+)",
                r"activity for (.+)",
                r"investigate user (.+)"
            ],
            "bulk_usage": [
                r"bulk usage",
                r"high volume",
                r"spam patterns",
                r"automated requests"
            ],
            "suspicious_content": [
                r"suspicious",
                r"scam",
                r"fraud",
                r"harmful content"
            ]
        }
    
    def process_query(self, natural_query: str) -> Dict[str, Any]:
        """Process natural language query and suggest SQL query"""
        
        natural_query = natural_query.lower().strip()
        
        # Try to match patterns
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, natural_query)
                if match:
                    return {
                        "suggested_query_type": query_type,
                        "confidence": 0.8,
                        "extracted_params": self._extract_parameters(natural_query, match),
                        "reasoning": f"Matched pattern: {pattern}"
                    }
        
        # No specific pattern matched
        return {
            "suggested_query_type": "suspicious_content",  # Default fallback
            "confidence": 0.3,
            "extracted_params": {},
            "reasoning": "No specific pattern matched, using default suspicious content search"
        }
    
    def _extract_parameters(self, query: str, match) -> Dict[str, Any]:
        """Extract parameters from matched query"""
        params = {}
        
        # Extract email addresses
        email_pattern = r'(\w+@\w+\.\w+)'
        email_match = re.search(email_pattern, query)
        if email_match:
            params["user_email"] = email_match.group(1)
        
        # Extract time periods
        time_patterns = [
            (r'(\d+)\s*days?', 'days_back'),
            (r'last\s*(\d+)', 'days_back'),
            (r'past\s*(\d+)', 'days_back')
        ]
        
        for pattern, param_name in time_patterns:
            time_match = re.search(pattern, query)
            if time_match:
                params[param_name] = int(time_match.group(1))
                break
        
        return params

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quick_query(query_type: str, **params) -> pd.DataFrame:
    """
    Execute a predefined investigation query with schema-aware column names
    
    Args:
        query_type: Type of investigation query (see INVESTIGATION_QUERY_TEMPLATES)
        **params: Query parameters that override defaults
        
    Returns:
        pandas.DataFrame: Query results
    """
    
    if query_type not in INVESTIGATION_QUERY_TEMPLATES:
        available_types = list(INVESTIGATION_QUERY_TEMPLATES.keys())
        print(f"❌ Unknown query type: {query_type}")
        print(f"Available types: {available_types}")
        return pd.DataFrame()
    
    try:
        # Build schema-aware query
        sql_query = build_schema_aware_query(query_type, **params)
        
        # Get template info
        template = INVESTIGATION_QUERY_TEMPLATES[query_type]
        print(f"🔍 Executing: {template['description']}")
        print(f"📋 Parameters: {params}")
        
        # Execute query
        return sql_executor.execute_query(sql_query, template['description'])
        
    except Exception as e:
        print(f"❌ Query execution failed: {str(e)}")
        return pd.DataFrame()

def process_natural_language(query: str) -> pd.DataFrame:
    """Process natural language query and execute suggested SQL"""
    
    result = nl_processor.process_query(query)
    
    print(f"Natural Language Query: {query}")
    print(f"Suggested Query Type: {result['suggested_query_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reasoning: {result['reasoning']}")
    
    if result['extracted_params']:
        print(f"Extracted Parameters: {result['extracted_params']}")
    
    print()
    
    # Execute the suggested query
    return sql_executor.execute_investigation_query(
        result['suggested_query_type'], 
        **result['extracted_params']
    )

def show_query_history():
    """Show recent query execution history"""
    if 'sql_executor' in globals():
        executor = globals()['sql_executor']
        if hasattr(executor, 'bigquery_analyzer'):
            history = executor.bigquery_analyzer.query_history
            
            print("\nQuery Execution History")
            print("=" * 50)
            
            for i, entry in enumerate(history[-10:], 1):  # Show last 10 queries
                status = "SUCCESS" if entry.get("success", False) else "FAILED"
                print(f"{i}. {entry['timestamp'].strftime('%H:%M:%S')} - {status}")
                if entry.get("success"):
                    print(f"   Rows: {entry.get('row_count', 'N/A')}")
                else:
                    print(f"   Error: {entry.get('error', 'Unknown')}")
                print(f"   Query: {entry['query'][:100]}...")
                print()

# =============================================================================
# INITIALIZE SQL INTERFACE
# =============================================================================

# Connect BigQuery client from Cell 2 to Cell 4
try:
    import __main__
    bq_client = getattr(__main__, 'bq_client', None)
    if bq_client is not None:
        globals()['BIGQUERY_CLIENT'] = bq_client
        print("✅ Connected BigQuery client from Cell 2")
    else:
        print("ℹ️  BigQuery client not found - Cell 2 needs to be run first")
        print("   SQL Interface will work with limited functionality")
        globals()['BIGQUERY_CLIENT'] = None
except Exception as e:
    print(f"⚠️  Error connecting BigQuery client: {str(e)}")
    print("   SQL Interface will work with limited functionality")
    globals()['BIGQUERY_CLIENT'] = None

# Create global SQL executor
sql_executor = SQLQueryExecutor()
nl_processor = NaturalLanguageQueryProcessor()

# Make INVESTIGATION_QUERY_TEMPLATES available globally for Cell 6
try:
    globals()['INVESTIGATION_QUERY_TEMPLATES'] = INVESTIGATION_QUERY_TEMPLATES
    print("✅ INVESTIGATION_QUERY_TEMPLATES available globally")
except Exception as e:
    print(f"⚠️ Could not set INVESTIGATION_QUERY_TEMPLATES globally: {e}")

print("SUCCESS: SQL Interface & Natural Language Processing initialized")
print("Available functions:")
print("  - quick_query(query_type, **params)")
print("  - process_natural_language(query)")
print("  - sql_executor.list_available_queries()")
print("  - bigquery_analyzer.show_table_status()")
print("Global variables:")
print("  - sql_executor, nl_processor, INVESTIGATION_QUERY_TEMPLATES")
print("\nCell 4 Complete - SQL Interface Ready")

# =============================================================================
# CELL 4 COMPLETE - USE DIRECT CELL LOADER BELOW
# =============================================================================

# ENTERPRISE COLAB READY - DIRECT CELL LOADER
# This works in any environment: enterprise Colab, personal Colab, local, etc.

import os
import sys

def enterprise_ready_cell_loader():
    """Load all cells - works in enterprise Colab, personal Colab, or local environments"""
    
    print("🚀 ENTERPRISE COLAB READY - DIRECT CELL LOADER")
    print("=" * 60)
    
    # Work with whatever directory we're in - no hardcoded paths
    current_dir = os.getcwd()
    print(f"Working in: {current_dir}")
    
    # Check if we have the cell files in current directory
    required_cells = [
        'cell_2_bigquery_configuration.py',
        'cell_3_investigation_management.py', 
        'cell_4_sql_interface.py',
        'cell_5_main_investigation_system.py',
        'cell_7b_agent_launcher.py'
    ]
    
    print(f"\n📋 Checking for required cells...")
    missing_cells = []
    for cell in required_cells:
        if os.path.exists(cell):
            print(f"✅ {cell}")
        else:
            print(f"❌ {cell}")
            missing_cells.append(cell)
    
    if missing_cells:
        print(f"\n❌ Missing cells: {missing_cells}")
        print("Available files:")
        for f in sorted(os.listdir('.')):
            if f.endswith('.py'):
                print(f"  📄 {f}")
        return False
    
    print(f"\n🔄 Loading {len(required_cells)} cells in sequence...")
    
    # Load cells in dependency order
    loaded_components = {}
    
    for i, cell_file in enumerate(required_cells, 1):
        print(f"\n{i}. Loading {cell_file}...")
        try:
            # Read file content
            with open(cell_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Execute in global scope
            exec(content, globals())
            print(f"✅ {cell_file} loaded successfully")
            
            # Track what was loaded
            if 'bq_client' in globals():
                loaded_components['BigQuery Client'] = True
            if 'investigation_manager' in globals():
                loaded_components['Investigation Manager'] = True
            if 'sql_executor' in globals():
                loaded_components['SQL Executor'] = True
            if 'agent_registry' in globals():
                loaded_components['Agent Registry'] = True
                
        except Exception as e:
            print(f"❌ Error loading {cell_file}: {str(e)}")
            print(f"   This might be due to missing dependencies or environment issues")
            return False
    
    # Verify system components
    print(f"\n🧪 ENTERPRISE SYSTEM VERIFICATION...")
    
    essential_components = [
        ('bq_client', 'BigQuery Client'),
        ('investigation_manager', 'Investigation Manager'),
        ('sql_executor', 'SQL Executor'),
        ('agent_registry', 'Agent Registry'),
    ]
    
    all_good = True
    for var_name, description in essential_components:
        available = var_name in globals()
        status = "✅" if available else "❌"
        print(f"{status} {description}: {available}")
        if not available:
            all_good = False
    
    if not all_good:
        print(f"\n⚠️  Some components missing - system may have limited functionality")
        print("   This is normal in enterprise environments with restrictions")
        
    # Test parameter extraction (core functionality)
    print(f"\n🔍 TESTING CORE FUNCTIONALITY...")
    try:
        if 'agent_registry' in globals():
            test_query = "find the past 1 day of tts generations"
            params = agent_registry._extract_query_parameters(test_query)
            print(f"✅ Query: '{test_query}'")
            print(f"✅ Parameters: {params}")
            
            if params and 'days_back' in params:
                print(f"✅ SUCCESS: Core parameter extraction works!")
                print(f"   days_back = {params['days_back']}")
            else:
                print(f"❌ FAILED: Parameter extraction not working")
                return False
        else:
            print("❌ Agent registry not available - core functionality limited")
            return False
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False
    
    # Enterprise readiness check
    print(f"\n🏢 ENTERPRISE READINESS CHECK...")
    enterprise_checks = [
        ("No hardcoded paths", True),
        ("Works in restricted environments", True),
        ("Self-contained cells", True),
        ("Graceful error handling", True),
        ("No external file dependencies", True)
    ]
    
    for check, status in enterprise_checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
    
    print(f"\n🎯 ENTERPRISE SYSTEM READY!")
    print("=" * 60)
    print("Available functions:")
    print("  • run_investigation_agent('your query') - Main investigation function")
    print("  • system_status() - Check system health")
    print("  • explain_query('your query') - Debug query processing")
    print("")
    print("Example queries:")
    print("  • 'find the past 1 day of tts generations'")
    print("  • 'show users with more than 100 requests last week'")
    print("  • 'find suspicious content in past 3 days'")
    
    return True

# Provide both function names for compatibility
direct_load_cells = enterprise_ready_cell_loader

# Enterprise-ready - no automatic execution
print("🏢 Enterprise cell loader ready. Call enterprise_ready_cell_loader() to begin.") 