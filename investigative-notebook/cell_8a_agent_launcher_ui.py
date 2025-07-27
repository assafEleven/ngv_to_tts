# ================================================================
# 📦 Cell 08: Interface and Launcher
# Purpose: Unified investigation UI and agent launcher for dynamic, schema-aware investigations
# Depends on: BigQuery auth (Cell 02), Schema Index (Cell 09)
# ================================================================

# @title Cell 8a: Unified Investigation Interface — Agent Selection & Natural Language
# Comprehensive Jupyter UI for trust & safety investigations

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
from datetime import datetime
import json
import time
import inspect
import threading
from typing import Dict, Any, Optional, List, Tuple
from google.cloud import bigquery
from google.auth import default
import re
import difflib

# =============================================================================
# GLOBAL AGENT EXECUTION CONTROL SYSTEM
# =============================================================================

# Global flags and registries for agent management
STOP_AGENT = False  # Global flag for stopping agents
AGENT_STATUS = {}  # Global dictionary to track agent status
AGENT_RESULTS = {}  # Global dictionary to store agent results
AGENT_LOGS = {}  # Global dictionary to store agent execution logs
AGENT_EXECUTION_LOCK = threading.Lock()  # Thread safety for agent execution

def stop_agent():
    """Stop the currently running agent"""
    global STOP_AGENT
    STOP_AGENT = True
    print("🛑 Agent stop signal sent")

def reset_agent_controls():
    """Reset all agent control flags"""
    global STOP_AGENT
    STOP_AGENT = False
    print("🔄 Agent controls reset")

def update_agent_status(agent_name: str, status: str):
    """Update agent status in global registry"""
    with AGENT_EXECUTION_LOCK:
        AGENT_STATUS[agent_name] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'last_updated': time.time()
        }
        print(f"🔄 Agent {agent_name} status: {status}")

def log_agent_step(agent_name: str, step_description: str):
    """Log agent execution step"""
    with AGENT_EXECUTION_LOCK:
        if agent_name not in AGENT_LOGS:
            AGENT_LOGS[agent_name] = []
        
        log_entry = {
            'step': step_description,
            'timestamp': datetime.now().isoformat(),
            'time': time.time()
        }
        AGENT_LOGS[agent_name].append(log_entry)
        print(f"📝 {agent_name}: {step_description}")

def store_agent_results(agent_name: str, results: Dict[str, Any]):
    """Store agent results in global registry"""
    with AGENT_EXECUTION_LOCK:
        AGENT_RESULTS[agent_name] = {
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'execution_time': time.time()
        }
        print(f"💾 Results stored for {agent_name}")

def check_agent_stop_signal(agent_name: str) -> bool:
    """Check if agent should stop execution"""
    if STOP_AGENT:
        update_agent_status(agent_name, "Stopped")
        log_agent_step(agent_name, "Execution stopped by user")
        return True
    return False

def show_agent_dashboard():
    """Display comprehensive agent dashboard"""
    print("=" * 80)
    print("🎯 UNIFIED AGENT DASHBOARD")
    print("=" * 80)
    
    # Agent Status Section
    print("\n🔄 AGENT STATUS:")
    print("-" * 40)
    if not AGENT_STATUS:
        print("   No agents have been executed yet")
    else:
        for agent_name, status_info in AGENT_STATUS.items():
            print(f"   {agent_name}: {status_info['status']} ({status_info['timestamp']})")
    
    # Agent Results Section
    print("\n📊 AGENT RESULTS:")
    print("-" * 40)
    if not AGENT_RESULTS:
        print("   No results available")
    else:
        for agent_name, result_info in AGENT_RESULTS.items():
            print(f"   {agent_name}:")
            results = result_info['results']
            if hasattr(results, 'records_found'):
                print(f"     Records Found: {results.records_found}")
                print(f"     High Risk Items: {results.high_risk_items}")
                print(f"     Execution Time: {results.execution_time:.2f}s")
            else:
                print(f"     Results: {str(results)[:100]}...")
    
    # Agent Logs Section
    print("\n📝 RECENT AGENT LOGS:")
    print("-" * 40)
    if not AGENT_LOGS:
        print("   No logs available")
    else:
        for agent_name, logs in AGENT_LOGS.items():
            print(f"   {agent_name} (last 3 steps):")
            for log_entry in logs[-3:]:
                print(f"     • {log_entry['step']}")
    
    # System Status
    print("\n⚙️ SYSTEM STATUS:")
    print("-" * 40)
    print(f"   Stop Signal: {'🛑 ACTIVE' if STOP_AGENT else '✅ Clear'}")
    print(f"   Active Agents: {len([s for s in AGENT_STATUS.values() if s['status'] in ['Running', 'Started']])}")
    print(f"   Total Executions: {len(AGENT_RESULTS)}")

# =============================================================================
# ENHANCED AGENT EXECUTION WRAPPER WITH CONVERSATIONAL OUTPUT
# =============================================================================

def execute_agent_with_monitoring(agent_name: str, original_function, query: str, **params):
    """
    Wrapper function to execute agents with monitoring, logging, and stop control
    """
    global STOP_AGENT
    
    # Reset stop flag for new execution
    STOP_AGENT = False
    
    # Initialize agent execution
    update_agent_status(agent_name, "Starting")
    log_agent_step(agent_name, f"Initializing with query: {query}")
    
    try:
        # Check for stop signal before starting
        if check_agent_stop_signal(agent_name):
            return None
            
        update_agent_status(agent_name, "Running")
        log_agent_step(agent_name, "Beginning execution")
        
        # Execute the original agent function
        result = original_function(query, **params)
        
        # Check for stop signal after execution
        if check_agent_stop_signal(agent_name):
            return None
        
        # CRITICAL: Validate result properly - no false success reporting
        execution_success = validate_agent_result(result)
        
        if execution_success:
            store_agent_results(agent_name, result)
            update_agent_status(agent_name, "Completed")
            log_agent_step(agent_name, f"Execution completed successfully")
            
            # Generate conversational summary
            print_agent_conversational_summary(agent_name, result, query, **params)
        else:
            update_agent_status(agent_name, "Failed")
            log_agent_step(agent_name, "Execution failed - invalid or empty results")
            
            # Generate failure summary
            print_agent_failure_summary(agent_name, result, query, **params)
            
        return result
        
    except Exception as e:
        update_agent_status(agent_name, "Error")
        log_agent_step(agent_name, f"Error during execution: {str(e)}")
        print_agent_error_summary(agent_name, e, query, **params)
        return None

def validate_agent_result(result) -> bool:
    """Validate if agent result represents actual success"""
    if not result:
        return False
    
    # Check if result has error attribute
    if hasattr(result, 'error') and result.error:
        return False
    
    # Check if result has analysis_results and they're not empty
    if hasattr(result, 'analysis_results'):
        return len(result.analysis_results) > 0
    
    # Check if result has any meaningful data
    if hasattr(result, 'records_found'):
        return result.records_found > 0
    
    # If we can't determine, be conservative and return False
    return False

def print_agent_conversational_summary(agent_name: str, result, query: str, **params):
    """Generate conversational summary of agent execution"""
    print("\n" + "="*80)
    print(f"🤖 {agent_name.upper().replace('_', ' ')} INVESTIGATION SUMMARY")
    print("="*80)
    
    # What the agent looked for
    print(f"🗣️  \"I searched for: {query}\"")
    
    # Parameters used
    print(f"\n🔧 Search Parameters:")
    print(f"   • Time range: Past {params.get('days_back', 7)} days")
    print(f"   • Result limit: {params.get('limit', 100)}")
    if params.get('source_filter'):
        print(f"   • Source filter: {params.get('source_filter')}")
    if params.get('target_uid'):
        print(f"   • Target focus: {params.get('target_uid')}")
    
    # Table usage summary
    print(f"\n📊 Tables Queried:")
    table_summary = get_table_usage_summary_from_sql_history(agent_name)
    if table_summary:
        for table_info in table_summary:
            status_icon = "✅" if table_info['success'] else "❌"
            print(f"   {status_icon} {table_info['table_name']}")
            if table_info['success']:
                print(f"      → {table_info['rows']} rows examined")
            else:
                print(f"      → Error: {table_info['error']}")
    else:
        print("   • TTS Usage table (primary data source)")
        print("   • Classification Flags table (optional - graceful degradation)")
    
    # Results analysis
    if hasattr(result, 'analysis_results') and result.analysis_results:
        print(f"\n🔍 Investigation Results:")
        print(f"   • Total matches found: {len(result.analysis_results)}")
        
        # Show risk breakdown
        risk_counts = {}
        for analysis in result.analysis_results:
            risk_level = analysis.get('risk_level', 'Unknown')
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        for risk_level, count in risk_counts.items():
            risk_icon = "🚨" if risk_level == "HIGH" else "⚠️" if risk_level == "MEDIUM" else "ℹ️"
            print(f"   {risk_icon} {risk_level}: {count} items")
        
        # Show sample results
        print(f"\n📝 Sample Results:")
        for i, analysis in enumerate(result.analysis_results[:3], 1):
            print(f"   {i}. User: {analysis.get('email', 'Unknown')}")
            print(f"      Risk: {analysis.get('risk_level', 'Unknown')}")
            if analysis.get('tts_text'):
                text_preview = analysis.get('tts_text', '')[:100]
                print(f"      Content: \"{text_preview}{'...' if len(text_preview) == 100 else ''}\"")
        
        if len(result.analysis_results) > 3:
            print(f"   ... and {len(result.analysis_results) - 3} more results")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if any(r.get('risk_level') == 'HIGH' for r in result.analysis_results):
            print(f"   • Immediate review recommended for HIGH risk items")
        print(f"   • Consider expanding search parameters for broader analysis")
        
    else:
        print(f"\n🔍 Investigation Results: No matches found")
        print_detailed_no_results_explanation(agent_name, result, **params)

def print_agent_failure_summary(agent_name: str, result, query: str, **params):
    """Generate failure summary for agent execution"""
    print("\n" + "="*80)
    print(f"❌ {agent_name.upper().replace('_', ' ')} EXECUTION FAILED")
    print("="*80)
    
    print(f"🗣️  \"I attempted to search for {query} but encountered an error\"")
    
    # Table usage summary
    print("\n📊 Tables Attempted:")
    table_summary = get_table_usage_summary_from_sql_history(agent_name)
    if table_summary:
        for table_info in table_summary:
            status_icon = "✅" if table_info['success'] else "❌"
            print(f"   {status_icon} {table_info['table_name']}")
            if not table_info['success']:
                print(f"      Issue: {table_info['error']}")
    else:
        print("   • TTS Usage table (attempted)")
        print("   • Classification Flags table (optional)")
    
    # Error details
    if result and hasattr(result, 'error'):
        print(f"\n🚨 Error Details: {result.error}")
    
    print(f"\n💡 Recommendation: Check system logs and retry with different parameters")

def print_agent_error_summary(agent_name: str, error: Exception, query: str, **params):
    """Generate error summary for agent execution"""
    print("\n" + "="*80)
    print(f"💥 {agent_name.upper().replace('_', ' ')} SYSTEM ERROR")
    print("="*80)
    
    print(f"🗣️  \"I encountered a system error while trying to search for {query}\"")
    print(f"\n🚨 Error: {str(error)}")
    
    # Provide specific guidance based on error type
    if "'DataFrame' object has no attribute 'result'" in str(error):
        print(f"\n💡 Diagnosis: SQL executor integration issue")
        print(f"   This suggests the enhanced SQL executor wasn't properly integrated")
        print(f"   Recommendation: Restart the notebook and re-run all cells in order")
    elif "Not found: Dataset" in str(error):
        print(f"\n💡 Diagnosis: Database access issue")
        print(f"   Recommendation: Check BigQuery permissions and table availability")
    elif "Unrecognized name" in str(error):
        print(f"\n💡 Diagnosis: Table schema mismatch")
        print(f"   Recommendation: Check table schema and column names")
    else:
        print(f"\n💡 Recommendation: Contact system administrator or check agent configuration")

def print_detailed_no_results_explanation(agent_name: str, result, **params):
    """Provide detailed explanation of why no results were found"""
    print(f"\n❓ Why no results were found:")
    
    # Check if queries actually ran
    if 'sql_executor' in globals() and hasattr(sql_executor, 'get_query_history'):
        query_history = sql_executor.get_query_history()
        successful_queries = [q for q in query_history if q.get('success', False)]
        failed_queries = [q for q in query_history if not q.get('success', True)]
        
        if failed_queries:
            print(f"   • {len(failed_queries)} queries failed due to errors")
            for failed_query in failed_queries[-2:]:  # Show last 2 failures
                print(f"     → {failed_query.get('error', 'Unknown error')}")
        
        if successful_queries:
            total_rows = sum(q.get('rows', 0) for q in successful_queries)
            print(f"   • {len(successful_queries)} queries succeeded but returned {total_rows} total rows")
            if total_rows == 0:
                print(f"     → No data exists matching your criteria")
            else:
                print(f"     → Data exists but didn't match search filters")
    else:
        print(f"   • Query execution history not available")
    
    # Suggest improvements
    print(f"\n💡 Suggestions to find more results:")
    print(f"   • Try increasing days_back from {params.get('days_back', 7)} to 14 or 30")
    print(f"   • Consider broader search terms")
    if params.get('source_filter'):
        print(f"   • Remove or broaden the source filter '{params.get('source_filter')}'")
    print(f"   • Check if similar searches work with different parameters")

def get_table_usage_summary_from_sql_history(agent_name: str) -> list:
    """Get table usage summary from SQL execution history"""
    tables = []
    
    if 'sql_executor' in globals() and hasattr(sql_executor, 'get_query_history'):
        query_history = sql_executor.get_query_history()
        
        # Extract table names from queries
        for query_info in query_history:
            # Simple regex to find table names (this is a basic implementation)
            import re
            table_matches = re.findall(r'FROM `([^`]+)`', query_info.get('query', ''))
            
            for table_name in table_matches:
                tables.append({
                    'table_name': table_name,
                    'success': query_info.get('success', False),
                    'rows': query_info.get('rows', 0),
                    'error': query_info.get('error', None)
                })
    
    return tables

# =============================================================================
# ENHANCED SQL QUERY EXECUTOR - REPLACE EXISTING IMPLEMENTATION
# =============================================================================

class EnhancedSQLQueryExecutor:
    """Enhanced SQL Query Executor with proper error handling and conversational output"""
    
    def __init__(self, bq_client):
        self.bq_client = bq_client
        self.last_query = None
        self.last_error = None
        self.query_history = []
        
    def query(self, sql_template: str, **params) -> pd.DataFrame:
        """
        Execute SQL query with proper error handling and parameter substitution
        
        Args:
            sql_template: SQL query with parameter placeholders
            **params: Parameters to substitute in the query
            
        Returns:
            pd.DataFrame: Query results or empty DataFrame on error
        """
        try:
            # Store the query for debugging
            self.last_query = sql_template
            self.last_error = None
            
            # Parameter substitution
            formatted_query = sql_template.format(**params)
            
            print(f"🔍 Executing SQL query...")
            print(f"📝 Query preview: {formatted_query[:100]}...")
            
            # Execute query
            query_job = self.bq_client.query(formatted_query)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            # Track successful queries
            self.query_history.append({
                'query': formatted_query,
                'success': True,
                'rows': len(df),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Query executed successfully - {len(df)} rows returned")
            return df
            
        except Exception as e:
            self.last_error = str(e)
            
            # Track failed queries
            self.query_history.append({
                'query': sql_template,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"❌ SQL Query failed: {str(e)}")
            
            # Check for specific errors and provide helpful messages
            if "Unrecognized name" in str(e):
                print(f"💡 Hint: Column name not found in table schema")
            elif "Table not found" in str(e):
                print(f"💡 Hint: Table does not exist or access denied")
            elif "Access Denied" in str(e):
                print(f"💡 Hint: Check BigQuery permissions")
            
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def execute_query(self, query: str, **params) -> pd.DataFrame:
        """Alias for query method for backward compatibility"""
        return self.query(query, **params)
    
    def get_last_error(self) -> str:
        """Get the last error message"""
        return self.last_error
    
    def get_query_history(self) -> list:
        """Get history of executed queries"""
        return self.query_history

# =============================================================================
# MONKEY PATCH TO REPLACE EXISTING SQLQueryExecutor
# =============================================================================

# Replace the existing SQLQueryExecutor globally
if 'sql_executor' in globals():
    # Replace the existing sql_executor with enhanced version
    sql_executor = EnhancedSQLQueryExecutor(bq_client)
    print("✅ SQLQueryExecutor replaced with EnhancedSQLQueryExecutor")

# Also replace in main_system if it exists
if 'main_system' in globals() and hasattr(main_system, 'sql_executor'):
    main_system.sql_executor = EnhancedSQLQueryExecutor(bq_client)
    print("✅ main_system.sql_executor replaced with EnhancedSQLQueryExecutor")

# =============================================================================
# FIXED DETECTION MODE VALIDATION
# =============================================================================

def validate_detection_mode_accurately():
    """Accurately validate detection mode without contradictions"""
    
    print("🔍 Validating detection capabilities...")
    
    # Test Classification Flags table directly
    try:
        if 'bq_client' not in globals():
            print("❌ BigQuery client not available")
            return 'basic_scam_detection'
        
        # Try a simple query to the Classification Flags table
        test_query = """
        SELECT COUNT(*) as row_count 
        FROM `eleven-team-safety.trust_safety.content_analysis` 
        LIMIT 1
        """
        
        query_job = bq_client.query(test_query)
        result = query_job.result()
        
        print("✅ Classification Flags table accessible - Enhanced detection available")
        return 'enhanced_scam_detection'
        
    except Exception as e:
        print(f"⚠️  Classification Flags table not accessible: {str(e)}")
        print("→ Falling back to basic detection mode")
        return 'basic_scam_detection'

# Replace the existing get_detection_mode function
def get_detection_mode():
    """Get detection mode with accurate validation"""
    return validate_detection_mode_accurately()

# =============================================================================
# ENHANCED GRACEFUL DEGRADATION SYSTEM
# =============================================================================

def check_table_availability_enhanced(table_name: str, bq_client=None) -> dict:
    """
    Enhanced table availability check with detailed error reporting
    
    Returns:
        dict: {
            'available': bool,
            'error': str or None,
            'can_fallback': bool,
            'fallback_reason': str or None
        }
    """
    result = {
        'available': False,
        'error': None,
        'can_fallback': False,
        'fallback_reason': None
    }
    
    try:
        if not bq_client:
            if 'bq_client' in globals():
                bq_client = globals()['bq_client']
            else:
                result['error'] = "BigQuery client not available"
                return result
        
        # Try a simple query to test table accessibility
        test_query = f"SELECT COUNT(*) as row_count FROM `{table_name}` LIMIT 1"
        query_job = bq_client.query(test_query)
        query_job.result()
        
        result['available'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        
        # Determine if we can fallback
        if "Unrecognized name" in str(e):
            result['can_fallback'] = True
            result['fallback_reason'] = "Schema mismatch - will use basic detection"
        elif "Table not found" in str(e):
            result['can_fallback'] = True
            result['fallback_reason'] = "Table not found - will use alternative data source"
        elif "Access Denied" in str(e):
            result['can_fallback'] = True
            result['fallback_reason'] = "Access denied - will use available tables only"
        else:
            result['can_fallback'] = False
            result['fallback_reason'] = "Unknown error - cannot fallback"
        
        return result

def get_detection_mode_enhanced():
    """Enhanced detection mode determination with detailed reporting"""
    classification_flags_status = check_table_availability_enhanced(
        'eleven-team-safety.trust_safety.content_analysis'
    )
    
    if classification_flags_status['available']:
        print("✅ Enhanced detection mode: Classification Flags table available")
        return 'enhanced_scam_detection'
    else:
        print(f"⚠️  Basic detection mode: {classification_flags_status['fallback_reason']}")
        if classification_flags_status['error']:
            print(f"   Error details: {classification_flags_status['error']}")
        return 'basic_scam_detection'

# =============================================================================
# DEPENDENCY CHECKS AND GLOBAL VARIABLE INITIALIZATION
# =============================================================================

# Ensure all required global variables are available
print("Checking system dependencies...")

# Check and initialize investigation_manager
try:
    if 'investigation_manager' in globals():
        if not hasattr(investigation_manager, 'current_investigation'):
            print("Fixing missing current_investigation attribute...")
            investigation_manager.current_investigation = None
            print("current_investigation attribute added")
        print("investigation_manager is available")
    else:
        print("❌ WARNING: investigation_manager not found - some features will be limited")
        investigation_manager = None
except NameError:
    print("❌ WARNING: investigation_manager not found - some features will be limited")
    investigation_manager = None

# Check agent_registry
try:
    if 'agent_registry' not in globals():
        print("❌ WARNING: agent_registry not found - some features will be limited")
        agent_registry = None
    else:
        print("agent_registry is available")
except NameError:
    print("❌ WARNING: agent_registry not found - some features will be limited")
    agent_registry = None

# Check main_system
try:
    if 'main_system' not in globals():
        print("❌ WARNING: main_system not found - some features will be limited")
        main_system = None
    else:
        print("main_system is available")
except NameError:
    print("❌ WARNING: main_system not found - some features will be limited")
    main_system = None

print("Dependency check complete - initializing UI...")

# =============================================================================
# SECTION 1: NATURAL LANGUAGE INVESTIGATION FUNCTIONS
# =============================================================================

def run_natural_language_investigation(query: str, show_data: bool = True, max_rows: int = 10):
    """
    Enhanced natural language investigation with proper error handling
    """
    
    print("=" * 80)
    print(f"NATURAL LANGUAGE INVESTIGATION")
    print("=" * 80)
    print(f"Query: {query}")
    print()
    
    # Check if agent_registry is available
    if not agent_registry:
        print("❌ ERROR: agent_registry not available - cannot run natural language investigation")
        return None
    
    # Check detection mode with accurate validation
    detection_mode = get_detection_mode()
    print(f"Detection Mode: {detection_mode}")
    
    # Step 1: Test intent detection
    print("Step 1: Intent Detection")
    print("-" * 30)
    
    try:
        intent = agent_registry.detect_intent(query)
        if intent:
            print(f"SUCCESS: Detected intent")
            print(f"  Abuse Type: {intent.abuse_type}")
            print(f"  Suggested Agent: {intent.suggested_agent}")
            print(f"  Confidence: {intent.confidence}")
            print(f"  Extracted Parameters: {intent.extracted_params}")
            print()
        else:
            print("WARNING: No intent detected")
            print()
            return None
    except Exception as e:
        print(f"ERROR in intent detection: {str(e)}")
        return None
    
    # Step 2: Execute the agent with enhanced monitoring
    print("Step 2: Agent Execution")
    print("-" * 30)
    
    try:
        # Use the enhanced monitoring wrapper for execution
        result = execute_agent_with_monitoring(
            intent.suggested_agent,
            lambda q, **p: agent_registry.run_agent(intent.suggested_agent, q, **p),
            query,
            **intent.extracted_params
        )
        
        return result
            
    except Exception as e:
        print(f"❌ ERROR: Investigation failed: {str(e)}")
        return None

def quick_scam_investigation():
    """Run a quick scam investigation"""
    query = "Find recent scam activity from PlayAPI users"
    return run_natural_language_investigation(query)

def quick_agent_status():
    """Show quick agent system status"""
    print("Agent System Status")
    print("=" * 40)
    
    try:
        if agent_registry:
            print(f"Available agents: {len(agent_registry.agents) if hasattr(agent_registry, 'agents') else 'Unknown'}")
            print(f"Registered handlers: {len(agent_registry.agent_handlers) if hasattr(agent_registry, 'agent_handlers') else 'Unknown'}")
            print(f"Intent patterns: {len(agent_registry.intent_patterns) if hasattr(agent_registry, 'intent_patterns') else 'Unknown'}")
            print()
            
            if hasattr(agent_registry, 'agents'):
                print("Quick Agent List:")
                for name, config in agent_registry.agents.items():
                    description = config.description if hasattr(config, 'description') else 'No description'
                    print(f"  - {name}: {description}")
        else:
            print("❌ agent_registry not available")
        
        print()
        print("Current Investigation:")
        if investigation_manager and hasattr(investigation_manager, 'current_investigation') and investigation_manager.current_investigation is not None:
            inv = investigation_manager.current_investigation
            print(f"  Title: {inv.title if hasattr(inv, 'title') else 'N/A'}")
            print(f"  Status: {inv.status if hasattr(inv, 'status') else 'N/A'}")
            print(f"  Risk Level: {inv.risk_level if hasattr(inv, 'risk_level') else 'N/A'}")
        else:
            print("  No active investigation")
    except Exception as e:
        print(f"Error getting agent status: {str(e)}")

# =============================================================================
# SECTION 2: UNIFIED INVESTIGATION UI
# =============================================================================

class UnifiedInvestigationUI:
    """Unified UI for both agent selection and natural language investigations"""
    
    def __init__(self):
        self.last_run_summary = None
        self.setup_widgets()
        self.setup_layout()
        
        # Initialize help text for the default selected agent
        if self.agent_dropdown.value:
            self.on_agent_change({'new': self.agent_dropdown.value})
        
    def setup_widgets(self):
        """Initialize all UI widgets with enhanced styling"""
        
        # Section Header
        self.header = widgets.HTML(
            value="""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="margin: 0;">Unified Investigation Interface</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Choose between agent selection or natural language investigations</p>
            </div>
            """,
            layout=widgets.Layout(margin='0 0 20px 0')
        )
        
        # Create tabs for different modes
        self.setup_agent_selection_widgets()
        self.setup_natural_language_widgets()
        self.setup_dashboard_widgets()
        
        # Create the tab interface
        self.agent_tab = widgets.VBox([
            self.agent_dropdown,
            self.agent_help_text,
            self.query_input,
            self.template_button,
            self.advanced_toggle,
            self.advanced_params_box,
            self.run_button,
            self.stop_button,
            self.loading_indicator,
            self.last_run_box
        ])
        
        self.nl_tab = widgets.VBox([
            self.nl_query_input,
            self.nl_options_box,
            self.nl_buttons_box,
            self.nl_examples_box
        ])
        
        self.dashboard_tab = widgets.VBox([
            self.dashboard_refresh_button,
            self.dashboard_output
        ])
        
        # Create tab widget
        self.tab_widget = widgets.Tab()
        self.tab_widget.children = [self.agent_tab, self.nl_tab, self.dashboard_tab]
        self.tab_widget.set_title(0, 'Agent Selection')
        self.tab_widget.set_title(1, 'Natural Language')
        self.tab_widget.set_title(2, 'Agent Dashboard')
        
        # Shared output area
        self.output_area = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                border_radius='5px',
                padding='10px',
                margin='10px 0'
            )
        )
    
    def setup_agent_selection_widgets(self):
        """Setup widgets for agent selection mode"""
        
        # Agent selection with enhanced styling and error handling
        try:
            if agent_registry and hasattr(agent_registry, 'get_available_agents'):
                available_agents = agent_registry.get_available_agents()
                agent_options = [(f"{agent.replace('_', ' ').title()}", agent) for agent in available_agents]
            else:
                available_agents = []
                agent_options = [("No agents available", "none")]
        except Exception as e:
            available_agents = []
            agent_options = [("No agents available", "none")]
            print(f"Warning: Could not load agents: {e}")
        
        self.agent_dropdown = widgets.Dropdown(
            options=agent_options,
            description='Agent:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='350px', margin='0 0 10px 0')
        )
        self.agent_dropdown.observe(self.on_agent_change, names='value')
        
        # Query input with template support
        self.query_input = widgets.Textarea(
            placeholder='Enter your investigation query...',
            description='Query:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='700px', height='100px', margin='0 0 10px 0')
        )
        
        # Template button
        self.template_button = widgets.Button(
            description='Use Template',
            button_style='info',
            layout=widgets.Layout(width='150px', margin='0 0 15px 0'),
            tooltip='Fill query with a template for the selected agent'
        )
        self.template_button.on_click(self.use_query_template)
        
        # Dynamic help text for selected agent
        self.agent_help_text = widgets.HTML(
            value='<p style="color: #666; font-size: 0.9em; margin: 5px 0;">Select an agent to see its description and available templates.</p>',
            layout=widgets.Layout(margin='0 0 10px 0')
        )
        
        # Advanced parameters section (collapsible)
        self.advanced_toggle = widgets.ToggleButton(
            value=False,
            description='Advanced Parameters',
            button_style='',
            layout=widgets.Layout(width='200px', margin='0 0 10px 0')
        )
        self.advanced_toggle.observe(self.toggle_advanced_params, names='value')
        
        # Advanced parameter widgets
        self.days_back_input = widgets.IntText(
            value=7,
            description='Days Back:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='180px'),
            tooltip='Number of days to look back for data'
        )
        
        self.limit_input = widgets.IntText(
            value=100,
            description='Limit:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='180px'),
            tooltip='Maximum number of records to analyze'
        )
        
        self.source_filter_input = widgets.Text(
            placeholder='e.g., PlayAPI, VoiceAPI',
            description='Source Filter:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='250px'),
            tooltip='Filter by specific data source or API'
        )
        
        # Advanced parameters container (initially hidden)
        self.advanced_params_box = widgets.HBox([
            self.days_back_input,
            self.limit_input,
            self.source_filter_input
        ], layout=widgets.Layout(margin='10px 0', display='none'))
        
        # Execution section
        self.run_button = widgets.Button(
            description='Launch Investigation',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px', margin='20px 10px 20px 0'),
            tooltip='Execute the investigation with selected agent'
        )
        self.run_button.on_click(self.run_agent_investigation)
        
        # Stop button
        self.stop_button = widgets.Button(
            description='Stop Agent',
            button_style='danger',
            layout=widgets.Layout(width='150px', height='40px', margin='20px 0'),
            tooltip='Stop the currently running agent'
        )
        self.stop_button.on_click(self.stop_agent_execution)
        
        # Loading indicator with clearer feedback
        self.loading_indicator = widgets.Label(value="")
        
        # Last run summary box
        self.last_run_box = widgets.HTML(
            value='',
            layout=widgets.Layout(display='none', margin='10px 0')
        )
    
    def setup_natural_language_widgets(self):
        """Setup widgets for natural language mode"""
        
        # Natural language query input
        self.nl_query_input = widgets.Textarea(
            value='Find recent scam activity from PlayAPI users',
            placeholder='Enter your natural language investigation query...',
            description='Investigation Query:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='700px', height='100px', margin='0 0 10px 0')
        )
        
        # Options for natural language mode
        self.nl_show_data = widgets.Checkbox(
            value=True,
            description='Show data preview',
            style={'description_width': 'initial'}
        )
        
        self.nl_max_rows = widgets.IntSlider(
            value=10,
            min=5,
            max=50,
            step=5,
            description='Max rows to display:',
            style={'description_width': 'initial'}
        )
        
        self.nl_options_box = widgets.HBox([
            self.nl_show_data,
            self.nl_max_rows
        ], layout=widgets.Layout(margin='0 0 10px 0'))
        
        # Buttons for natural language mode
        self.nl_run_button = widgets.Button(
            description='Run Natural Language Investigation',
            button_style='primary',
            layout=widgets.Layout(width='300px', margin='0 10px 0 0')
        )
        self.nl_run_button.on_click(self.run_natural_language_investigation)
        
        self.nl_quick_scam_button = widgets.Button(
            description='Quick Scam Check',
            button_style='info',
            layout=widgets.Layout(width='150px', margin='0 10px 0 0')
        )
        self.nl_quick_scam_button.on_click(self.run_quick_scam)
        
        self.nl_status_button = widgets.Button(
            description='Agent Status',
            button_style='',
            layout=widgets.Layout(width='120px')
        )
        self.nl_status_button.on_click(self.show_agent_status)
        
        self.nl_buttons_box = widgets.HBox([
            self.nl_run_button,
            self.nl_quick_scam_button,
            self.nl_status_button
        ], layout=widgets.Layout(margin='0 0 15px 0'))
        
        # Example queries
        self.nl_examples_box = widgets.HTML(
            value="""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0;">Example Queries:</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Find recent scam activity from PlayAPI users</li>
                    <li>Detect investment scams in the last week</li>
                    <li>Look for suspicious financial schemes</li>
                    <li>Find fraudulent content with high confidence</li>
                </ul>
            </div>
            """,
            layout=widgets.Layout(margin='10px 0')
        )
    
    def setup_dashboard_widgets(self):
        """Setup widgets for the agent dashboard"""
        
        # Dashboard refresh button
        self.dashboard_refresh_button = widgets.Button(
            description='Refresh Dashboard',
            button_style='info',
            layout=widgets.Layout(width='200px', margin='0 0 10px 0')
        )
        self.dashboard_refresh_button.on_click(self.refresh_dashboard)
        
        # Dashboard output area
        self.dashboard_output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                border_radius='5px',
                padding='10px',
                margin='10px 0',
                min_height='400px'
            )
        )
    
    def setup_layout(self):
        """Create the main layout container with all UI components"""
        
        # Create main container with tabs and output area
        self.main_container = widgets.VBox([
            self.header,
            self.tab_widget,
            self.output_area
        ], layout=widgets.Layout(padding='20px', max_width='800px'))
    
    def display(self):
        """Display the UI interface"""
        from IPython.display import display
        display(self.main_container)
    
    def toggle_advanced_params(self, change):
        """Toggle advanced parameters visibility"""
        if change['new']:
            self.advanced_params_box.layout.display = 'block'
        else:
            self.advanced_params_box.layout.display = 'none'
    
    def use_query_template(self, button):
        """Fill query with template for selected agent"""
        # Basic template functionality
        agent_name = self.agent_dropdown.value
        if agent_name and agent_name != "none":
            template = f"Run {agent_name.replace('_', ' ')} investigation with default parameters"
            self.query_input.value = template
    
    def on_agent_change(self, change):
        """Update help text when agent selection changes"""
        agent_name = change['new']
        if agent_name and agent_name != "none":
            self.agent_help_text.value = f'<p style="color: #666; font-size: 0.9em; margin: 5px 0;">Selected: {agent_name.replace("_", " ").title()}</p>'
        else:
            self.agent_help_text.value = '<p style="color: #666; font-size: 0.9em; margin: 5px 0;">Select an agent to see its description.</p>'
    
    def stop_agent_execution(self, button):
        """Stop the currently running agent"""
        stop_agent()
        self.loading_indicator.value = "🛑 Agent stopped"
    
    def run_agent_investigation(self, button):
        """Execute the investigation with enhanced error handling"""
        with self.output_area:
            from IPython.display import clear_output
            clear_output()
            
            self.loading_indicator.value = "⏳ Running investigation..."
            
            try:
                agent_name = self.agent_dropdown.value
                query = self.query_input.value.strip()
                
                if not query:
                    print("❌ ERROR: Please enter an investigation query")
                    return
                    
                if agent_name == "none":
                    print("❌ ERROR: Please select an agent")
                    return
                
                if not agent_registry:
                    print("❌ ERROR: Agent registry not available")
                    return
                
                print(f"🚀 Launching {agent_name.replace('_', ' ').title()} investigation...")
                print(f"📋 Query: {query}")
                print(f"📅 Days back: {self.days_back_input.value}")
                print(f"🔢 Limit: {self.limit_input.value}")
                if self.source_filter_input.value:
                    print(f"🔍 Source filter: {self.source_filter_input.value}")
                
                # Run the investigation with enhanced monitoring
                if agent_registry and main_system:
                    result = execute_agent_with_monitoring(
                        agent_name,
                        lambda q, **p: agent_registry.run_agent(agent_name, q, **p),
                        query,
                        days_back=self.days_back_input.value,
                        limit=self.limit_input.value,
                        source_filter=self.source_filter_input.value or None
                    )
                    
                    # The conversational summary is now handled by execute_agent_with_monitoring
                    if result:
                        print(f"\n📊 Use the Agent Dashboard tab to see execution logs and status")
                    else:
                        print(f"\n❌ Investigation failed - see details above")
                        
                else:
                    print("\n⚠️ Agent system not fully initialized")
                    print("Some features may be limited")
                
            except Exception as e:
                print(f"❌ ERROR: Investigation failed: {str(e)}")
                update_agent_status(agent_name if 'agent_name' in locals() else 'unknown', "Error")
                import traceback
                traceback.print_exc()
            
            finally:
                self.loading_indicator.value = ""
    
    def run_natural_language_investigation(self, button):
        """Execute natural language investigation"""
        with self.output_area:
            from IPython.display import clear_output
            clear_output()
            
            try:
                query = self.nl_query_input.value.strip()
                if not query:
                    print("ERROR: Please enter an investigation query")
                    return
                
                result = run_natural_language_investigation(
                    query=query,
                    show_data=self.nl_show_data.value,
                    max_rows=self.nl_max_rows.value
                )
                
                if result:
                    print(f"\n📊 Use show_agent_dashboard() to see full details")
                
            except Exception as e:
                print(f"ERROR: Natural language investigation failed: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def run_quick_scam(self, button):
        """Run quick scam investigation"""
        with self.output_area:
            from IPython.display import clear_output
            clear_output()
            
            print("🔍 Running quick scam investigation...")
            try:
                if not agent_registry:
                    print("ERROR: Agent registry not available")
                    return
                
                result = execute_agent_with_monitoring(
                    "scam_agent",
                    lambda q, **p: agent_registry.run_agent("scam_agent", q, **p),
                    "Find recent scam activity from PlayAPI users in the past 3 days"
                )
                if result:
                    print("\n✅ Quick scam check completed!")
                else:
                    print("\n❌ Quick scam check failed")
            except Exception as e:
                print(f"ERROR: Quick scam check failed: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def show_agent_status(self, button):
        """Show agent system status"""
        with self.output_area:
            from IPython.display import clear_output
            clear_output()
            
            try:
                quick_agent_status()
                print("\n")
                show_agent_dashboard()
            except Exception as e:
                print(f"ERROR: Could not get agent status: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def refresh_dashboard(self, button):
        """Refresh the agent dashboard"""
        with self.dashboard_output:
            from IPython.display import clear_output
            clear_output()
            show_agent_dashboard()
    
    def setup_authentication(self, button):
        """Setup BigQuery authentication"""
        with self.schema_output:
            from IPython.display import clear_output
            clear_output()
            
            try:
                result = setup_bigquery_authentication()
                if result:
                    print("✅ Authentication setup successful")
                    print("💡 You can now try building the schema index")
                else:
                    print("⚠️  Authentication setup needs attention")
                    print("💡 Follow the instructions above to complete setup")
            except Exception as e:
                print(f"⚠️  Authentication setup failed: {e}")
                import traceback
                traceback.print_exc()

# =============================================================================
# INSTANTIATE AND DISPLAY THE UI
# =============================================================================

print("Creating Unified Investigation Interface...")
unified_ui = UnifiedInvestigationUI()

print("Displaying Unified Investigation Interface...")
unified_ui.display()

print("✅ Unified Investigation Interface is now active!")
print()
print("📋 Features available:")
print("   • Agent Selection tab: Choose specific agents with advanced parameters")
print("   • Natural Language tab: Use natural language queries with auto-intent detection")
print("   • Agent Dashboard tab: Real-time agent status, logs, and results")
print("   • Quick actions: Scam investigation, agent status, and more")
print("   • Agent stop control: Stop agents mid-execution")
print("   • Step-by-step logging: Track agent execution progress")
print()
print("🚀 Ready for investigations!")

# Initialize the dashboard
print("\n" + "="*80)
print("🎯 INITIAL AGENT DASHBOARD")
print("="*80)
show_agent_dashboard()

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def show_investigation_dashboard():
    """Backward compatibility function - shows the agent dashboard"""
    show_agent_dashboard()

print("\n✅ System initialization complete - all functions properly defined!")

# =============================================================================
# COMPREHENSIVE INTEGRATION OF ENHANCED SQL EXECUTOR
# =============================================================================

# Replace the existing SQLQueryExecutor globally and in all systems
def integrate_enhanced_sql_executor():
    """Integrate enhanced SQL executor into all system components"""
    
    # Check if we have the required components
    if 'bq_client' not in globals():
        print("⚠️  BigQuery client not available - cannot integrate enhanced SQL executor")
        return False
    
    try:
        # Create enhanced executor instance
        enhanced_executor = EnhancedSQLQueryExecutor(bq_client)
        
        # 1. Replace global sql_executor
        globals()['sql_executor'] = enhanced_executor
        print("✅ Global sql_executor replaced with EnhancedSQLQueryExecutor")
        
        # 2. Replace in main_system if it exists
        if 'main_system' in globals() and hasattr(main_system, 'sql_executor'):
            main_system.sql_executor = enhanced_executor
            print("✅ main_system.sql_executor replaced with EnhancedSQLQueryExecutor")
        
        # 3. Replace in __main__ namespace (critical for dependency injection)
        import __main__
        if hasattr(__main__, 'sql_executor'):
            setattr(__main__, 'sql_executor', enhanced_executor)
            print("✅ __main__.sql_executor replaced with EnhancedSQLQueryExecutor")
        
        # 4. Add the enhanced executor to globals for dependency injection
        __main__.sql_executor = enhanced_executor
        print("✅ EnhancedSQLQueryExecutor integrated into dependency injection system")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to integrate enhanced SQL executor: {e}")
        return False

# Perform the integration
print("🔧 Integrating enhanced SQL executor into all systems...")
integration_success = integrate_enhanced_sql_executor()

if integration_success:
    print("✅ Enhanced SQL executor successfully integrated")
else:
    print("⚠️  Enhanced SQL executor integration failed - some features may be limited")

# =============================================================================
# COMPREHENSIVE DETECTION MODE VALIDATION FIX
# =============================================================================

def validate_detection_mode_without_conflicts():
    """Validate detection mode without causing conflicts with existing systems"""
    
    print("🔍 Validating detection capabilities...")
    
    # Test Classification Flags table directly
    try:
        if 'bq_client' not in globals():
            print("❌ BigQuery client not available")
            return 'basic_scam_detection'
        
        # Try a simple query to the Classification Flags table
        test_query = """
        SELECT COUNT(*) as row_count 
        FROM `eleven-team-safety.trust_safety.content_analysis` 
        LIMIT 1
        """
        
        query_job = bq_client.query(test_query)
        result = query_job.result()
        
        print("✅ Classification Flags table accessible - Enhanced detection available")
        return 'enhanced_scam_detection'
        
    except Exception as e:
        error_msg = str(e)
        if "Unrecognized name" in error_msg:
            print(f"⚠️  Classification Flags table has schema issues: {error_msg}")
            print("→ Falling back to basic detection mode")
        elif "Not found" in error_msg:
            print(f"⚠️  Classification Flags table not found: {error_msg}")
            print("→ Falling back to basic detection mode")
        else:
            print(f"⚠️  Classification Flags table not accessible: {error_msg}")
        
        return 'basic_scam_detection'

# Replace the existing get_detection_mode function with the conflict-free version
def get_detection_mode():
    """Get detection mode with accurate validation and no conflicts"""
    return validate_detection_mode_without_conflicts()

# =============================================================================
# DYNAMIC SCHEMA DISCOVERY SYSTEM
# =============================================================================

# Global schema index - populated by scanning BigQuery
GLOBAL_SCHEMA_INDEX: pd.DataFrame = pd.DataFrame()
GLOBAL_LAST_UPDATED: Optional[datetime] = None

def build_global_schema_index(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build comprehensive schema index by scanning all available BigQuery datasets
    Returns:
        pd.DataFrame with columns: project, dataset, table, column, data_type
    """
    global GLOBAL_SCHEMA_INDEX, GLOBAL_LAST_UPDATED
    
    # Check if we need to refresh
    if not force_refresh and not GLOBAL_SCHEMA_INDEX.empty and GLOBAL_LAST_UPDATED:
        time_since_update = (datetime.now() - GLOBAL_LAST_UPDATED).total_seconds()
        if time_since_update < 300:  # 5 minutes cache
            print(f"✅ Using cached schema index ({len(GLOBAL_SCHEMA_INDEX)} columns)")
            return GLOBAL_SCHEMA_INDEX
    
    print("🔍 BUILDING DYNAMIC SCHEMA INDEX")
    print("=" * 60)
    
    # Projects to scan in order of preference
    try:
        credentials, default_project = default()
        projects_to_scan = [default_project] if default_project else []
        additional_projects = ["eleven-team-safety", "xi-labs", "analytics-dev-421514"]
        for project in additional_projects:
            if project not in projects_to_scan:
                projects_to_scan.append(project)
    except Exception:
        projects_to_scan = ["eleven-team-safety", "xi-labs", "analytics-dev-421514"]
    
    all_schema_data = []
    
    for project_id in projects_to_scan:
        print(f"\n📁 Scanning project: {project_id}")
        try:
            # Use service account authentication (Vertex AI) or default credentials
            try:
                client = bigquery.Client(project=project_id)
            except Exception as auth_error:
                print(f"   ⚠️  Default credentials failed: {auth_error}")
                print(f"   🔄 Trying with environment credentials...")
                client = bigquery.Client(project=project_id)
            # List all datasets in the project
            datasets = list(client.list_datasets(project=project_id))
            if not datasets:
                print(f"   ⚠️  No datasets found in project {project_id}")
                continue
            print(f"   📚 Found {len(datasets)} datasets")
            for dataset in datasets:
                dataset_id = dataset.dataset_id
                fq_dataset = f"{project_id}.{dataset_id}"
                print(f"   🔍 Querying {fq_dataset}.INFORMATION_SCHEMA.COLUMNS ...")
                try:
                    schema_query = f"""
                    SELECT 
                        '{project_id}' as project,
                        '{dataset_id}' as dataset,
                        table_name as table,
                        column_name as column,
                        data_type,
                        is_nullable
                    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                    ORDER BY table_name, ordinal_position
                    """
                    query_job = client.query(schema_query)
                    results = query_job.result()
                    dataset_schema = results.to_dataframe()
                    if not dataset_schema.empty:
                        all_schema_data.append(dataset_schema)
                        print(f"      ✅ {len(dataset_schema['table'].unique())} tables, {len(dataset_schema)} columns")
                    else:
                        print(f"      ⚠️  No tables found in dataset {dataset_id}")
                except Exception as e:
                    print(f"      ❌ Error querying {fq_dataset}: {str(e)}")
                    continue
        except Exception as e:
            print(f"   ❌ Error scanning {project_id}: {str(e)}")
            continue
    # Combine all project data
    if all_schema_data:
        GLOBAL_SCHEMA_INDEX = pd.concat(all_schema_data, ignore_index=True)
        GLOBAL_LAST_UPDATED = datetime.now()
        print(f"\n📊 SCHEMA INDEX SUMMARY:")
        print(f"   Total projects: {GLOBAL_SCHEMA_INDEX['project'].nunique()}")
        print(f"   Total datasets: {GLOBAL_SCHEMA_INDEX['dataset'].nunique()}")
        print(f"   Total tables: {GLOBAL_SCHEMA_INDEX['table'].nunique()}")
        print(f"   Total columns: {len(GLOBAL_SCHEMA_INDEX)}")
        tts_tables = GLOBAL_SCHEMA_INDEX[GLOBAL_SCHEMA_INDEX['table'].str.contains('tts|usage', case=False, na=False)]
        if not tts_tables.empty:
            print(f"   🎯 TTS/Usage tables: {tts_tables.groupby(['project', 'dataset', 'table']).size().head(3).to_dict()}")
        safety_tables = GLOBAL_SCHEMA_INDEX[GLOBAL_SCHEMA_INDEX['table'].str.contains('safety|analysis|classification', case=False, na=False)]
        if not safety_tables.empty:
            print(f"   🛡️  Safety tables: {safety_tables.groupby(['project', 'dataset', 'table']).size().head(3).to_dict()}")
        print(f"\n✅ Schema index built successfully")
        return GLOBAL_SCHEMA_INDEX
    else:
        print(f"\n❌ No schema data found - check project access and authentication")
        return pd.DataFrame()

def find_tables_by_schema(required_columns: List[str], preferred_keywords: List[str] = None) -> pd.DataFrame:
    """
    Find tables that contain all required columns
    
    Args:
        required_columns: List of column names that must be present
        preferred_keywords: List of keywords to prefer in table names
    
    Returns:
        DataFrame with matching tables and their metadata
    """
    global GLOBAL_SCHEMA_INDEX
    
    if GLOBAL_SCHEMA_INDEX.empty:
        print("⚠️  Schema index is empty - building now...")
        build_global_schema_index()
    
    if GLOBAL_SCHEMA_INDEX.empty:
        return pd.DataFrame()
    
    print(f"🔍 Finding tables with required columns: {required_columns}")
    
    # Group by table to check column coverage
    table_columns = GLOBAL_SCHEMA_INDEX.groupby(['project', 'dataset', 'table'])['column'].apply(list).reset_index()
    
    # Check which tables have all required columns
    matching_tables = []
    
    for _, row in table_columns.iterrows():
        available_columns = [col.lower() for col in row['column']]
        required_lower = [col.lower() for col in required_columns]
        
        # Check if all required columns are present
        has_all_columns = all(req_col in available_columns for req_col in required_lower)
        
        if has_all_columns:
            # Calculate match score
            score = 0
            
            # Bonus for preferred keywords
            if preferred_keywords:
                table_name_lower = row['table'].lower()
                for keyword in preferred_keywords:
                    if keyword.lower() in table_name_lower:
                        score += 10
            
            # Bonus for more columns (more complete table)
            score += len(row['column']) * 0.1
            
            matching_tables.append({
                'project': row['project'],
                'dataset': row['dataset'],
                'table': row['table'],
                'full_table_id': f"{row['project']}.{row['dataset']}.{row['table']}",
                'available_columns': row['column'],
                'column_count': len(row['column']),
                'match_score': score
            })
    
    if matching_tables:
        # Sort by match score (descending)
        matching_df = pd.DataFrame(matching_tables).sort_values('match_score', ascending=False)
        
        print(f"✅ Found {len(matching_df)} tables with required columns:")
        for _, match in matching_df.head(3).iterrows():
            print(f"   🎯 {match['full_table_id']} (score: {match['match_score']:.1f})")
            print(f"      Columns: {match['available_columns'][:8]}...")
        
        return matching_df
    else:
        print(f"❌ No tables found with required columns: {required_columns}")
        
        # Show what's available
        print(f"💡 Available columns in schema:")
        common_columns = GLOBAL_SCHEMA_INDEX['column'].value_counts().head(10)
        for col, count in common_columns.items():
            print(f"   • {col}: {count} tables")
        
        return pd.DataFrame()

def get_table_sample(table_id: str, limit: int = 5) -> pd.DataFrame:
    """
    Get a sample of data from a table to understand its structure
    
    Args:
        table_id: Full table ID (project.dataset.table)
        limit: Number of rows to sample
    
    Returns:
        DataFrame with sample data
    """
    try:
        # Try multiple authentication methods
        try:
            credentials, default_project = default()
            client = bigquery.Client(credentials=credentials, location="US")
        except Exception as auth_error:
            print(f"⚠️  Default credentials failed: {auth_error}")
            print(f"🔄 Trying with environment credentials...")
            client = bigquery.Client(location="US")
        
        query = f"SELECT * FROM `{table_id}` LIMIT {limit}"
        print(f"🔍 Sampling {limit} rows from {table_id}")
        
        query_job = client.query(query)
        result = query_job.result()
        
        df = result.to_dataframe()
        print(f"✅ Retrieved {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"❌ Error sampling {table_id}: {str(e)}")
        return pd.DataFrame()

def search_tables_by_keyword(keyword: str) -> pd.DataFrame:
    """
    Search for tables containing a specific keyword
    
    Args:
        keyword: Search term
    
    Returns:
        DataFrame with matching tables
    """
    global GLOBAL_SCHEMA_INDEX
    
    if GLOBAL_SCHEMA_INDEX.empty:
        build_global_schema_index()
    
    if GLOBAL_SCHEMA_INDEX.empty:
        return pd.DataFrame()
    
    # Search in table names and descriptions
    mask = (
        GLOBAL_SCHEMA_INDEX['table'].str.contains(keyword, case=False, na=False) |
        GLOBAL_SCHEMA_INDEX['description'].str.contains(keyword, case=False, na=False)
    )
    
    matching_tables = GLOBAL_SCHEMA_INDEX[mask]
    
    if not matching_tables.empty:
        # Group by table and show summary
        table_summary = matching_tables.groupby(['project', 'dataset', 'table']).size().reset_index(name='column_count')
        
        print(f"🔍 Tables matching '{keyword}':")
        for _, row in table_summary.head(10).iterrows():
            print(f"   🎯 {row['project']}.{row['dataset']}.{row['table']} ({row['column_count']} columns)")
        
        return table_summary
    else:
        print(f"❌ No tables found matching '{keyword}'")
        return pd.DataFrame()

# =============================================================================
# AUTHENTICATION HELPERS
# =============================================================================

def setup_bigquery_authentication():
    """
    Help user set up BigQuery authentication for different environments
    """
    print("🔐 BIGQUERY AUTHENTICATION SETUP")
    print("=" * 60)
    
    # Test current authentication status
    print("\n🧪 Testing current authentication...")
    
    try:
        from google.auth import default
        credentials, default_project = default()
        print(f"✅ Default credentials found")
        print(f"   Project: {default_project}")
        print(f"   Credentials type: {type(credentials).__name__}")
        
        # Test BigQuery access
        try:
            client = bigquery.Client(credentials=credentials, location="US")
            print(f"✅ BigQuery client created successfully")
            return True
        except Exception as bq_error:
            print(f"❌ BigQuery client creation failed: {bq_error}")
            
    except Exception as auth_error:
        print(f"❌ Authentication failed: {auth_error}")
    
    # Provide setup instructions
    print(f"\n📋 AUTHENTICATION SETUP INSTRUCTIONS:")
    print(f"=" * 60)
    
    print(f"\n🏠 For Local Development:")
    print(f"   1. Run: gcloud auth application-default login")
    print(f"   2. Follow the browser authentication flow")
    print(f"   3. Restart this notebook")
    
    print(f"\n☁️  For Vertex AI Notebooks:")
    print(f"   1. Authentication should be automatic via service account")
    print(f"   2. If failing, check IAM permissions for the service account")
    print(f"   3. Required roles: BigQuery Data Viewer, BigQuery Job User")
    
    print(f"\n🔑 For Service Account (Advanced):")
    print(f"   1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    print(f"   2. Point to your service account key file")
    print(f"   3. Ensure the service account has BigQuery access")
    
    print(f"\n🧪 Test Command:")
    print(f"   After setup, run: setup_bigquery_authentication()")
    
    return False

def test_project_access(project_id: str):
    """
    Test access to a specific BigQuery project
    
    Args:
        project_id: GCP project ID to test
    """
    print(f"🧪 Testing access to project: {project_id}")
    
    try:
        # Try multiple authentication methods
        try:
            credentials, default_project = default()
            client = bigquery.Client(project=project_id, credentials=credentials, location="US")
        except Exception as auth_error:
            print(f"   ⚠️  Default credentials failed, trying environment credentials...")
            client = bigquery.Client(project=project_id, location="US")
        
        # Test with a simple query
        test_query = f"SELECT 1 as test_value"
        query_job = client.query(test_query)
        result = query_job.result()
        
        print(f"✅ Project {project_id} is accessible")
        print(f"   Job ID: {query_job.job_id}")
        return True
        
    except Exception as e:
        print(f"❌ Project {project_id} access failed: {str(e)}")
        
        # Provide specific error guidance
        if "403" in str(e):
            print(f"   💡 Permission denied - check IAM roles")
        elif "404" in str(e):
            print(f"   💡 Project not found - verify project ID")
        elif "Reauthentication" in str(e):
            print(f"   💡 Run: gcloud auth application-default login")
        else:
            print(f"   💡 Check authentication and project access")
        
        return False

# Initialize schema index on startup
print("🚀 Schema index is not built by default in this cell.")
print("   Please run Cell 09 (schema builder) to build GLOBAL_SCHEMA_INDEX before using schema-dependent features.")

# =============================================================================
# SCHEMA-AWARE AGENT INTEGRATION
# =============================================================================

def find_agent_tables(agent_type: str, user_query: str = None) -> Dict[str, str]:
    """
    Find appropriate tables for a specific agent type based on required schema
    
    Args:
        agent_type: Type of agent ('scam_agent', 'exploratory_agent', 'email_network_agent')
        user_query: Optional user query to help with table selection
    
    Returns:
        Dictionary mapping table purposes to full table IDs
    """
    print(f"🔍 Finding tables for {agent_type}")
    
    # Define required columns for each agent type
    agent_requirements = {
        'scam_agent': {
            'primary_table': {
                'required_columns': ['text', 'timestamp', 'user_id', 'email'],
                'preferred_keywords': ['tts', 'usage', 'generation', 'content'],
                'purpose': 'TTS usage data for scam detection'
            },
            'classification_table': {
                'required_columns': ['id', 'timestamp', 'has_scam', 'scam_2'],
                'preferred_keywords': ['classification', 'analysis', 'safety', 'flags'],
                'purpose': 'Content classification flags'
            }
        },
        'exploratory_agent': {
            'primary_table': {
                'required_columns': ['text', 'timestamp', 'user_id'],
                'preferred_keywords': ['tts', 'usage', 'generation', 'content'],
                'purpose': 'TTS usage data for exploration'
            }
        },
        'email_network_agent': {
            'primary_table': {
                'required_columns': ['email', 'timestamp', 'user_id'],
                'preferred_keywords': ['tts', 'usage', 'generation', 'user'],
                'purpose': 'User activity data for network analysis'
            },
            'classification_table': {
                'required_columns': ['id', 'timestamp', 'has_scam'],
                'preferred_keywords': ['classification', 'analysis', 'safety'],
                'purpose': 'Content classification for risk scoring'
            }
        }
    }
    
    if agent_type not in agent_requirements:
        print(f"❌ Unknown agent type: {agent_type}")
        return {}
    
    agent_tables = {}
    requirements = agent_requirements[agent_type]
    
    for table_purpose, table_spec in requirements.items():
        print(f"\n📋 Looking for {table_purpose} ({table_spec['purpose']})")
        
        # Find tables matching the requirements
        matching_tables = find_tables_by_schema(
            required_columns=table_spec['required_columns'],
            preferred_keywords=table_spec['preferred_keywords']
        )
        
        if not matching_tables.empty:
            # Select the best matching table
            best_table = matching_tables.iloc[0]
            agent_tables[table_purpose] = best_table['full_table_id']
            
            print(f"   ✅ Selected: {best_table['full_table_id']}")
            print(f"   📊 Columns: {len(best_table['available_columns'])}")
        else:
            print(f"   ❌ No suitable table found for {table_purpose}")
            
            # For optional tables, continue without error
            if table_purpose == 'classification_table':
                print(f"   ⚠️  Classification table is optional - agent will use basic detection")
            else:
                print(f"   🚨 This is a required table - agent may fail")
    
    return agent_tables

def build_dynamic_query(agent_type: str, table_mapping: Dict[str, str], user_query: str, **params) -> str:
    """
    Build a dynamic query based on discovered tables and their actual schemas
    
    Args:
        agent_type: Type of agent
        table_mapping: Dictionary mapping table purposes to full table IDs
        user_query: User's query
        **params: Additional parameters (days_back, limit, etc.)
    
    Returns:
        SQL query string
    """
    print(f"🔍 Building dynamic query for {agent_type}")
    
    # Get parameters with defaults
    days_back = params.get('days_back', 7)
    limit = params.get('limit', 100)
    
    # Get actual column names from schema index
    def get_column_mapping(table_id: str) -> Dict[str, str]:
        """Get column mapping for a specific table"""
        table_columns = GLOBAL_SCHEMA_INDEX[
            GLOBAL_SCHEMA_INDEX['table'] == table_id.split('.')[-1]
        ]['column'].tolist()
        
        # Map logical column names to actual column names
        column_mapping = {}
        
        for col in table_columns:
            col_lower = col.lower()
            
            # Map common column patterns
            if 'user' in col_lower and 'id' in col_lower:
                column_mapping['user_id'] = col
            elif 'email' in col_lower:
                column_mapping['email'] = col
            elif 'text' in col_lower or 'content' in col_lower:
                column_mapping['text'] = col
            elif 'timestamp' in col_lower or 'created' in col_lower:
                column_mapping['timestamp'] = col
            elif col_lower in ['id', 'record_id', 'request_id']:
                column_mapping['id'] = col
            elif 'scam' in col_lower and ('has' in col_lower or 'is' in col_lower):
                column_mapping['has_scam'] = col
            elif 'scam' in col_lower and '2' in col_lower:
                column_mapping['scam_2'] = col
        
        return column_mapping
    
    # Build query based on agent type
    if agent_type == 'scam_agent':
        primary_table = table_mapping.get('primary_table')
        classification_table = table_mapping.get('classification_table')
        
        if not primary_table:
            return "-- ERROR: No primary table found for scam detection"
        
        # Get column mappings
        primary_cols = get_column_mapping(primary_table)
        
        # Build base query
        query = f"""
        WITH recent_activity AS (
            SELECT 
                {primary_cols.get('user_id', 'user_id')} as user_id,
                {primary_cols.get('email', 'email')} as email,
                {primary_cols.get('text', 'text')} as text,
                {primary_cols.get('timestamp', 'timestamp')} as timestamp,
                {primary_cols.get('id', 'id')} as id
            FROM `{primary_table}`
            WHERE DATE({primary_cols.get('timestamp', 'timestamp')}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                AND {primary_cols.get('text', 'text')} IS NOT NULL
                AND LENGTH({primary_cols.get('text', 'text')}) > 10
        )
        """
        
        # Add classification join if available
        if classification_table:
            classification_cols = get_column_mapping(classification_table)
            query += f"""
            , classified_content AS (
                SELECT 
                    {classification_cols.get('id', 'id')} as id,
                    {classification_cols.get('has_scam', 'has_scam')} as has_scam,
                    {classification_cols.get('scam_2', 'scam_2')} as scam_2
                FROM `{classification_table}`
                WHERE DATE({classification_cols.get('timestamp', 'timestamp')}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            )
            SELECT 
                ra.*,
                COALESCE(cc.has_scam, false) as has_scam,
                COALESCE(cc.scam_2, false) as scam_2
            FROM recent_activity ra
            LEFT JOIN classified_content cc ON ra.id = cc.id
            """
        else:
            query += """
            SELECT *,
                false as has_scam,
                false as scam_2
            FROM recent_activity
            """
        
        query += f"""
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
    elif agent_type == 'exploratory_agent':
        primary_table = table_mapping.get('primary_table')
        
        if not primary_table:
            return "-- ERROR: No primary table found for exploration"
        
        # Get column mappings
        primary_cols = get_column_mapping(primary_table)
        
        query = f"""
        SELECT 
            {primary_cols.get('user_id', 'user_id')} as user_id,
            {primary_cols.get('email', 'email')} as email,
            {primary_cols.get('text', 'text')} as text,
            {primary_cols.get('timestamp', 'timestamp')} as timestamp
        FROM `{primary_table}`
        WHERE DATE({primary_cols.get('timestamp', 'timestamp')}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
        ORDER BY {primary_cols.get('timestamp', 'timestamp')} DESC
        LIMIT {limit}
        """
        
    elif agent_type == 'email_network_agent':
        primary_table = table_mapping.get('primary_table')
        
        if not primary_table:
            return "-- ERROR: No primary table found for email network analysis"
        
        # Get column mappings
        primary_cols = get_column_mapping(primary_table)
        
        query = f"""
        SELECT 
            {primary_cols.get('email', 'email')} as email,
            {primary_cols.get('user_id', 'user_id')} as user_id,
            {primary_cols.get('timestamp', 'timestamp')} as timestamp,
            COUNT(*) as activity_count
        FROM `{primary_table}`
        WHERE DATE({primary_cols.get('timestamp', 'timestamp')}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
        GROUP BY {primary_cols.get('email', 'email')}, {primary_cols.get('user_id', 'user_id')}, {primary_cols.get('timestamp', 'timestamp')}
        ORDER BY activity_count DESC
        LIMIT {limit}
        """
        
    else:
        return f"-- ERROR: Unknown agent type: {agent_type}"
    
    print(f"✅ Generated dynamic query ({len(query)} characters)")
    return query

def execute_agent_with_dynamic_tables(agent_type: str, user_query: str, **params) -> Dict[str, Any]:
    """
    Execute an agent using dynamic table discovery
    
    Args:
        agent_type: Type of agent to execute
        user_query: User's query
        **params: Additional parameters
    
    Returns:
        Dictionary with execution results
    """
    print(f"🚀 Executing {agent_type} with dynamic table discovery")
    print(f"📋 Query: {user_query}")
    print(f"📊 Parameters: {params}")
    
    # Step 1: Find appropriate tables
    table_mapping = find_agent_tables(agent_type, user_query)
    
    if not table_mapping:
        return {
            'success': False,
            'error': f'No suitable tables found for {agent_type}',
            'explanation': f'The agent requires specific table schemas but none were found in the available datasets.',
            'suggestion': 'Check if the required datasets are accessible and contain the expected columns.'
        }
    
    # Step 2: Build dynamic query
    query = build_dynamic_query(agent_type, table_mapping, user_query, **params)
    
    if query.startswith('-- ERROR'):
        return {
            'success': False,
            'error': 'Query generation failed',
            'explanation': query,
            'suggestion': 'Check table schemas and column mappings.'
        }
    
    # Step 3: Execute query
    try:
        # Try multiple authentication methods
        try:
            credentials, default_project = default()
            client = bigquery.Client(credentials=credentials, location="US")
        except Exception as auth_error:
            print(f"   ⚠️  Default credentials failed: {auth_error}")
            print(f"   🔄 Trying with environment credentials...")
            client = bigquery.Client(location="US")
        
        print(f"🔍 Executing query...")
        query_job = client.query(query)
        results = query_job.result()
        
        # Convert to DataFrame
        df = results.to_dataframe()
        
        print(f"✅ Query executed successfully - {len(df)} rows returned")
        
        # Step 4: Generate conversational summary
        summary = generate_conversational_summary(agent_type, user_query, table_mapping, df, **params)
        
        return {
            'success': True,
            'data': df,
            'row_count': len(df),
            'tables_used': table_mapping,
            'query': query,
            'summary': summary,
            'job_id': query_job.job_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'explanation': f'Query execution failed: {str(e)}',
            'suggestion': 'Check BigQuery permissions and table access.',
            'query': query,
            'tables_attempted': table_mapping
        }

def generate_conversational_summary(agent_type: str, user_query: str, table_mapping: Dict[str, str], results_df: pd.DataFrame, **params) -> str:
    """
    Generate a conversational summary of agent execution
    
    Args:
        agent_type: Type of agent
        user_query: User's original query
        table_mapping: Tables that were used
        results_df: Query results
        **params: Additional parameters
    
    Returns:
        Conversational summary string
    """
    summary = f"""
🤖 {agent_type.replace('_', ' ').title()} Investigation Summary
{'=' * 60}

🗣️  Query: "{user_query}"
🔧 Parameters: {params.get('days_back', 7)} days back, limit {params.get('limit', 100)}

📊 Tables Used:
"""
    
    for purpose, table_id in table_mapping.items():
        summary += f"   • {purpose}: {table_id}\n"
    
    if not results_df.empty:
        summary += f"""
🔍 Results: {len(results_df)} records found

📈 Analysis:
"""
        
        # Show column summary
        for col in results_df.columns[:5]:  # Show first 5 columns
            unique_values = results_df[col].nunique()
            summary += f"   • {col}: {unique_values} unique values\n"
        
        # Show sample data
        summary += f"""
📝 Sample Results:
"""
        for i, row in results_df.head(3).iterrows():
            summary += f"   {i+1}. {dict(row)}\n"
            
    else:
        summary += """
🔍 Results: No records found

❓ Possible reasons:
   • Date range too narrow (try increasing days_back)
   • Query filters too restrictive
   • Tables contain no matching data
   
💡 Suggestions:
   • Try broader search terms
   • Increase the time range
   • Check if tables contain recent data
"""
    
    return summary

# =============================================================================
# ENHANCED AGENT WRAPPER WITH DYNAMIC DISCOVERY
# =============================================================================

def run_agent_with_dynamic_discovery(agent_type: str, user_query: str, **params):
    """
    Enhanced agent execution wrapper that uses dynamic table discovery
    
    Args:
        agent_type: Type of agent to run
        user_query: User's query
        **params: Additional parameters
    
    Returns:
        Agent execution results
    """
    print(f"🎯 Running {agent_type} with dynamic discovery")
    
    # Update agent status
    update_agent_status(agent_type, "Starting")
    log_agent_step(agent_type, f"Beginning dynamic table discovery for: {user_query}")
    
    try:
        # Execute with dynamic table discovery
        result = execute_agent_with_dynamic_tables(agent_type, user_query, **params)
        
        if result['success']:
            update_agent_status(agent_type, "Completed")
            log_agent_step(agent_type, f"Successfully processed {result['row_count']} records")
            
            # Store results
            store_agent_results(agent_type, result)
            
            # Print conversational summary
            print(result['summary'])
            
        else:
            update_agent_status(agent_type, "Failed")
            log_agent_step(agent_type, f"Failed: {result['error']}")
            
            # Print error summary
            print(f"\n❌ {agent_type.replace('_', ' ').title()} Execution Failed")
            print(f"🗣️  Error: {result['error']}")
            print(f"💡 Suggestion: {result['suggestion']}")
            
        return result
        
    except Exception as e:
        update_agent_status(agent_type, "Error")
        log_agent_step(agent_type, f"System error: {str(e)}")
        
        print(f"\n💥 {agent_type.replace('_', ' ').title()} System Error")
        print(f"🗣️  Error: {str(e)}")
        
        return {
            'success': False,
            'error': str(e),
            'explanation': f'System error during {agent_type} execution'
        }

# =============================================================================
# TESTING AND VALIDATION FUNCTIONS
# =============================================================================

def test_schema_discovery():
    """Test the dynamic schema discovery system"""
    print("🧪 TESTING DYNAMIC SCHEMA DISCOVERY SYSTEM")
    print("=" * 60)
    
    # Test 1: Build schema index
    print("\n1. Testing schema index building...")
    schema_df = build_global_schema_index(force_refresh=True)
    
    if not schema_df.empty:
        print(f"   ✅ Schema index built: {len(schema_df)} columns across {schema_df['table'].nunique()} tables")
    else:
        print(f"   ❌ Schema index empty")
        return False
    
    # Test 2: Search for TTS tables
    print("\n2. Testing TTS table search...")
    tts_tables = search_tables_by_keyword('tts')
    
    if not tts_tables.empty:
        print(f"   ✅ Found {len(tts_tables)} TTS-related tables")
    else:
        print(f"   ⚠️  No TTS tables found")
    
    # Test 3: Test agent table discovery
    print("\n3. Testing agent table discovery...")
    for agent_type in ['scam_agent', 'exploratory_agent', 'email_network_agent']:
        tables = find_agent_tables(agent_type)
        print(f"   {agent_type}: {len(tables)} tables found")
    
    # Test 4: Test column matching
    print("\n4. Testing column matching...")
    test_columns = ['text', 'timestamp', 'user_id', 'email']
    matching_tables = find_tables_by_schema(test_columns)
    
    if not matching_tables.empty:
        print(f"   ✅ Found {len(matching_tables)} tables with required columns")
    else:
        print(f"   ⚠️  No tables found with required columns")
    
    print(f"\n✅ Schema discovery system test completed")
    return True

def test_agent_execution():
    """Test agent execution with dynamic discovery"""
    print("🧪 TESTING AGENT EXECUTION WITH DYNAMIC DISCOVERY")
    print("=" * 60)
    
    # Test data
    test_cases = [
        ('scam_agent', 'Find scam activity in recent TTS generations'),
        ('exploratory_agent', 'Show recent TTS usage patterns'),
        ('email_network_agent', 'Analyze email networks for suspicious activity')
    ]
    
    for agent_type, query in test_cases:
        print(f"\n🔍 Testing {agent_type}")
        print(f"   Query: {query}")
        
        try:
            result = run_agent_with_dynamic_discovery(
                agent_type=agent_type,
                user_query=query,
                days_back=7,
                limit=10
            )
            
            if result['success']:
                print(f"   ✅ Success: {result['row_count']} rows returned")
            else:
                print(f"   ❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"   💥 Error: {str(e)}")
    
    print(f"\n✅ Agent execution test completed")

def show_schema_summary():
    """Show a summary of the current schema index"""
    global GLOBAL_SCHEMA_INDEX
    
    if GLOBAL_SCHEMA_INDEX.empty:
        print("⚠️  Schema index is empty - building now...")
        build_global_schema_index()
    
    if GLOBAL_SCHEMA_INDEX.empty:
        print("❌ No schema data available")
        return
    
    print("📊 SCHEMA INDEX SUMMARY")
    print("=" * 60)
    
    # Project summary
    print("\n🏢 Projects:")
    for project in GLOBAL_SCHEMA_INDEX['project'].unique():
        datasets = GLOBAL_SCHEMA_INDEX[GLOBAL_SCHEMA_INDEX['project'] == project]['dataset'].nunique()
        tables = GLOBAL_SCHEMA_INDEX[GLOBAL_SCHEMA_INDEX['project'] == project]['table'].nunique()
        print(f"   • {project}: {datasets} datasets, {tables} tables")
    
    # Dataset summary
    print("\n📁 Top Datasets:")
    dataset_summary = GLOBAL_SCHEMA_INDEX.groupby(['project', 'dataset']).size().sort_values(ascending=False).head(10)
    for (project, dataset), col_count in dataset_summary.items():
        table_count = GLOBAL_SCHEMA_INDEX[(GLOBAL_SCHEMA_INDEX['project'] == project) & (GLOBAL_SCHEMA_INDEX['dataset'] == dataset)]['table'].nunique()
        print(f"   • {project}.{dataset}: {table_count} tables, {col_count} columns")
    
    # Table summary
    print("\n📋 Interesting Tables:")
    
    # TTS related tables
    tts_mask = GLOBAL_SCHEMA_INDEX['table'].str.contains('tts|usage|generation', case=False, na=False)
    tts_tables = GLOBAL_SCHEMA_INDEX[tts_mask].groupby(['project', 'dataset', 'table']).size().head(5)
    if not tts_tables.empty:
        print("   🎯 TTS/Usage Tables:")
        for (project, dataset, table), col_count in tts_tables.items():
            print(f"      • {project}.{dataset}.{table} ({col_count} columns)")
    
    # Safety related tables
    safety_mask = GLOBAL_SCHEMA_INDEX['table'].str.contains('safety|analysis|classification|flag', case=False, na=False)
    safety_tables = GLOBAL_SCHEMA_INDEX[safety_mask].groupby(['project', 'dataset', 'table']).size().head(5)
    if not safety_tables.empty:
        print("   🛡️  Safety/Analysis Tables:")
        for (project, dataset, table), col_count in safety_tables.items():
            print(f"      • {project}.{dataset}.{table} ({col_count} columns)")
    
    # Common columns
    print("\n📊 Most Common Columns:")
    common_cols = GLOBAL_SCHEMA_INDEX['column'].value_counts().head(10)
    for col, count in common_cols.items():
        print(f"   • {col}: {count} tables")

# =============================================================================
# UPDATED NATURAL LANGUAGE INVESTIGATION
# =============================================================================

def run_natural_language_investigation_dynamic(query: str, show_data: bool = True, max_rows: int = 10):
    """
    Enhanced natural language investigation using dynamic table routing and brutally honest output
    """
    print("=" * 80)
    print(f"NATURAL LANGUAGE INVESTIGATION (DYNAMIC)")
    print("=" * 80)
    print(f"Query: {query}")
    print()
    # Step 1: Intent detection (simplified for now)
    print("Step 1: Intent Detection")
    print("-" * 30)
    query_lower = query.lower()
    if any(word in query_lower for word in ['scam', 'fraud', 'suspicious', 'malicious']):
        intent = 'tts_text_search'  # for demo, treat as text search
        print(f"✅ Detected intent: Scam/Text Search")
    elif any(word in query_lower for word in ['email', 'network', 'connection', 'user']):
        intent = 'user_tts_search'
        print(f"✅ Detected intent: User/Text Search")
    else:
        intent = 'tts_text_search'
        print(f"✅ Detected intent: Text Search")
    print(f"   Selected intent: {intent}")
    print()
    # Step 2: Dynamic agent execution with routing
    print("Step 2: Dynamic Agent Execution with Routing")
    print("-" * 30)
    result = run_agent_with_dynamic_routing(
        user_query=query,
        intent=intent,
        min_match=0.7,
        days_back=7,
        limit=max_rows
    )
    if result is not None and show_data and hasattr(result, 'head'):
        print(f"\n📊 Data Preview (first {max_rows} rows):")
        print(result.head(max_rows).to_string())
    elif result is None:
        print(f"❌ Investigation failed. See above for details.")
    return result

# Patch the NL investigation entry point
def run_natural_language_investigation(query: str, show_data: bool = True, max_rows: int = 10):
    return run_natural_language_investigation_dynamic(query, show_data, max_rows)

# --- UPGRADED AGENT EXECUTION WITH DYNAMIC ROUTING ---
def run_agent_with_dynamic_routing(user_query: str, intent: str = 'tts_text_search', min_match: float = 0.7, days_back: int = 7, limit: int = 10):
    """
    Launch an agent using dynamic table routing and brutally honest conversational output.
    """
    print("\n================ DYNAMIC AGENT EXECUTION ================")
    print(f"Query: {user_query}")
    # 1. Parse intent and required fields
    if intent == 'tts_text_search':
        required_fields = ['text', 'timestamp']
    elif intent == 'user_tts_search':
        required_fields = ['text', 'timestamp', 'user_id']
    else:
        print(f"❌ Unsupported intent: {intent}")
        update_agent_status(intent, "Error")
        log_agent_step(intent, f"Unsupported intent: {intent}")
        return None
    # 2. Route to best-fit table
    best_row, mapping, score = find_best_semantic_table(required_fields, min_match=min_match, verbose=True)
    if not best_row:
        msg = f"❌ No table found matching required fields: {required_fields}\n"
        msg += "   Please check your schema or provide a table/column hint."
        print(msg)
        update_agent_status(intent, "Failed")
        log_agent_step(intent, msg)
        return None
    if score < 0.7:
        msg = f"⚠️  Weak match ({int(score*100)}%). Please provide a table or column hint."
        print(msg)
        update_agent_status(intent, "Failed")
        log_agent_step(intent, msg)
        return None
    # 3. Build query
    table_id = f"{best_row['project']}.{best_row['dataset']}.{best_row['table']}"
    import re
    m = re.search(r'"([^"]+)"', user_query)
    if m:
        keyword = m.group(1)
    else:
        keyword = user_query.split()[-1]
    text_col = mapping.get('text')
    time_col = mapping.get('timestamp')
    user_col = mapping.get('user_id')
    select_cols = [c for c in [text_col, time_col, user_col] if c]
    if not select_cols:
        select_cols = ['*']
    select_expr = ', '.join(select_cols)
    where_clauses = []
    if text_col:
        where_clauses.append(f"LOWER({text_col}) LIKE '%{keyword.lower()}%'")
    if time_col:
        where_clauses.append(f"DATE({time_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)")
    where_sql = ' AND '.join(where_clauses) if where_clauses else '1=1'
    query = f"""
    SELECT {select_expr}
    FROM `{table_id}`
    WHERE {where_sql}
    ORDER BY {time_col if time_col else select_cols[0]} DESC
    LIMIT {limit}
    """
    print(f"✅ Generated query for {table_id} (keyword: '{keyword}')")
    # 4. Execute query
    try:
        credentials, default_project = default()
        client = bigquery.Client(credentials=credentials, location="US")
        print(f"🔍 Executing query...")
        query_job = client.query(query)
        results = query_job.result()
        df = results.to_dataframe()
        print(f"✅ Query executed successfully - {len(df)} rows returned")
        if not df.empty:
            print(df.head(limit).to_string(index=False))
            summary = f"\n✅ Investigation completed.\nUsing `{table_id}` because it contains: "
            for f, v in mapping.items():
                if v:
                    summary += f"{f} → `{v}`; "
                else:
                    summary += f"{f} → ❌; "
            summary += f"\nRows returned: {len(df)}."
        else:
            summary = f"No results found in `{table_id}` for keyword '{keyword}'.\n"
            summary += f"Columns used: " + ", ".join(select_cols)
        update_agent_status(intent, "Completed")
        log_agent_step(intent, summary)
        store_agent_results(intent, {'results': df, 'table': table_id, 'mapping': mapping, 'query': query})
        print(summary)
        return df
    except Exception as e:
        msg = f"❌ Query execution failed: {e}"
        print(msg)
        update_agent_status(intent, "Error")
        log_agent_step(intent, msg)
        return None

# --- SEMANTIC TABLE MATCHING AGENT INTEGRATION ---
COLUMN_ALIASES = {
    "text": ["text", "tts_input_text", "input_text"],
    "user_id": ["user_id", "userid", "user_uid", "user"],
    "timestamp": ["timestamp", "created", "generation_time"]
}

def semantic_field_match_score(table_columns, required_fields, alias_map):
    """
    Returns (score, mapping) where mapping is {field: matched_column or None}
    """
    mapping = {}
    match_count = 0
    for field in required_fields:
        found = None
        # Direct match
        for col in table_columns:
            if col.lower() == field.lower():
                found = col
                break
        # Alias match
        if not found:
            for alias in alias_map.get(field, []):
                for col in table_columns:
                    if alias.lower() == col.lower():
                        found = col
                        break
                if found:
                    break
        # Fuzzy/partial match
        if not found:
            for alias in alias_map.get(field, []):
                for col in table_columns:
                    if alias.lower() in col.lower() or col.lower() in alias.lower():
                        found = col
                        break
                if found:
                    break
        mapping[field] = found
        if found:
            match_count += 1
    score = match_count / len(required_fields)
    return score, mapping

def find_best_semantic_table(required_fields, min_match=0.6, verbose=True):
    global GLOBAL_SCHEMA_INDEX
    if GLOBAL_SCHEMA_INDEX.empty:
        if verbose:
            print("⚠️  Schema index is empty - building now...")
        build_global_schema_index()
    if GLOBAL_SCHEMA_INDEX.empty:
        if verbose:
            print("❌ No schema data available")
        return None, None, 0
    alias_map = {k: [a.lower() for a in v] for k, v in COLUMN_ALIASES.items()}
    table_columns = GLOBAL_SCHEMA_INDEX.groupby(['project', 'dataset', 'table'])['column'].apply(list).reset_index()
    best = None
    best_score = 0
    best_mapping = None
    all_candidates = []
    for _, row in table_columns.iterrows():
        score, mapping = semantic_field_match_score(row['column'], required_fields, alias_map)
        all_candidates.append((score, mapping, row))
        if score > best_score:
            best = row
            best_score = score
            best_mapping = mapping
    # Filter by min_match
    candidates = [(s, m, r) for (s, m, r) in all_candidates if s >= min_match]
    if not candidates:
        if verbose:
            print(f"❌ No table found matching required fields: {required_fields}")
            missing = {f for f in required_fields}
            for s, m, r in all_candidates:
                for f, v in m.items():
                    if v:
                        missing.discard(f)
            if missing:
                print(f"   No table contains: {', '.join(missing)}")
            print("   You may provide a table or column hint.")
        return None, None, 0
    # Pick best
    best_score, best_mapping, best_row = max(candidates, key=lambda x: x[0])
    if verbose:
        print(f"✅ Using `{best_row['project']}.{best_row['dataset']}.{best_row['table']}` (match: {int(best_score*100)}%) because it contains:")
        for f, v in best_mapping.items():
            if v:
                print(f"   {f} → `{v}`")
            else:
                print(f"   {f} → ❌ (not found)")
    return best_row, best_mapping, best_score

def agent_semantic_query(user_query, intent='tts_text_search', min_match=0.6, days_back=7, limit=10):
    """
    Full semantic agent: parse intent, match table, build and run query, explain everything.
    """
    print("\n================ SEMANTIC AGENT ===============")
    print(f"Query: {user_query}")
    # 1. Parse intent and required fields
    if intent == 'tts_text_search':
        required_fields = ['text', 'timestamp']
    elif intent == 'user_tts_search':
        required_fields = ['text', 'timestamp', 'user_id']
    else:
        print(f"❌ Unsupported intent: {intent}")
        return None
    # 2. Semantic table match
    best_row, mapping, score = find_best_semantic_table(required_fields, min_match=min_match, verbose=True)
    if not best_row:
        print("❌ No suitable table found for semantic agent.")
        return None
    if score < 0.7:
        print(f"⚠️  Weak match ({int(score*100)}%). Please provide a table or column hint.")
        return None
    # 3. Build query
    table_id = f"{best_row['project']}.{best_row['dataset']}.{best_row['table']}"
    # Extract keyword(s) from user_query (quoted string or last word)
    import re
    m = re.search(r'"([^"]+)"', user_query)
    if m:
        keyword = m.group(1)
    else:
        keyword = user_query.split()[-1]
    text_col = mapping.get('text')
    time_col = mapping.get('timestamp')
    user_col = mapping.get('user_id')
    select_cols = [c for c in [text_col, time_col, user_col] if c]
    if not select_cols:
        select_cols = ['*']
    select_expr = ', '.join(select_cols)
    where_clauses = []
    if text_col:
        where_clauses.append(f"LOWER({text_col}) LIKE '%{keyword.lower()}%'")
    if time_col:
        where_clauses.append(f"DATE({time_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)")
    where_sql = ' AND '.join(where_clauses) if where_clauses else '1=1'
    query = f"""
    SELECT {select_expr}
    FROM `{table_id}`
    WHERE {where_sql}
    ORDER BY {time_col if time_col else select_cols[0]} DESC
    LIMIT {limit}
    """
    print(f"✅ Generated semantic query for {table_id} (keyword: '{keyword}')")
    # 4. Execute query
    try:
        credentials, default_project = default()
        client = bigquery.Client(credentials=credentials, location="US")
        print(f"🔍 Executing query...")
        query_job = client.query(query)
        results = query_job.result()
        df = results.to_dataframe()
        print(f"✅ Query executed successfully - {len(df)} rows returned")
        if not df.empty:
            print(df.head(limit).to_string(index=False))
        else:
            print("No results found.")
        return df
    except Exception as e:
        print(f"❌ Query execution failed: {e}")
        return None
    print("\n================ END SEMANTIC AGENT ================\n")

# 💡 Smart Fallback Suggestion System (Dynamic Prompt Recovery)
def suggest_alternate_prompts(required_fields, original_query, max_suggestions=3):
    # Find partial matches in GLOBAL_SCHEMA_INDEX
    if 'GLOBAL_SCHEMA_INDEX' not in globals() or GLOBAL_SCHEMA_INDEX.empty:
        return ["(No schema available for suggestions)"]
    table_columns = GLOBAL_SCHEMA_INDEX.groupby(['project', 'dataset', 'table'])['column'].apply(list).reset_index()
    suggestions = []
    for _, row in table_columns.iterrows():
        columns = set([c.lower() for c in row['column']])
        matched = [f for f in required_fields if any(f.lower() == c or f.lower() in c for c in columns)]
        if matched and len(matched) < len(required_fields):
            # Suggest a prompt using only the matched fields
            prompt = f"Show me rows from {row['table']} with " + ", ".join(matched)
            suggestions.append(prompt)
        elif matched:
            # If all fields match, this is not a fallback
            continue
    # Fallback: suggest simpler prompts
    if not suggestions:
        for f in required_fields:
            suggestions.append(f"Show me all rows with {f}")
    # Try to rephrase the original query with only available fields
    available_fields = set(GLOBAL_SCHEMA_INDEX['column'].str.lower().unique())
    reduced_fields = [f for f in required_fields if f.lower() in available_fields]
    if reduced_fields and reduced_fields != required_fields:
        suggestions.insert(0, f"Try searching for: {', '.join(reduced_fields)}")
    # Limit to max_suggestions
    return suggestions[:max_suggestions]

# ================================================================
# 📦 Cell 08: Unified Investigation UI & Dynamic Schema (DYNAMIC SCHEMA MODE)
# WARNING: This cell uses DYNAMIC_SCHEMA_INDEX (a DataFrame) for dynamic schema discovery.
#          Do NOT run this cell in the same session as Cells 09/10 (restricted schema mode).
# ================================================================

import pandas as pd
from datetime import datetime
from google.cloud import bigquery
from google.auth import default
from typing import Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Dynamic Schema Index ---
DYNAMIC_SCHEMA_INDEX: pd.DataFrame = pd.DataFrame()
DYNAMIC_LAST_UPDATED: Optional[datetime] = None

def build_dynamic_schema_index(force_refresh: bool = False) -> pd.DataFrame:
    global DYNAMIC_SCHEMA_INDEX, DYNAMIC_LAST_UPDATED
    if not force_refresh and not DYNAMIC_SCHEMA_INDEX.empty and DYNAMIC_LAST_UPDATED:
        time_since_update = (datetime.now() - DYNAMIC_LAST_UPDATED).total_seconds()
        if time_since_update < 300:
            print(f"✅ Using cached schema index ({len(DYNAMIC_SCHEMA_INDEX)} columns)")
            return DYNAMIC_SCHEMA_INDEX
    print("🔍 BUILDING SCHEMA INDEX FOR HARDCODED TABLES")
    print("=" * 60)
    all_schema_data = []
    try:
        credentials, default_project = default()
        client = bigquery.Client(credentials=credentials)
    except Exception:
        client = bigquery.Client()
    for table_ref in ALLOWED_TABLES:
        try:
            table = client.get_table(table_ref)
            rows = []
            for field in table.schema:
                rows.append({
                    'project': table.project,
                    'dataset': table.dataset_id,
                    'table': table.table_id,
                    'column': field.name,
                    'data_type': field.field_type,
                    'is_nullable': field.is_nullable if hasattr(field, 'is_nullable') else None
                })
            if rows:
                df = pd.DataFrame(rows)
                all_schema_data.append(df)
                print(f"✅ Indexed: {table_ref} ({len(rows)} columns)")
            else:
                print(f"⚠️  No columns found in {table_ref}")
        except Exception as e:
            print(f"❌ Failed to index {table_ref}: {e}")
    if all_schema_data:
        DYNAMIC_SCHEMA_INDEX = pd.concat(all_schema_data, ignore_index=True)
        DYNAMIC_LAST_UPDATED = datetime.now()
        print(f"\n📊 SCHEMA INDEX SUMMARY:")
        print(f"   Total tables: {DYNAMIC_SCHEMA_INDEX['table'].nunique()}")
        print(f"   Total columns: {len(DYNAMIC_SCHEMA_INDEX)}")
        print(f"\n✅ Schema index built for hardcoded tables only.")
        return DYNAMIC_SCHEMA_INDEX
    else:
        print(f"\n❌ No schema data found - check table names and permissions.")
        return pd.DataFrame()

def show_dynamic_schema_summary():
    global DYNAMIC_SCHEMA_INDEX
    if DYNAMIC_SCHEMA_INDEX.empty:
        print("⚠️  Schema index is empty - building now...")
        build_dynamic_schema_index()
    if DYNAMIC_SCHEMA_INDEX.empty:
        print("❌ No schema data available")
        return
    print("📊 HARDCODED SCHEMA INDEX SUMMARY")
    print("=" * 60)
    print(f"\nTables:")
    for table in DYNAMIC_SCHEMA_INDEX['table'].unique():
        df = DYNAMIC_SCHEMA_INDEX[DYNAMIC_SCHEMA_INDEX['table'] == table]
        project = df['project'].iloc[0]
        dataset = df['dataset'].iloc[0]
        print(f"   • {project}.{dataset}.{table}: {len(df)} columns")
    print("\nMost Common Columns:")
    common_cols = DYNAMIC_SCHEMA_INDEX['column'].value_counts().head(10)
    for col, count in common_cols.items():
        print(f"   • {col}: {count} tables")

# --- Minimal UI Example (expand as needed) ---
class UnifiedInvestigationUI:
    def __init__(self):
        self.build_button = widgets.Button(description="Build Schema Index", button_style='primary')
        self.build_button.on_click(self.build_schema)
        self.schema_summary_button = widgets.Button(description="Show Schema Summary", button_style='info')
        self.schema_summary_button.on_click(self.show_schema_summary)
        self.output = widgets.Output()
        self.main = widgets.VBox([
            widgets.HTML(
                "<b>Unified Investigation UI (Hardcoded Tables Only)</b><br>"
                "<span style='color:red;'>Only the tables in ALLOWED_TABLES are available.</span>"
            ),
            self.build_button,
            self.schema_summary_button,
            self.output
        ])

    def build_schema(self, b):
        with self.output:
            clear_output()
            df = build_dynamic_schema_index(force_refresh=True)
            if not df.empty:
                print(f"✅ Schema index built: {len(df)} columns across {df['table'].nunique()} tables")
            else:
                print("❌ Failed to build schema index.")

    def show_schema_summary(self, b):
        with self.output:
            clear_output()
            show_dynamic_schema_summary()

    def display(self):
        display(self.main)

print("🚨 WARNING: Only the tables in ALLOWED_TABLES are available. No auto-discovery is performed.")
ui = UnifiedInvestigationUI()
ui.display()

# Cell 8a: Full Unified Investigation UI & Dynamic Table Router (HARD-CODED TABLES ONLY)
# ================================================================
# ✅ UI for agent workflows, dashboard, schema summary, NL investigation, and safe dynamic routing
# ================================================================

import difflib
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import re
from datetime import datetime, timedelta

# --- Dynamic Schema Index assertion ---
def assert_dynamic_schema_ready():
    if 'DYNAMIC_SCHEMA_INDEX' not in globals():
        raise RuntimeError("❌ DYNAMIC_SCHEMA_INDEX is not defined.")
    if not isinstance(DYNAMIC_SCHEMA_INDEX, pd.DataFrame):
        raise RuntimeError("❌ DYNAMIC_SCHEMA_INDEX must be a pandas DataFrame.")
    if DYNAMIC_SCHEMA_INDEX.empty:
        raise RuntimeError("❌ DYNAMIC_SCHEMA_INDEX is empty — run dynamic schema loader first.")

# --- Dynamic Table Router ---
def route_query_to_table_dynamic(required_fields, alias_map=None, min_match=None, verbose=True):
    assert_dynamic_schema_ready()
    if alias_map is None:
        alias_map = {}
    best_score = 0
    best_row = None
    best_mapping = {}
    for _, row in DYNAMIC_SCHEMA_INDEX.iterrows():
        schema_fields = set(row['fields']) if 'fields' in row else set(DYNAMIC_SCHEMA_INDEX[DYNAMIC_SCHEMA_INDEX['table'] == row['table']]['column'])
        field_mapping = {}
        match_count = 0
        for req_field in required_fields:
            if req_field in schema_fields:
                field_mapping[req_field] = req_field
                match_count += 1
                continue
            aliases = alias_map.get(req_field, [])
            for alias in aliases:
                if alias in schema_fields:
                    field_mapping[req_field] = alias
                    match_count += 1
                    break
            else:
                candidates = [f for f in schema_fields if req_field.lower() in f.lower() or f.lower() in req_field.lower()]
                if candidates:
                    closest = difflib.get_close_matches(req_field, candidates, n=1)
                    if closest:
                        field_mapping[req_field] = closest[0]
                        match_count += 1
        if verbose:
            print(f"🔍 {row['table']}: matched {match_count}/{len(required_fields)} fields")
        if min_match:
            if match_count >= min_match and match_count > best_score:
                best_score = match_count
                best_row = row
                best_mapping = field_mapping.copy()
        else:
            if match_count == len(required_fields):
                if verbose:
                    print(f"✅ Exact match: {row['table']}")
                return row, field_mapping, match_count
    if min_match and best_row is not None:
        if verbose:
            print(f"✅ Best partial match: {best_row['table']} ({best_score}/{len(required_fields)})")
        return best_row, best_mapping, best_score
    raise ValueError(f"❌ No table matched required fields: {required_fields}")

# --- Schema Summary ---
def show_dynamic_schema_summary():
    assert_dynamic_schema_ready()
    print("📊 HARDCODED SCHEMA INDEX SUMMARY")
    print("=" * 60)
    print(f"\nTables:")
    for table in DYNAMIC_SCHEMA_INDEX['table'].unique():
        df = DYNAMIC_SCHEMA_INDEX[DYNAMIC_SCHEMA_INDEX['table'] == table]
        project = df['project'].iloc[0]
        dataset = df['dataset'].iloc[0]
        print(f"   • {project}.{dataset}.{table}: {len(df)} columns")
    print("\nMost Common Columns:")
    common_cols = DYNAMIC_SCHEMA_INDEX['column'].value_counts().head(10)
    for col, count in common_cols.items():
        print(f"   • {col}: {count} tables")

# --- NL Query Parsing and Execution ---
def parse_nl_query(nl_query):
    # Simple parser: looks for table, keyword, and days
    table = None
    keyword = None
    days = 7
    m = re.search(r'from the ([\w_]+) table', nl_query)
    if m:
        table = m.group(1)
    m = re.search(r'words? ([\w\s]+) (are|is|were|was) mentioned', nl_query)
    if m:
        keyword = m.group(1).strip()
    m = re.search(r'past (\d+) days', nl_query)
    if m:
        days = int(m.group(1))
    return table, keyword, days

def run_nl_investigation(nl_query, alias_map=None, min_match=1, verbose=True):
    assert_dynamic_schema_ready()
    table_hint, keyword, days = parse_nl_query(nl_query)
    required_fields = ['text', 'timestamp']
    # Find best table (optionally filter by table_hint)
    candidates = DYNAMIC_SCHEMA_INDEX
    if table_hint:
        candidates = candidates[candidates['table'].str.contains(table_hint, case=False, na=False)]
    if candidates.empty:
        raise ValueError(f"No table found matching hint: {table_hint}")
    # Use the first candidate for demo (or use router for more complex logic)
    row = candidates.iloc[0]
    table_id = f"{row['project']}.{row['dataset']}.{row['table']}"
    text_col = 'text' if 'text' in candidates['column'].values else candidates['column'].iloc[0]
    time_col = 'timestamp' if 'timestamp' in candidates['column'].values else candidates['column'].iloc[0]
    # Simulate a query (in real use, run a real query or filter a DataFrame)
    print(f"Querying table: {table_id}")
    print(f"Keyword: {keyword}")
    print(f"Days back: {days}")
    # In a real system, you would run a query here. For demo, just print the info.
    return {
        'table': table_id,
        'keyword': keyword,
        'days': days,
        'text_col': text_col,
        'time_col': time_col
    }

# --- Full Unified Investigation UI ---
class UnifiedInvestigationUI:
    def __init__(self):
        self.build_widgets()
        self.setup_layout()
    def build_widgets(self):
        self.header = widgets.HTML(
            """
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
                <h2 style='margin: 0;'>Unified Investigation Interface (Hardcoded Tables Only)</h2>
                <p style='margin: 5px 0 0 0; opacity: 0.9;'>No auto-discovery. Only your specified tables are available.</p>
            </div>
            """
        )
        self.schema_summary_button = widgets.Button(description="Show Schema Summary", button_style='info')
        self.schema_summary_button.on_click(self.show_schema_summary)
        # --- Routing Test Block ---
        self.required_fields_input = widgets.Text(
            value='timestamp, user_id',
            description='Fields:',
            placeholder='Comma-separated required fields',
            layout=widgets.Layout(width='50%')
        )
        self.alias_map_input = widgets.Textarea(
            value='{"user_id": ["uid", "userId"]}',
            description='Alias map:',
            placeholder='JSON alias map',
            layout=widgets.Layout(width='50%', height='100px')
        )
        self.min_match_slider = widgets.IntSlider(
            value=1,
            min=0,
            max=10,
            step=1,
            description='Min match:',
            continuous_update=False
        )
        self.test_button = widgets.Button(description='Test Routing', button_style='success')
        self.test_button.on_click(self.on_test_route_click)
        self.routing_output = widgets.Output()
        self.schema_output = widgets.Output()
        # --- Agent Selection (table-based, only your tables) ---
        self.agent_dropdown = widgets.Dropdown(
            options=[(f"{row['project']}.{row['dataset']}.{row['table']}", row['table']) for _, row in DYNAMIC_SCHEMA_INDEX.drop_duplicates('table').iterrows()],
            description='Table:',
            layout=widgets.Layout(width='50%')
        )
        self.agent_query_input = widgets.Text(
            value='',
            description='Query:',
            placeholder='Enter your investigation query',
            layout=widgets.Layout(width='50%')
        )
        self.agent_run_button = widgets.Button(description='Run Agent', button_style='primary')
        self.agent_run_button.on_click(self.on_run_agent_click)
        self.agent_output = widgets.Output()
        # --- NL Investigation ---
        self.nl_query_input = widgets.Text(
            value='',
            description='NL Query:',
            placeholder='Enter a natural language query',
            layout=widgets.Layout(width='50%')
        )
        self.nl_run_button = widgets.Button(description='Run NL Investigation', button_style='primary')
        self.nl_run_button.on_click(self.on_run_nl_click)
        self.nl_output = widgets.Output()
    def setup_layout(self):
        self.main = widgets.VBox([
            self.header,
            widgets.HBox([
                self.schema_summary_button
            ]),
            widgets.HTML("<h4>Test Table Routing</h4>"),
            self.required_fields_input,
            self.alias_map_input,
            self.min_match_slider,
            self.test_button,
            self.routing_output,
            widgets.HTML("<h4>Agent Selection (Table-based)</h4>"),
            self.agent_dropdown,
            self.agent_query_input,
            self.agent_run_button,
            self.agent_output,
            widgets.HTML("<h4>Natural Language Investigation</h4>"),
            self.nl_query_input,
            self.nl_run_button,
            self.nl_output,
            widgets.HTML("<h4>Schema Summary</h4>"),
            self.schema_output
        ])
    def show_schema_summary(self, b=None):
        with self.schema_output:
            clear_output()
            show_dynamic_schema_summary()
    def on_test_route_click(self, _):
        with self.routing_output:
            clear_output()
            try:
                req_fields = [f.strip() for f in self.required_fields_input.value.split(',') if f.strip()]
                alias_map = eval(self.alias_map_input.value) if self.alias_map_input.value.strip() else {}
                row, mapping, score = route_query_to_table_dynamic(req_fields, alias_map=alias_map, min_match=self.min_match_slider.value)
                print(f"\n🎯 Best match: {row['table']}")
                print(f"Field mapping: {mapping}")
            except Exception as e:
                print(f"❌ Error: {e}")
    def on_run_agent_click(self, _):
        with self.agent_output:
            clear_output()
            try:
                table = self.agent_dropdown.value
                query = self.agent_query_input.value
                print(f"🚀 Running agent on table: {table}")
                print(f"Query: {query}")
                # Stub: In a real system, you would run your agent logic here
                print(f"(Stub) Agent executed successfully.")
            except Exception as e:
                print(f"❌ Error: {e}")
    def on_run_nl_click(self, _):
        with self.nl_output:
            clear_output()
            try:
                nl_query = self.nl_query_input.value
                result = run_nl_investigation(nl_query)
                print(f"\n✅ NL Investigation Result:")
                for k, v in result.items():
                    print(f"   {k}: {v}")
            except Exception as e:
                print(f"❌ Error: {e}")
    def display(self):
        display(self.main)

print("🚨 WARNING: Only the tables in DYNAMIC_SCHEMA_INDEX are available. No auto-discovery is performed.")
ui = UnifiedInvestigationUI()
ui.display()

# ... existing code ...
# Remove all legacy schema discovery UI and backend logic, and refactor to use only hardcoded tables (ALLOWED_TABLES/VERIFIED_TABLES)

# 1. Remove setup_schema_discovery_widgets and all schema discovery buttons from UnifiedInvestigationUI
# 2. Remove all references to GLOBAL_SCHEMA_INDEX, build_global_schema_index, and dynamic/discovery-based table routing
# 3. Refactor agent selection, schema summary, and routing to use only ALLOWED_TABLES or VERIFIED_TABLES
# 4. Remove any code that could trigger project/dataset scans or display tables not in the hardcoded list
# 5. Clean up unreachable/obsolete code

# --- At the top, import VERIFIED_TABLES from Cell 2 if available ---
try:
    import __main__
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
    if VERIFIED_TABLES is None:
        print("❌ VERIFIED_TABLES not found - run Cell 2 to verify tables.")
except Exception:
    VERIFIED_TABLES = None

# --- Helper: Get allowed table options for dropdowns ---
def get_allowed_table_options():
    if VERIFIED_TABLES:
        return [(f"{v['table_id']} ({k})", v['table_id']) for k, v in VERIFIED_TABLES.items() if v.get('accessible', True)]
    return []

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

# --- UnifiedInvestigationUI (refactored) ---
class UnifiedInvestigationUI:
    def __init__(self):
        self.header = widgets.HTML(
            """
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
                <h2 style='margin: 0;'>Unified Investigation Interface (Allowed Tables Only)</h2>
                <p style='margin: 5px 0 0 0; opacity: 0.9;'>No auto-discovery. Only your specified tables are available.</p>
            </div>
            """
        )
        # --- Agent Selection ---
        self.agent_dropdown = widgets.Dropdown(
            options=get_allowed_table_options(),
            description='Table:',
            layout=widgets.Layout(width='60%')
        )
        self.agent_query_input = widgets.Text(
            value='',
            description='Query:',
            placeholder='Enter your investigation query',
            layout=widgets.Layout(width='60%')
        )
        self.agent_run_button = widgets.Button(description='Run Agent', button_style='primary')
        self.agent_run_button.on_click(self.on_run_agent_click)
        self.agent_output = widgets.Output()
        # --- NL Investigation ---
        self.nl_query_input = widgets.Text(
            value='',
            description='NL Query:',
            placeholder='Enter a natural language query',
            layout=widgets.Layout(width='60%')
        )
        self.nl_run_button = widgets.Button(description='Run NL Investigation', button_style='primary')
        self.nl_run_button.on_click(self.on_run_nl_click)
        self.nl_output = widgets.Output()
        # --- Schema Summary ---
        self.schema_summary_button = widgets.Button(description="Show Schema Summary", button_style='info')
        self.schema_summary_button.on_click(self.show_schema_summary)
        self.schema_output = widgets.Output()
        # --- Layout ---
        self.main = widgets.VBox([
            self.header,
            widgets.HTML("<h4>Agent Selection (Allowed Tables Only)</h4>"),
            self.agent_dropdown,
            self.agent_query_input,
            self.agent_run_button,
            self.agent_output,
            widgets.HTML("<h4>Natural Language Investigation</h4>"),
            self.nl_query_input,
            self.nl_run_button,
            self.nl_output,
            widgets.HTML("<h4>Schema Summary</h4>"),
            self.schema_summary_button,
            self.schema_output
        ])
    def show_schema_summary(self, b=None):
        with self.schema_output:
            clear_output()
            show_allowed_schema_summary()
    def on_run_agent_click(self, _):
        with self.agent_output:
            clear_output()
            table_id = self.agent_dropdown.value
            query = self.agent_query_input.value.strip()
            if not table_id or not query:
                print("❌ Please select a table and enter a query.")
                return
            print(f"🚀 Running agent on {table_id}\nQuery: {query}")
            # Insert actual agent execution logic here, using only VERIFIED_TABLES
            # ...
            print("(Agent execution logic goes here)")
    def on_run_nl_click(self, _):
        with self.nl_output:
            clear_output()
            query = self.nl_query_input.value.strip()
            if not query:
                print("❌ Please enter a natural language query.")
                return
            print(f"🚀 Running NL investigation: {query}")
            # Insert actual NL investigation logic here, using only VERIFIED_TABLES
            # ...
            print("(NL investigation logic goes here)")
    def display(self):
        display(self.main)

print("🚨 WARNING: Only the tables in VERIFIED_TABLES are available. No auto-discovery is performed.")
ui = UnifiedInvestigationUI()
ui.display()
# ... existing code ...
