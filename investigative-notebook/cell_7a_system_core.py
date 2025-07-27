# ==============================================================
# CELL 07a: System Status Dashboard
# Purpose: Agent + system state visualizer
# Dependencies: Investigation manager, agent registry, runtime manager
# ==============================================================

# @title Cell 7a: System Core — Investigation Framework + Performance Monitor
# Core system diagnostics and investigation management framework

import os
import time
import json
from datetime import datetime
from google.cloud import bigquery
import pandas as pd

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# =============================================================================
# SYSTEM DIAGNOSTICS MODULE
# =============================================================================

class SystemDiagnostics:
    """System diagnostics and performance monitoring"""

    def __init__(self):
        self.start_time = time.time()
        self.token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def get_pdt_timestamp(self):
        """Get current PDT timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S PDT")

    def get_system_metrics(self):
        """Get system performance metrics"""
        metrics = {
            "timestamp": self.get_pdt_timestamp(),
            "uptime_seconds": time.time() - self.start_time
        }

        if PSUTIL_AVAILABLE:
            try:
                metrics.update({
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                    "disk_usage_percent": psutil.disk_usage('/').percent
                })
            except Exception as e:
                metrics["psutil_error"] = str(e)
        else:
            metrics["psutil_status"] = "not_available"

        return metrics

    def check_component_status(self):
        """Check status of core investigation components"""
        components = {
            "bigquery_analyzer": self._check_bigquery_analyzer(),
            "sql_executor": self._check_sql_executor(),
            "investigation_manager": self._check_investigation_manager(),
            "main_system": self._check_main_system(),
            "config": self._check_config()
        }
        return components

    def _check_bigquery_analyzer(self):
        """Check BigQuery analyzer component"""
        try:
            # Check for VERIFIED_TABLES in __main__ module (set by Cell 2)
            import __main__
            VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
            
            if VERIFIED_TABLES and len(VERIFIED_TABLES) > 0:
                accessible_tables = sum(1 for t in VERIFIED_TABLES.values() if t.get("accessible", False))
                total_tables = len(VERIFIED_TABLES)
                return {
                    "status": "healthy" if accessible_tables > 0 else "degraded",
                    "details": f"{accessible_tables}/{total_tables} tables accessible",
                    "last_check": self.get_pdt_timestamp()
                }
            else:
                return {"status": "error", "details": "VERIFIED_TABLES not found — run Cell 2 first", "last_check": self.get_pdt_timestamp()}
        except Exception as e:
            return {"status": "error", "details": f"BigQuery check failed: {str(e)}", "last_check": self.get_pdt_timestamp()}

    def _check_sql_executor(self):
        """Check SQL executor component"""
        try:
            if 'sql_executor' in globals() and sql_executor:
                return {"status": "healthy", "details": "SQL executor initialized", "last_check": self.get_pdt_timestamp()}
            else:
                return {"status": "error", "details": "SQL executor not initialized", "last_check": self.get_pdt_timestamp()}
        except Exception as e:
            return {"status": "error", "details": str(e), "last_check": self.get_pdt_timestamp()}

    def _check_investigation_manager(self):
        """Check investigation manager component"""
        try:
            if 'investigation_manager' in globals() and investigation_manager:
                return {"status": "healthy", "details": "Investigation manager ready", "last_check": self.get_pdt_timestamp()}
            else:
                return {"status": "error", "details": "Investigation manager not initialized", "last_check": self.get_pdt_timestamp()}
        except Exception as e:
            return {"status": "error", "details": str(e), "last_check": self.get_pdt_timestamp()}

    def _check_main_system(self):
        """Check main investigation system component"""
        try:
            if 'main_system' in globals() and main_system:
                analyzer_status = "available" if main_system.analyzer else "unavailable"
                return {"status": "healthy", "details": f"Main system ready, OpenAI analyzer: {analyzer_status}", "last_check": self.get_pdt_timestamp()}
            else:
                return {"status": "error", "details": "Main system not initialized", "last_check": self.get_pdt_timestamp()}
        except Exception as e:
            return {"status": "error", "details": str(e), "last_check": self.get_pdt_timestamp()}

    def _check_config(self):
        """Check configuration component"""
        try:
            openai_configured = 'OPENAI_API_KEY' in os.environ
            
            # Check for VERIFIED_TABLES in __main__ module (set by Cell 2)
            import __main__
            VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
            verified_tables_ready = VERIFIED_TABLES and len(VERIFIED_TABLES) > 0
            
            if openai_configured and verified_tables_ready:
                return {"status": "healthy", "details": "All credentials configured", "last_check": self.get_pdt_timestamp()}
            elif verified_tables_ready:
                return {"status": "degraded", "details": "GCP configured, OpenAI missing", "last_check": self.get_pdt_timestamp()}
            else:
                return {"status": "error", "details": "Missing critical configuration — run Cell 2 for BigQuery setup", "last_check": self.get_pdt_timestamp()}
        except Exception as e:
            return {"status": "error", "details": f"Config check failed: {str(e)}", "last_check": self.get_pdt_timestamp()}

# =============================================================================
# INVESTIGATION FRAMEWORK EXTENSION
# =============================================================================

class InvestigationFramework:
    """Extended investigation framework for system-level operations"""

    def __init__(self):
        self.current_investigation_id = None
        self.framework_start_time = time.time()

    def create_system_investigation(self, title, description, investigator_id="system"):
        """Create a new investigation using the core investigation manager"""
        try:
            if 'investigation_manager' in globals() and investigation_manager:
                # Use the existing investigation manager from Cell 3
                investigation = investigation_manager.create_investigation(
                    title=title,
                    description=description,
                    risk_level="medium"
                )
                
                if investigation:
                    self.current_investigation_id = investigation.investigation_id
                    print(f"SUCCESS: Investigation created: {investigation.investigation_id}")
                    return investigation.investigation_id
                else:
                    print("FAILED: Investigation creation returned None")
                    return None
            else:
                print("ERROR: Investigation manager not available")
                return None

        except Exception as e:
            print(f"ERROR: Failed to create investigation: {str(e)}")
            return None

    def log_system_action(self, action_type, details, notes=""):
        """Log a system action to the current investigation"""
        try:
            if not self.current_investigation_id:
                print("WARNING: No active investigation for logging")
                return False

            if 'investigation_manager' in globals() and investigation_manager:
                investigation_manager.add_action(
                    action_type,
                    f"{details} - {notes}" if notes else details
                )
                print(f"SUCCESS: Action logged: {action_type}")
                return True
            else:
                print("ERROR: Investigation manager not available")
                return False

        except Exception as e:
            print(f"ERROR: Logging failed: {str(e)}")
            return False

    def get_framework_status(self):
        """Get status of the investigation framework"""
        try:
            status = {
                "framework_uptime": time.time() - self.framework_start_time,
                "current_investigation": self.current_investigation_id,
                "timestamp": datetime.now().isoformat()
            }

            if 'investigation_manager' in globals() and investigation_manager:
                if investigation_manager.current_investigation:
                    inv = investigation_manager.current_investigation
                    status.update({
                        "investigation_title": inv.title,
                        "investigation_status": inv.status,
                        "investigation_findings": len(inv.findings),
                        "investigation_actions": len(inv.actions_taken)
                    })
                else:
                    status["investigation_details"] = "No active investigation"
            else:
                status["investigation_manager"] = "Not available"

            return status

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

# =============================================================================
# DISPLAY DASHBOARD
# =============================================================================

def show_system_status():
    """Display comprehensive system status dashboard"""
    print("Trust & Safety Investigation System Status")
    print("=" * 60)

    # Initialize diagnostics
    diagnostics = SystemDiagnostics()

    # System metrics
    print("System Performance:")
    metrics = diagnostics.get_system_metrics()
    print(f"   Timestamp: {metrics['timestamp']}")
    print(f"   Uptime: {metrics['uptime_seconds']:.1f} seconds")

    if PSUTIL_AVAILABLE and 'cpu_percent' in metrics:
        print(f"   CPU Usage: {metrics['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {metrics['memory_percent']:.1f}%")
        print(f"   Memory Available: {metrics['memory_available_gb']:.1f} GB")
        print(f"   Disk Usage: {metrics['disk_usage_percent']:.1f}%")
    else:
        print("   System metrics: psutil not available")

    # Component status
    print("\nComponent Health:")
    components = diagnostics.check_component_status()
    for name, status in components.items():
        status_icon = "SUCCESS" if status["status"] == "healthy" else "WARNING" if status["status"] == "degraded" else "ERROR"
        print(f"   {status_icon}: {name.replace('_', ' ').title()}")
        print(f"      Status: {status['status']}")
        print(f"      Details: {status['details']}")

    # Investigation system status
    print("\nInvestigation Framework:")
    
    if 'investigation_manager' in globals() and investigation_manager:
        if investigation_manager.current_investigation:
            inv = investigation_manager.current_investigation
            print(f"   Current Investigation: {inv.title}")
            print(f"   Status: {inv.status}")
            print(f"   Risk Level: {inv.risk_level}")
            print(f"   Findings: {len(inv.findings)}")
            print(f"   Actions: {len(inv.actions_taken)}")
        else:
            print("   Current Investigation: None")
    else:
        print("   Investigation Manager: Not initialized")

    # Check for VERIFIED_TABLES in __main__ module (set by Cell 2)
    import __main__
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
    if VERIFIED_TABLES:
        accessible_tables = sum(1 for t in VERIFIED_TABLES.values() if t.get("accessible", False))
        total_tables = len(VERIFIED_TABLES)
        print(f"   Data Sources: {accessible_tables}/{total_tables} tables accessible")
    else:
        print("   Data Sources: Not initialized — run Cell 2 first")

    # Main system status
    print("\nMain Investigation System:")
    if 'main_system' in globals() and main_system:
        analyzer_status = "Available" if main_system.analyzer else "Unavailable"
        print(f"   OpenAI Analyzer: {analyzer_status}")
        
        if main_system.analyzer:
            stats = main_system.analyzer.get_analysis_stats()
            print(f"   Total Analyzed: {stats['total_analyzed']}")
            if stats['total_analyzed'] > 0:
                print(f"   Token Usage: {stats['token_usage']['total']}")
    else:
        print("   Main System: Not initialized")

    # Environment status
    print("\nEnvironment:")
    print(f"   OpenAI API: {'Configured' if 'OPENAI_API_KEY' in os.environ else 'Missing'}")
    print(f"   SQL Executor: {'Active' if 'sql_executor' in globals() and sql_executor else 'Not initialized'}")

    print("\n" + "=" * 60)

    # Overall status
    # Check for VERIFIED_TABLES in __main__ module (set by Cell 2)
    import __main__
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
    
    components_ready = (
        'main_system' in globals() and main_system and
        'investigation_manager' in globals() and investigation_manager and
        'sql_executor' in globals() and sql_executor and
        VERIFIED_TABLES is not None
    )

    if components_ready:
        print("System Status: READY for investigations")
    else:
        print("System Status: Setup incomplete")

# =============================================================================
# WORKFLOW TESTING
# =============================================================================

def test_investigation_workflow():
    """Test the investigation framework with a sample workflow"""
    print("\nTesting Investigation Framework")
    print("=" * 40)

    # Initialize investigation framework
    framework = InvestigationFramework()

    # Create test investigation
    investigation_id = framework.create_system_investigation(
        title="System Framework Test",
        description="Testing investigation logging and framework components"
    )

    if investigation_id:
        # Log test actions
        framework.log_system_action(
            action_type="framework_test",
            details="Testing investigation logging functionality",
            notes="Automated system test"
        )

        # Get framework status
        status = framework.get_framework_status()
        if status and 'error' not in status:
            print(f"SUCCESS: Investigation framework test completed")
            print(f"   Investigation ID: {investigation_id}")
            print(f"   Framework uptime: {status['framework_uptime']:.1f} seconds")
            if 'investigation_findings' in status:
                print(f"   Findings: {status['investigation_findings']}")
                print(f"   Actions: {status['investigation_actions']}")
        else:
            print("WARNING: Framework status check incomplete")
    else:
        print("ERROR: Investigation framework test failed")

def run_system_diagnostics():
    """Run comprehensive system diagnostics"""
    print("\nRunning System Diagnostics")
    print("=" * 40)

    diagnostics = SystemDiagnostics()
    
    # Get system metrics
    metrics = diagnostics.get_system_metrics()
    print(f"System uptime: {metrics['uptime_seconds']:.1f} seconds")
    
    # Check all components
    components = diagnostics.check_component_status()
    
    healthy_count = sum(1 for c in components.values() if c['status'] == 'healthy')
    total_count = len(components)
    
    print(f"Component health: {healthy_count}/{total_count} healthy")
    
    # Show unhealthy components
    for name, status in components.items():
        if status['status'] != 'healthy':
            print(f"   Issue: {name} - {status['status']} - {status['details']}")
    
    if healthy_count == total_count:
        print("SUCCESS: All components healthy")
        return True
    else:
        print("WARNING: Some components need attention")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Initialize core components
system_diagnostics = SystemDiagnostics()
investigation_framework = InvestigationFramework()

# Display system status
show_system_status()

print("\nAvailable Functions:")
print("  - show_system_status() - Display system dashboard")
print("  - test_investigation_workflow() - Test investigation framework")
print("  - run_system_diagnostics() - Run comprehensive diagnostics")
print("  - investigation_framework.create_system_investigation(title, description)")
print("  - investigation_framework.log_system_action(action_type, details, notes)")
print("  - investigation_framework.get_framework_status()")

print("\n" + "=" * 60)
print("Cell 7a Complete - System Core Ready") 