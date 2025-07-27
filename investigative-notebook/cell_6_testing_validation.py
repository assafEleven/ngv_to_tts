# @title Cell 6: Testing & Validation

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

class SystemValidator:
    """Validates system components for Vertex AI compatibility"""

    def __init__(self):
        self.test_results = []
        self.validation_passed = True

    def validate_environment(self):
        """Validate environment setup"""
        print("VALIDATING: Environment setup...")

        # Test 1: Check imports
        try:
            import google.cloud.bigquery
            import pandas as pd
            import openai
            self.log_test("imports", True, "All required imports successful")
        except ImportError as e:
            self.log_test("imports", False, f"Import failed: {e}")

        # Test 2: Check main system initialization
        try:
            if 'main_system' in globals() and main_system:
                self.log_test("main_system", True, "Main investigation system initialized")
            else:
                self.log_test("main_system", False, "Main investigation system not available")
        except Exception as e:
            self.log_test("main_system", False, f"Main system error: {e}")

        # Test 3: Check BigQuery connection
        try:
            if 'VERIFIED_TABLES' in globals() and VERIFIED_TABLES:
                accessible_count = sum(1 for t in VERIFIED_TABLES.values() if t.get('accessible', False))
                self.log_test("bigquery_connection", True, f"BigQuery tables accessible: {accessible_count}/{len(VERIFIED_TABLES)}")
            else:
                self.log_test("bigquery_connection", False, "VERIFIED_TABLES not available")
        except Exception as e:
            self.log_test("bigquery_connection", False, f"BigQuery connection error: {e}")

        # Test 4: Check investigation manager
        try:
            if 'investigation_manager' in globals() and investigation_manager:
                self.log_test("investigation_manager", True, "Investigation manager ready")
            else:
                self.log_test("investigation_manager", False, "Investigation manager not available")
        except Exception as e:
            self.log_test("investigation_manager", False, f"Investigation manager error: {e}")

        # Test 5: Check SQL executor
        try:
            if 'sql_executor' in globals() and sql_executor:
                self.log_test("sql_executor", True, "SQL executor ready")
            else:
                self.log_test("sql_executor", False, "SQL executor not available")
        except Exception as e:
            self.log_test("sql_executor", False, f"SQL executor error: {e}")

    def validate_bigquery_tables(self):
        """Validate BigQuery table access"""
        print("VALIDATING: BigQuery table access...")

        if 'VERIFIED_TABLES' not in globals():
            self.log_test("bigquery_tables", False, "VERIFIED_TABLES not available")
            return

        for table_name, table_info in VERIFIED_TABLES.items():
            try:
                if table_info.get('accessible', False):
                    self.log_test(f"table_{table_name.replace(' ', '_').lower()}", True, f"Table accessible: {table_info['table_id']}")
                else:
                    self.log_test(f"table_{table_name.replace(' ', '_').lower()}", False, f"Table not accessible: {table_info['table_id']}")
            except Exception as e:
                self.log_test(f"table_{table_name.replace(' ', '_').lower()}", False, f"Table access error: {e}")

    def validate_openai_integration(self):
        """Validate OpenAI integration"""
        print("VALIDATING: OpenAI integration...")

        if 'main_system' not in globals() or not main_system or not main_system.analyzer:
            self.log_test("openai_integration", False, "OpenAI analyzer not available")
            return

        try:
            # Test content analysis with simple test data
            test_content_items = [{
                'text': 'This is a test message for OpenAI analysis',
                'user_id': 'test_user',
                'timestamp': datetime.now().isoformat()
            }]

            results = main_system.analyzer.analyze_content_batch(test_content_items)

            if results and len(results) > 0:
                result = results[0]
                self.log_test("openai_analysis", True, f"OpenAI analysis successful, risk_score: {result.risk_score}")
            else:
                self.log_test("openai_analysis", False, "OpenAI analysis failed or returned empty result")

        except Exception as e:
            self.log_test("openai_analysis", False, f"OpenAI analysis error: {e}")

    def validate_query_templates(self):
        """Validate SQL query templates"""
        print("VALIDATING: SQL query templates...")

        if 'INVESTIGATION_QUERY_TEMPLATES' not in globals():
            self.log_test("query_templates", False, "INVESTIGATION_QUERY_TEMPLATES not available")
            return

        try:
            templates = INVESTIGATION_QUERY_TEMPLATES
            for template_name, template_info in templates.items():
                if 'query' in template_info and template_info['query']:
                    self.log_test(f"template_{template_name}", True, f"Query template valid: {template_name}")
                else:
                    self.log_test(f"template_{template_name}", False, f"Query template invalid: {template_name}")
        except Exception as e:
            self.log_test("query_templates", False, f"Query template error: {e}")

    def run_functional_tests(self):
        """Run functional tests"""
        print("RUNNING: Functional tests...")

        # Test 1: Investigation creation
        try:
            if 'investigation_manager' in globals() and investigation_manager:
                test_investigation = investigation_manager.create_investigation(
                    title="Test Investigation",
                    description="System validation test",
                    risk_level="low"
                )

                if test_investigation:
                    self.log_test("investigation_creation", True, f"Investigation created: {test_investigation.investigation_id}")

                    # Test 2: Adding findings
                    try:
                        investigation_manager.add_finding(
                            "Test finding from validation",
                            "System validation test finding details"
                        )
                        self.log_test("finding_addition", True, "Finding added successfully")
                    except Exception as e:
                        self.log_test("finding_addition", False, f"Finding addition failed: {e}")

                    # Test 3: Adding actions
                    try:
                        investigation_manager.add_action(
                            "Test action from validation",
                            "System validation test action details"
                        )
                        self.log_test("action_addition", True, "Action added successfully")
                    except Exception as e:
                        self.log_test("action_addition", False, f"Action addition failed: {e}")

                    # Test 4: Update investigation status
                    try:
                        investigation_manager.update_status("completed")
                        self.log_test("status_update", True, "Investigation status updated successfully")
                    except Exception as e:
                        self.log_test("status_update", False, f"Status update failed: {e}")

                else:
                    self.log_test("investigation_creation", False, "Investigation creation returned None")
            else:
                self.log_test("investigation_creation", False, "Investigation manager not available")

        except Exception as e:
            self.log_test("investigation_creation", False, f"Investigation test error: {e}")

        # Test 5: SQL query execution
        try:
            if 'sql_executor' in globals() and sql_executor:
                # Test simple query execution
                result = sql_executor.execute_investigation_query("bulk_tts_usage", days=1, min_requests=1)
                if result is not None:
                    self.log_test("sql_query_execution", True, f"SQL query executed, returned {len(result)} rows")
                else:
                    self.log_test("sql_query_execution", False, "SQL query returned None")
            else:
                self.log_test("sql_query_execution", False, "SQL executor not available")
        except Exception as e:
            self.log_test("sql_query_execution", False, f"SQL query error: {e}")

    def test_comprehensive_investigation(self):
        """Test comprehensive investigation workflow"""
        print("TESTING: Comprehensive investigation workflow...")

        try:
            if 'main_system' in globals() and main_system:
                # Test suspicious content investigation
                result = main_system.run_comprehensive_investigation(
                    "suspicious_content",
                    days=1,
                    min_severity=1
                )

                if result and result.get('success', False):
                    self.log_test("comprehensive_investigation", True, f"Investigation completed, analyzed {result['summary']['data_records']} records")
                else:
                    self.log_test("comprehensive_investigation", False, "Comprehensive investigation failed")
            else:
                self.log_test("comprehensive_investigation", False, "Main system not available")
        except Exception as e:
            self.log_test("comprehensive_investigation", False, f"Comprehensive investigation error: {e}")

    def log_test(self, test_name: str, passed: bool, message: str):
        """Log test result"""
        status = "SUCCESS" if passed else "FAILED"
        self.test_results.append({
            'test_name': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        if not passed:
            self.validation_passed = False

        print(f"{status}: {test_name} - {message}")

    def generate_validation_report(self):
        """Generate validation report"""
        print(f"\nValidation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t['passed']])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")

        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test in self.test_results:
                if not test['passed']:
                    print(f"  - {test['test_name']}: {test['message']}")

        if self.validation_passed:
            print("\nVALIDATION PASSED - System ready for production use!")
        else:
            print("\nVALIDATION FAILED - Some issues need to be addressed")

        return self.validation_passed

def run_complete_validation():
    """Run complete system validation"""
    validator = SystemValidator()

    print("Starting complete system validation...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run all validation tests
    validator.validate_environment()
    validator.validate_bigquery_tables()
    validator.validate_openai_integration()
    validator.validate_query_templates()
    validator.run_functional_tests()
    validator.test_comprehensive_investigation()

    # Generate report
    return validator.generate_validation_report()

# Helper functions for quick testing
def test_content_analysis(content: str = "Test content for analysis"):
    """Quick test of content analysis"""
    if 'main_system' in globals() and main_system and main_system.analyzer:
        content_items = [{
            'text': content,
            'user_id': 'test_user',
            'timestamp': datetime.now().isoformat()
        }]
        
        results = main_system.analyzer.analyze_content_batch(content_items)
        if results:
            result = results[0]
            print(f"SUCCESS: Content analysis completed")
            print(f"  Risk Score: {result.risk_score}")
            print(f"  Risk Level: {result.risk_level}")
            print(f"  Flags: {result.flags}")
            print(f"  Reasoning: {result.reasoning}")
            return result
        else:
            print("FAILED: Content analysis returned no results")
            return None
    else:
        print("FAILED: Content analyzer not available")
        return None

def test_sql_query(query_type: str = "bulk_tts_usage", **params):
    """Quick test of SQL query execution"""
    if 'sql_executor' in globals() and sql_executor:
        try:
            result = sql_executor.execute_investigation_query(query_type, **params)
            if result is not None:
                print(f"SUCCESS: SQL query executed, returned {len(result)} rows")
                if len(result) > 0:
                    print(f"  Columns: {list(result.columns)}")
                    print(f"  Sample row: {result.iloc[0].to_dict()}")
                return result
            else:
                print("FAILED: SQL query returned None")
                return None
        except Exception as e:
            print(f"FAILED: SQL query error - {e}")
            return None
    else:
        print("FAILED: SQL executor not available")
        return None

def test_investigation_workflow():
    """Quick test of investigation workflow"""
    if 'investigation_manager' in globals() and investigation_manager:
        try:
            # Create test investigation
            investigation = investigation_manager.create_investigation(
                title="Quick Test Investigation",
                description="Testing investigation workflow",
                risk_level="low"
            )

            if investigation:
                # Add a test finding
                investigation_manager.add_finding(
                    "Test finding",
                    "This is a test finding for workflow validation"
                )

                # Add a test action
                investigation_manager.add_action(
                    "Test action",
                    "This is a test action for workflow validation"
                )

                # Update status
                investigation_manager.update_status("completed")

                print(f"SUCCESS: Investigation workflow test completed")
                print(f"  Investigation ID: {investigation.investigation_id}")
                print(f"  Status: {investigation.status}")
                print(f"  Findings: {len(investigation.findings)}")
                print(f"  Actions: {len(investigation.actions_taken)}")
                return investigation
            else:
                print("FAILED: Investigation creation returned None")
                return None
        except Exception as e:
            print(f"FAILED: Investigation workflow error - {e}")
            return None
    else:
        print("FAILED: Investigation manager not available")
        return None

def test_system_status():
    """Test system status display"""
    if 'show_system_status' in globals():
        try:
            show_system_status()
            print("SUCCESS: System status displayed")
            return True
        except Exception as e:
            print(f"FAILED: System status error - {e}")
            return False
    else:
        print("FAILED: show_system_status function not available")
        return False

def quick_system_check():
    """Quick system health check"""
    print("Quick System Health Check")
    print("=" * 40)
    
    # Check main components
    components = {
        'main_system': 'main_system' in globals() and main_system is not None,
        'investigation_manager': 'investigation_manager' in globals() and investigation_manager is not None,
        'sql_executor': 'sql_executor' in globals() and sql_executor is not None,
        'verified_tables': 'VERIFIED_TABLES' in globals() and VERIFIED_TABLES is not None,
        'query_templates': 'INVESTIGATION_QUERY_TEMPLATES' in globals() and INVESTIGATION_QUERY_TEMPLATES is not None
    }
    
    all_good = True
    for component, status in components.items():
        status_text = "AVAILABLE" if status else "MISSING"
        print(f"{component}: {status_text}")
        if not status:
            all_good = False
    
    print(f"\nOverall Status: {'HEALTHY' if all_good else 'ISSUES DETECTED'}")
    return all_good

print("Testing & Validation functions ready")
print("Available functions:")
print("  - run_complete_validation() - Run full system validation")
print("  - test_content_analysis(content) - Test OpenAI content analysis")
print("  - test_sql_query(query_type, **params) - Test SQL query execution")
print("  - test_investigation_workflow() - Test investigation workflow")
print("  - test_system_status() - Test system status display")
print("  - quick_system_check() - Quick health check")
print("")
print("Run run_complete_validation() to validate the entire system")
print("\nCell 6 Complete - Testing & Validation Ready") 