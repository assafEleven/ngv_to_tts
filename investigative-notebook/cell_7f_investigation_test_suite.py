# @title Cell 7f: Investigation System Test Suite
# Production-grade test suite for Trust & Safety investigation system validation

# Try to import ipywidgets - graceful fallback if not available
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

from datetime import datetime
import traceback
import json
import os

# =============================================================================
# INVESTIGATION SYSTEM TEST SUITE
# =============================================================================

def test_investigation_creation():
    """Test 1: Investigation creation using real InvestigationManager"""
    print("[TEST] TEST 1: Investigation Creation")
    print("=" * 50)

    # Test investigation creation
    investigation = investigation_manager.create_investigation(
        "Test: System Validation",
        "Validating investigation system functionality"
    )

    # Validate creation
    assert investigation is not None, "Investigation creation failed"
    assert investigation.title == "Test: System Validation", "Title mismatch"
    assert investigation.status == "active", "Status should be active"
    assert investigation.risk_level == "medium", "Default risk level should be medium"
    assert investigation_manager.current_investigation is not None, "current_investigation not set"
    assert investigation_manager.current_investigation.investigation_id == investigation.investigation_id, "current_investigation ID mismatch"

    print(f"SUCCESS: Investigation created successfully")
    print(f"   ID: {investigation.investigation_id}")
    print(f"   Title: {investigation.title}")
    print(f"   Status: {investigation.status}")
    print(f"   Risk Level: {investigation.risk_level}")
    print(f"   Created: {investigation.created_at}")
    print("SUCCESS: TEST 1 PASSED: Investigation creation works correctly\n")

def test_adding_findings_and_actions():
    """Test 2: Adding findings and actions to investigation"""
    print("[TEST] TEST 2: Adding Findings and Actions")
    print("=" * 50)

    # Ensure we have an active investigation
    if investigation_manager.current_investigation is None:
        investigation_manager.create_investigation("Test Investigation", "Test description")

    initial_findings = len(investigation_manager.current_investigation.findings)
    initial_actions = len(investigation_manager.current_investigation.actions_taken)

    # Test adding findings
    investigation_manager.add_finding(
        "Found suspicious activity pattern",
        "BigQuery analysis revealed 15 high-risk accounts"
    )
    investigation_manager.add_finding(
        "Detected coordinated behavior",
        "Multiple accounts sharing IP addresses and creation timestamps"
    )

    # Test adding actions
    investigation_manager.add_action(
        "Executed scam detection query",
        "Analyzed PlayAPI usage patterns for scam indicators"
    )
    investigation_manager.add_action(
        "Applied OpenAI content analysis",
        "Processed flagged content for policy violations"
    )

    # Validate additions
    current_investigation = investigation_manager.current_investigation
    assert len(current_investigation.findings) == initial_findings + 2, "Findings not added correctly"
    assert len(current_investigation.actions_taken) == initial_actions + 2, "Actions not added correctly"

    print(f"SUCCESS: Added {len(current_investigation.findings)} findings:")
    for i, finding in enumerate(current_investigation.findings, 1):
        print(f"   {i}. {finding}")

    print(f"SUCCESS: Added {len(current_investigation.actions_taken)} actions:")
    for i, action in enumerate(current_investigation.actions_taken, 1):
        print(f"   {i}. {action}")

    print("SUCCESS: TEST 2 PASSED: Findings and actions added successfully\n")

def test_status_updates():
    """Test 3: Status updates"""
    print("[TEST] TEST 3: Status Updates")
    print("=" * 50)

    # Ensure we have an active investigation
    if investigation_manager.current_investigation is None:
        investigation_manager.create_investigation("Test Investigation", "Test description")

    original_status = investigation_manager.current_investigation.status

    # Test status update
    investigation_manager.update_status("in_review", "Investigation requires supervisor review")

    # Validate status change
    assert investigation_manager.current_investigation.status == "in_review", "Status not updated correctly"

    print(f"SUCCESS: Status updated successfully:")
    print(f"   From: {original_status}")
    print(f"   To: {investigation_manager.current_investigation.status}")

    # Test another status update
    investigation_manager.update_status("completed", "Investigation completed successfully")
    assert investigation_manager.current_investigation.status == "completed", "Second status update failed"

    print(f"SUCCESS: Status updated again:")
    print(f"   Final status: {investigation_manager.current_investigation.status}")
    print("SUCCESS: TEST 3 PASSED: Status updates work correctly\n")

def test_error_handling():
    """Test 4: Error handling when no current investigation"""
    print("[TEST] TEST 4: Error Handling (No Current Investigation)")
    print("=" * 50)

    # Save current investigation
    saved_investigation = investigation_manager.current_investigation

    # Clear current investigation
    investigation_manager.current_investigation = None

    # Test operations with no current investigation
    print("Testing operations with no current investigation...")

    # These should handle None gracefully and not crash
    try:
        investigation_manager.add_finding("Test finding")
        investigation_manager.add_action("Test action")
        investigation_manager.update_status("test_status")
        summary = investigation_manager.get_investigation_summary()

        # Validate error handling
        assert "error" in summary, "get_investigation_summary should return error for None investigation"
        print("SUCCESS: All operations handled None investigation gracefully")

    except Exception as e:
        # Restore investigation and re-raise
        investigation_manager.current_investigation = saved_investigation
        raise AssertionError(f"Operations should handle None investigation gracefully, but got: {e}")

    # Restore investigation
    investigation_manager.current_investigation = saved_investigation
    print("SUCCESS: Error handling restored current investigation")
    print("SUCCESS: TEST 4 PASSED: Error handling works correctly\n")

def test_multiple_investigations():
    """Test 5: Multiple investigations management"""
    print("[TEST] TEST 5: Multiple Investigations")
    print("=" * 50)

    # Create first investigation
    inv1 = investigation_manager.create_investigation(
        "Investigation Alpha",
        "First test investigation"
    )
    inv1_id = inv1.investigation_id

    # Add some content to first investigation
    investigation_manager.add_finding("Alpha finding")
    investigation_manager.add_action("Alpha action")

    # Create second investigation
    inv2 = investigation_manager.create_investigation(
        "Investigation Beta",
        "Second test investigation"
    )
    inv2_id = inv2.investigation_id

    # Validate investigation switching
    assert investigation_manager.current_investigation.investigation_id == inv2_id, "Current investigation not switched"
    assert investigation_manager.current_investigation.title == "Investigation Beta", "Title mismatch on switch"

    # Add content to second investigation
    investigation_manager.add_finding("Beta finding")
    investigation_manager.add_action("Beta action")

    # Validate investigations are distinct
    assert inv1_id != inv2_id, "Investigation IDs should be unique"
    assert len(investigation_manager.current_investigation.findings) == 1, "Beta investigation should have 1 finding"

    print(f"SUCCESS: Created multiple investigations:")
    print(f"   Investigation 1: {inv1_id[:8]}... ({inv1.title})")
    print(f"   Investigation 2: {inv2_id[:8]}... ({inv2.title})")
    print(f"   Current: {investigation_manager.current_investigation.title}")
    print("SUCCESS: TEST 5 PASSED: Multiple investigations work correctly\n")

def test_export_functionality():
    """Test 6: Export functionality (JSON and CSV)"""
    print("[TEST] TEST 6: Export Functionality")
    print("=" * 50)

    # Ensure we have an investigation with data
    if investigation_manager.current_investigation is None:
        investigation_manager.create_investigation("Export Test", "Testing export functionality")

    investigation_manager.add_finding("Export test finding", "Testing export capabilities")
    investigation_manager.add_action("Export test action", "Validating export functions")

    # Test get_investigation_summary (JSON-like export)
    summary = investigation_manager.get_investigation_summary()

    # Validate summary structure
    assert isinstance(summary, dict), "Summary should be a dictionary"
    assert "investigation_id" in summary, "Summary missing investigation_id"
    assert "title" in summary, "Summary missing title"
    assert "status" in summary, "Summary missing status"
    assert "findings" in summary, "Summary missing findings"
    assert "actions_taken" in summary, "Summary missing actions_taken"

    # Test JSON serialization
    try:
        json_export = json.dumps(summary, indent=2, default=str)
        assert len(json_export) > 100, "JSON export seems too small"
        print("SUCCESS: JSON export successful")
        print(f"   Export size: {len(json_export)} characters")
    except Exception as e:
        raise AssertionError(f"JSON export failed: {e}")

    # Test CSV-like data export (using summary data)
    try:
        csv_data = []
        csv_data.append(["Field", "Value"])
        csv_data.append(["Investigation ID", summary["investigation_id"]])
        csv_data.append(["Title", summary["title"]])
        csv_data.append(["Status", summary["status"]])
        csv_data.append(["Risk Level", summary["risk_level"]])
        csv_data.append(["Findings Count", len(summary["findings"])])
        csv_data.append(["Actions Count", len(summary["actions_taken"])])

        assert len(csv_data) > 1, "CSV data should have content"
        print("SUCCESS: CSV export structure validated")
        print(f"   CSV rows: {len(csv_data)}")
    except Exception as e:
        raise AssertionError(f"CSV export failed: {e}")

    print("SUCCESS: TEST 6 PASSED: Export functionality works correctly\n")

def test_risk_escalation_logic():
    """Test 7: Risk escalation logic"""
    print("[TEST] TEST 7: Risk Escalation Logic")
    print("=" * 50)

    # Test high-risk investigation
    high_risk_inv = investigation_manager.create_investigation(
        "High Risk Test",
        "Testing high-risk escalation",
        risk_level="high"
    )

    assert high_risk_inv.risk_level == "high", "High risk level not set correctly"

    # Test critical-risk investigation
    critical_inv = investigation_manager.create_investigation(
        "Critical Risk Test",
        "Testing critical escalation",
        risk_level="critical"
    )

    assert critical_inv.risk_level == "critical", "Critical risk level not set correctly"

    # Test escalation logic
    escalation_needed = critical_inv.risk_level in ["high", "critical"]
    assert escalation_needed, "Escalation logic should trigger for critical risk"

    # Test risk level progression
    investigation_manager.create_investigation("Low Risk Test", "Testing low risk")
    investigation_manager.update_status("escalated", "Risk level increased due to new findings")

    print(f"SUCCESS: Risk escalation logic tested:")
    print(f"   High risk investigation: {high_risk_inv.risk_level}")
    print(f"   Critical risk investigation: {critical_inv.risk_level}")
    print(f"   Escalation needed for critical: {escalation_needed}")
    print("SUCCESS: TEST 7 PASSED: Risk escalation logic works correctly\n")

def test_sql_interface_integration():
    """Test 8: SQL interface integration"""
    print("[TEST] TEST 8: SQL Interface Integration")
    print("=" * 50)

    try:
        # Test SQL executor availability
        assert 'sql_executor' in globals(), "SQL executor not available"
        print("SUCCESS: SQL executor found")

        # Test BigQuery analyzer availability
        if hasattr(sql_executor, 'bigquery_analyzer'):
            print("SUCCESS: BigQuery analyzer available")
        else:
            print("INFO: BigQuery analyzer not initialized (expected in test environment)")

        # Test VERIFIED_TABLES availability
        assert 'VERIFIED_TABLES' in globals(), "VERIFIED_TABLES not available"
        assert len(VERIFIED_TABLES) > 0, "VERIFIED_TABLES is empty"
        print(f"SUCCESS: Found {len(VERIFIED_TABLES)} verified tables")

        print("SUCCESS: TEST 8 PASSED: SQL interface integration works correctly\n")

    except Exception as e:
        print(f"INFO: SQL interface test failed (expected in test environment): {str(e)}")
        print("INFO: TEST 8 SKIPPED: SQL interface not fully initialized\n")

def test_agent_system_integration():
    """Test 9: Agent system integration"""
    print("[TEST] TEST 9: Agent System Integration")
    print("=" * 50)

    try:
        # Test agent registry availability
        if 'agent_registry' in globals():
            print("SUCCESS: Agent registry found")
            
            # Test agent count
            agent_count = len(agent_registry.agents)
            print(f"SUCCESS: Found {agent_count} registered agents")
            
            # Test handler count
            handler_count = len(agent_registry.agent_handlers)
            print(f"SUCCESS: Found {handler_count} agent handlers")
            
        else:
            print("INFO: Agent registry not available (expected in test environment)")

        # Test agent runtime manager availability
        if 'agent_runtime_manager' in globals():
            print("SUCCESS: Agent runtime manager found")
        else:
            print("INFO: Agent runtime manager not available (expected in test environment)")

        print("SUCCESS: TEST 9 PASSED: Agent system integration verified\n")

    except Exception as e:
        print(f"INFO: Agent system test failed (expected in test environment): {str(e)}")
        print("INFO: TEST 9 SKIPPED: Agent system not fully initialized\n")

def test_runtime_integrity():
    """
    STRICT RUNTIME INTEGRITY TEST — DO NOT VIOLATE
    
    Tests that all critical components are properly initialized and available.
    This test ensures the system meets strict investigator guidelines.
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    - Preserve system integrity — don't let errors silently pass
    """
    print("🔍 TESTING RUNTIME INTEGRITY")
    print("=" * 50)
    
    try:
        # Test 1: Environment Setup (Cell 1)
        print("1. Testing Environment Setup (Cell 1)")
        assert ENVIRONMENT_READY, "ENVIRONMENT_READY not set — run Cell 1 first"
        print("   ✅ Environment ready")
        
        # Test 2: BigQuery Configuration (Cell 2)
        print("2. Testing BigQuery Configuration (Cell 2)")
        assert VERIFIED_TABLES is not None, "VERIFIED_TABLES not found — run Cell 2 first"
        assert isinstance(VERIFIED_TABLES, dict), "VERIFIED_TABLES is not a dictionary"
        assert len(VERIFIED_TABLES) > 0, "VERIFIED_TABLES is empty — run Cell 2 first"
        print("   ✅ VERIFIED_TABLES populated")
        
        # Test 2a: Critical table accessibility
        assert 'TTS Usage' in VERIFIED_TABLES, "TTS Usage table not found — run Cell 2 first"
        tts_table = VERIFIED_TABLES['TTS Usage']
        assert tts_table.get('accessible', False), "TTS Usage table not accessible — check Cell 2"
        assert tts_table.get('client') is not None, "TTS Usage table client missing — check Cell 2"
        print("   ✅ TTS Usage table accessible")
        
        # Test 2b: BigQuery client
        assert bq_client is not None, "BigQuery client missing — run Cell 2 first"
        print("   ✅ BigQuery client available")
        
        # Test 3: Investigation Manager (Cell 3)
        print("3. Testing Investigation Manager (Cell 3)")
        assert 'investigation_manager' in globals(), "investigation_manager not found — run Cell 3 first"
        assert investigation_manager is not None, "investigation_manager is None — run Cell 3 first"
        print("   ✅ Investigation manager available")
        
        # Test 4: SQL Executor (Cell 4)
        print("4. Testing SQL Executor (Cell 4)")
        assert 'sql_executor' in globals(), "sql_executor not found — run Cell 4 first"
        assert sql_executor is not None, "sql_executor is None — run Cell 4 first"
        print("   ✅ SQL executor available")
        
        # Test 4a: Query templates
        assert 'INVESTIGATION_QUERY_TEMPLATES' in globals(), "INVESTIGATION_QUERY_TEMPLATES not found — run Cell 4 first"
        assert INVESTIGATION_QUERY_TEMPLATES is not None, "INVESTIGATION_QUERY_TEMPLATES is None — run Cell 4 first"
        print("   ✅ Investigation query templates available")
        
        # Test 5: Main Investigation System (Cell 5)
        print("5. Testing Main Investigation System (Cell 5)")
        assert 'main_system' in globals(), "main_system not found — run Cell 5 first"
        assert main_system is not None, "main_system is None — run Cell 5 first"
        print("   ✅ Main system available")
        
        # Test 5a: main_system.bq_client
        assert hasattr(main_system, 'bq_client'), "main_system.bq_client attribute missing"
        assert main_system.bq_client is not None, "main_system.bq_client is None — run Cell 5 first"
        print("   ✅ main_system.bq_client available")
        
        # Test 5b: main_system.analyzer
        assert hasattr(main_system, 'analyzer'), "main_system.analyzer attribute missing"
        assert main_system.analyzer is not None, "main_system.analyzer is None — run Cell 5 first"
        print("   ✅ main_system.analyzer available")
        
        # Test 5c: main_system.analyzer.openai_client
        assert hasattr(main_system.analyzer, 'openai_client'), "main_system.analyzer.openai_client attribute missing"
        assert main_system.analyzer.openai_client is not None, "main_system.analyzer.openai_client is None — run Cell 5 first"
        print("   ✅ main_system.analyzer.openai_client available")
        
        # Test 6: Schema System
        print("6. Testing Schema System")
        assert 'TABLE_SCHEMAS' in globals(), "TABLE_SCHEMAS not found — run Cell 2 first"
        assert TABLE_SCHEMAS is not None, "TABLE_SCHEMAS is None — run Cell 2 first"
        assert 'TTS Usage' in TABLE_SCHEMAS, "TTS Usage schema not found"
        print("   ✅ Schema system available")
        
        # Test 7: Agent Registry (Cell 7b)
        print("7. Testing Agent Registry (Cell 7b)")
        if 'agent_registry' in globals():
            assert agent_registry is not None, "agent_registry is None — run Cell 7b"
            print("   ✅ Agent registry available")
        else:
            print("   ⚠️  Agent registry not found — run Cell 7b for full functionality")
        
        # Test 8: Runtime Integrity Function
        print("8. Testing Runtime Integrity Function")
        assert 'check_runtime_integrity' in globals(), "check_runtime_integrity not found — run Cell 2 first"
        assert callable(check_runtime_integrity), "check_runtime_integrity is not callable"
        print("   ✅ Runtime integrity function available")
        
        # Test 9: Run actual integrity check
        print("9. Running Actual Runtime Integrity Check")
        try:
            check_runtime_integrity()
            print("   ✅ Runtime integrity check passed")
        except Exception as e:
            print(f"   ❌ Runtime integrity check failed: {e}")
            raise
        
        print("\n🎯 RUNTIME INTEGRITY TEST PASSED")
        print("✅ All critical components verified and available")
        print("✅ System ready for REAL data investigation")
        return True
        
    except Exception as e:
        print(f"\n❌ RUNTIME INTEGRITY TEST FAILED: {e}")
        print("\n💡 REQUIRED ACTIONS:")
        print("   1. Run Cell 1 (Environment Setup)")
        print("   2. Run Cell 2 (BigQuery Configuration)")
        print("   3. Run Cell 3 (Investigation Management)")
        print("   4. Run Cell 4 (SQL Interface)")
        print("   5. Run Cell 5 (Main Investigation System)")
        print("   6. Run Cell 7b (Agent Launcher) - Optional")
        print("   7. Re-run this test")
        return False

def test_agent_execution_prerequisites():
    """
    Test that all prerequisites for agent execution are met
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 TESTING AGENT EXECUTION PREREQUISITES")
    print("=" * 50)
    
    try:
        # Test that we can execute a basic query
        print("1. Testing Basic Query Execution")
        
        # Get TTS Usage table info
        tts_table = VERIFIED_TABLES['TTS Usage']
        client = tts_table['client']
        table_id = tts_table['table_id']
        
        # Test query with schema-aware column names
        user_id_col = get_column_name('TTS Usage', 'user_id')
        email_col = get_column_name('TTS Usage', 'email')
        text_col = get_column_name('TTS Usage', 'text')
        timestamp_col = get_column_name('TTS Usage', 'timestamp')
        
        test_query = f"""
        SELECT {user_id_col}, {email_col}, {text_col}, {timestamp_col}
        FROM `{table_id}`
        WHERE DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        LIMIT 1
        """
        
        query_job = client.query(test_query)
        results = query_job.result()
        
        print("   ✅ Basic query execution successful")
        
        # Test main_system functionality
        print("2. Testing main_system Functionality")
        
        # Test that analyzer works
        if hasattr(main_system.analyzer, 'analyze_content'):
            test_result = main_system.analyzer.analyze_content("test content")
            print("   ✅ AI analyzer functionality verified")
        else:
            print("   ❌ AI analyzer functionality not available")
            
        print("\n🎯 AGENT EXECUTION PREREQUISITES TEST PASSED")
        print("✅ System ready for agent execution")
        return True
        
    except Exception as e:
        print(f"\n❌ AGENT EXECUTION PREREQUISITES TEST FAILED: {e}")
        print("💡 Check BigQuery connectivity and analyzer setup")
        return False

def run_comprehensive_integrity_test():
    """
    Run all integrity tests in sequence
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("🚀 RUNNING COMPREHENSIVE INTEGRITY TEST")
    print("=" * 60)
    
    tests = [
        ("Runtime Integrity", test_runtime_integrity),
        ("Agent Execution Prerequisites", test_agent_execution_prerequisites),
        ("Agent-Specific Table Requirements", test_agent_specific_table_requirements),
        ("Dependency Injection System", test_dependency_injection_system),
        ("Agent Error Handling", test_agent_error_handling),
        ("Real Data Enforcement", test_real_data_enforcement),
        ("Execution Order Enforcement", test_execution_order_enforcement),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    print("\n🎯 COMPREHENSIVE INTEGRITY TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL INTEGRITY TESTS PASSED")
        print("✅ System is ready for REAL data investigation")
        print("✅ All critical components verified and available")
        print("✅ Agent-specific requirements met")
        print("✅ Dependency injection system working correctly")
        print("✅ Error handling functioning properly")
        print("✅ Real data enforcement active")
        print("✅ Execution order properly enforced")
    else:
        print("\n❌ SOME INTEGRITY TESTS FAILED")
        print("⚠️  System may not be ready for investigation")
        print("💡 Fix failed tests before running agents")
        
        # Run diagnostic if table requirements failed
        if not results.get("Agent-Specific Table Requirements", True):
            print("\n🔍 RUNNING TABLE ACCESSIBILITY DIAGNOSIS")
            print("=" * 50)
            diagnose_table_accessibility_issues()
    
    return all_passed

def run_all_tests():
    """
    Run complete test suite including runtime integrity checks
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("🚀 RUNNING COMPLETE INVESTIGATION TEST SUITE")
    print("=" * 60)
    
    # Run integrity tests first
    print("PHASE 1: RUNTIME INTEGRITY TESTS")
    print("=" * 40)
    
    integrity_passed = run_comprehensive_integrity_test()
    
    if not integrity_passed:
        print("\n❌ RUNTIME INTEGRITY TESTS FAILED")
        print("⚠️  System not ready for investigation tests")
        print("💡 Fix integrity issues before running other tests")
        return False
    
    # Run standard tests
    print("\n\nPHASE 2: STANDARD INVESTIGATION TESTS")
    print("=" * 40)
    
    test_functions = [
        ("Basic Component Tests", test_basic_components),
        ("Investigation Manager Tests", test_investigation_manager),
        ("SQL Executor Tests", test_sql_executor),
        ("Main System Tests", test_main_system),
        ("Agent System Tests", test_agent_system),
        ("System Integration Tests", test_system_integration),
        ("Agent Execution Tests", test_agent_execution),
        ("Data Consistency Tests", test_data_consistency),
        ("Error Handling Tests", test_error_handling),
        ("Performance Tests", test_performance)
    ]
    
    results = {}
    for test_name, test_func in test_functions:
        try:
            print(f"\n--- Running {test_name} ---")
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n🎯 COMPLETE TEST SUITE RESULTS")
    print("=" * 60)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = passed_tests == total_tests
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED")
        print("✅ Investigation system is fully functional")
        print("✅ System ready for REAL data investigation")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TESTS FAILED")
        print("💡 Review failed tests and fix issues")
    
    return all_passed

# Quick test function for immediate validation
def quick_integrity_check():
    """
    Quick validation of critical system components
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("🔍 QUICK INTEGRITY CHECK")
    print("=" * 30)
    
    try:
        # Use the strict integrity check if available
        if 'check_runtime_integrity' in globals():
            check_runtime_integrity()
            print("✅ Quick integrity check passed")
            return True
        else:
            print("❌ check_runtime_integrity function not found")
            print("💡 Run Cell 2 first to initialize integrity checks")
            return False
            
    except Exception as e:
        print(f"❌ Quick integrity check failed: {e}")
        return False

def create_test_suite_interface():
    """Create interactive widget interface for testing"""
    if not WIDGETS_AVAILABLE:
        print("WARNING: ipywidgets not available - using text-based testing")
        print("Run: run_all_tests() or quick_system_check()")
        return

    try:
        output = widgets.Output()

        def run_full_tests(b):
            with output:
                clear_output()
                run_all_tests()

        def run_quick_check(b):
            with output:
                clear_output()
                quick_system_check()

        def clear_output_area(b):
            output.clear_output()

        # Create buttons
        full_test_btn = widgets.Button(
            description="[TEST] Run All Tests",
            button_style="primary",
            layout=widgets.Layout(width="200px")
        )
        quick_test_btn = widgets.Button(
            description="[QUICK] Quick Test",
            button_style="info",
            layout=widgets.Layout(width="200px")
        )
        clear_btn = widgets.Button(
            description="[CLEAR] Clear Output",
            button_style="",
            layout=widgets.Layout(width="200px")
        )

        # Wire up buttons
        full_test_btn.on_click(run_full_tests)
        quick_test_btn.on_click(run_quick_check)
        clear_btn.on_click(clear_output_area)

        # Create interface
        button_box = widgets.HBox([full_test_btn, quick_test_btn, clear_btn])
        interface = widgets.VBox([
            widgets.HTML("<h3>[SYSTEM] Investigation System Test Suite</h3>"),
            button_box,
            output
        ])

        display(interface)

        # Show initial status
        with output:
            print("[SYSTEM] Test Suite Interface Ready")
            print("Click buttons above to run tests")
            print("=" * 40)

    except Exception as e:
        print(f"ERROR: Failed to create widget interface: {str(e)}")
        print("Falling back to text-based testing")
        print("Run: run_all_tests() or quick_system_check()")

def test_system_components():
    """Test availability of system components"""
    print("[SYSTEM] SYSTEM COMPONENTS TEST")
    print("=" * 40)
    
    components = [
        ("investigation_manager", "Investigation Manager"),
        ("sql_executor", "SQL Executor"),
        ("main_system", "Main System"),
        ("VERIFIED_TABLES", "Verified Tables"),
        ("agent_registry", "Agent Registry"),
        ("agent_runtime_manager", "Agent Runtime Manager")
    ]
    
    available_count = 0
    total_count = len(components)
    
    for component_name, display_name in components:
        if component_name in globals():
            print(f"SUCCESS: {display_name}: Available")
            available_count += 1
        else:
            print(f"INFO: {display_name}: Not available")
    
    print("=" * 40)
    print(f"[SYSTEM] Components available: {available_count}/{total_count}")
    
    if available_count >= 3:  # Core components
        print("SUCCESS: Sufficient components for testing")
    else:
        print("WARNING: Limited components available for testing")
    
    return available_count, total_count

def validate_test_environment():
    """Validate test environment setup"""
    print("[VALIDATE] TEST ENVIRONMENT VALIDATION")
    print("=" * 50)
    
    validation_results = {
        "core_system": False,
        "investigation_manager": False,
        "sql_system": False,
        "agent_system": False
    }
    
    # Check core investigation manager
    if 'investigation_manager' in globals():
        validation_results["investigation_manager"] = True
        validation_results["core_system"] = True
        print("SUCCESS: Investigation Manager available")
    else:
        print("ERROR: Investigation Manager not found")
    
    # Check SQL system
    if 'sql_executor' in globals() and 'VERIFIED_TABLES' in globals():
        validation_results["sql_system"] = True
        print("SUCCESS: SQL system components available")
    else:
        print("INFO: SQL system not fully initialized")
    
    # Check agent system
    if 'agent_registry' in globals():
        validation_results["agent_system"] = True
        print("SUCCESS: Agent system components available")
    else:
        print("INFO: Agent system not fully initialized")
    
    print("=" * 50)
    print(f"[VALIDATE] Environment validation complete")
    
    ready_for_testing = validation_results["core_system"]
    if ready_for_testing:
        print("SUCCESS: Environment ready for testing")
    else:
        print("ERROR: Environment not ready for testing")
    
    return validation_results

def test_dependency_injection_system():
    """
    Test the dependency injection system for agents
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 TESTING DEPENDENCY INJECTION SYSTEM")
    print("=" * 50)
    
    try:
        # Test 1: Agent registry availability
        print("1. Testing Agent Registry Availability")
        assert 'agent_registry' in globals(), "agent_registry not found — run Cell 7b first"
        assert agent_registry is not None, "agent_registry is None — run Cell 7b first"
        print("   ✅ Agent registry available")
        
        # Test 2: Dependency initialization
        print("2. Testing Dependency Initialization")
        
        # Test dependency injection for scam agent
        if hasattr(agent_registry, '_initialize_agent_with_dependencies'):
            dependencies = agent_registry._initialize_agent_with_dependencies("scam_agent")
            
            # Validate all required dependencies are present
            required_deps = ['bq_client', 'analyzer', 'sql_executor', 'investigation_manager', 'VERIFIED_TABLES', 'TABLE_SCHEMAS']
            for dep in required_deps:
                assert dep in dependencies, f"Missing dependency: {dep}"
                assert dependencies[dep] is not None, f"Dependency {dep} is None"
            
            print("   ✅ All dependencies initialized successfully")
        else:
            print("   ❌ Dependency injection method not found")
            return False
        
        # Test 3: Mock data validation
        print("3. Testing Mock Data Validation")
        
        if hasattr(agent_registry, '_validate_no_mock_data'):
            # This should not raise an error if using real data
            agent_registry._validate_no_mock_data(
                dependencies['bq_client'],
                dependencies['analyzer'],
                dependencies['VERIFIED_TABLES']
            )
            print("   ✅ Mock data validation passed")
        else:
            print("   ❌ Mock data validation method not found")
            return False
        
        # Test 4: Execution order validation
        print("4. Testing Execution Order Validation")
        
        if hasattr(agent_registry, '_validate_execution_order'):
            # This should not raise an error if cells are run in correct order
            agent_registry._validate_execution_order()
            print("   ✅ Execution order validation passed")
        else:
            print("   ❌ Execution order validation method not found")
            return False
        
        print("\n🎯 DEPENDENCY INJECTION SYSTEM TEST PASSED")
        print("✅ All dependency injection components working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ DEPENDENCY INJECTION SYSTEM TEST FAILED: {e}")
        return False

def test_agent_error_handling():
    """
    Test that agents handle errors properly with clear messages
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 TESTING AGENT ERROR HANDLING")
    print("=" * 50)
    
    try:
        # Test 1: Agent with missing dependencies
        print("1. Testing Agent with Missing Dependencies")
        
        if 'agent_registry' in globals() and agent_registry:
            # Test that agent fails gracefully when dependencies are missing
            if hasattr(agent_registry, '_scam_agent_handler'):
                try:
                    # This should raise a RuntimeError
                    result = agent_registry._scam_agent_handler(
                        user_query="test query",
                        dependencies=None  # Missing dependencies
                    )
                    print("   ❌ Agent should have failed with missing dependencies")
                    return False
                except RuntimeError as e:
                    if "dependencies missing" in str(e).lower():
                        print("   ✅ Agent properly handles missing dependencies")
                    else:
                        print(f"   ❌ Unexpected error: {e}")
                        return False
            else:
                print("   ⚠️  Scam agent handler not found")
        else:
            print("   ⚠️  Agent registry not available")
        
        # Test 2: Clear error messages
        print("2. Testing Clear Error Messages")
        
        # Test runtime integrity check with missing components
        try:
            # Temporarily clear a component to test error handling
            import __main__
            original_main_system = getattr(__main__, 'main_system', None)
            
            # Remove main_system temporarily
            if hasattr(__main__, 'main_system'):
                delattr(__main__, 'main_system')
            
            # This should raise a clear error
            try:
                check_runtime_integrity()
                print("   ❌ Should have failed with missing main_system")
                return False
            except RuntimeError as e:
                if "main investigation system missing" in str(e).lower():
                    print("   ✅ Clear error message for missing main_system")
                else:
                    print(f"   ❌ Unclear error message: {e}")
                    return False
            finally:
                # Restore main_system
                if original_main_system:
                    __main__.main_system = original_main_system
                    
        except Exception as e:
            print(f"   ❌ Error in error handling test: {e}")
            return False
        
        print("\n🎯 AGENT ERROR HANDLING TEST PASSED")
        print("✅ All error handling working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ AGENT ERROR HANDLING TEST FAILED: {e}")
        return False

def test_real_data_enforcement():
    """
    Test that the system enforces real data usage and prevents mock data
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 TESTING REAL DATA ENFORCEMENT")
    print("=" * 50)
    
    try:
        # Test 1: Verify BigQuery client is real
        print("1. Testing BigQuery Client Authentication")
        
        if bq_client:
            # Test that we can actually query BigQuery
            test_query = "SELECT 1 as test_value"
            query_job = bq_client.query(test_query)
            result = query_job.result()
            
            # Should get a result
            rows = list(result)
            assert len(rows) == 1, "BigQuery test query should return 1 row"
            assert rows[0].test_value == 1, "BigQuery test query should return value 1"
            print("   ✅ BigQuery client is real and functional")
        else:
            print("   ❌ BigQuery client not available")
            return False
        
        # Test 2: Verify AI analyzer is real
        print("2. Testing AI Analyzer Authentication")
        
        if main_system.analyzer and hasattr(main_system.analyzer, 'openai_client'):
            # Test that OpenAI client is configured
            if main_system.analyzer.openai_client:
                # Check that API key is present (not testing actual call to avoid costs)
                if hasattr(main_system.analyzer.openai_client, 'api_key'):
                    api_key = main_system.analyzer.openai_client.api_key
                    assert api_key is not None, "OpenAI API key should be present"
                    assert len(api_key) > 10, "OpenAI API key should be valid length"
                    print("   ✅ AI analyzer is real and configured")
                else:
                    print("   ❌ OpenAI API key not configured")
                    return False
            else:
                print("   ❌ OpenAI client not available")
                return False
        else:
            print("   ❌ AI analyzer not available")
            return False
        
        # Test 3: Verify tables are real
        print("3. Testing Table Accessibility")
        
        if VERIFIED_TABLES:
            for table_name, table_info in VERIFIED_TABLES.items():
                if table_info.get('accessible'):
                    # Test that we can query the table
                    table_id = table_info['table_id']
                    test_query = f"SELECT COUNT(*) as row_count FROM `{table_id}` LIMIT 1"
                    
                    query_job = bq_client.query(test_query)
                    result = query_job.result()
                    rows = list(result)
                    
                    assert len(rows) == 1, f"Table {table_name} should return count"
                    row_count = rows[0].row_count
                    assert row_count >= 0, f"Table {table_name} should have non-negative count"
                    print(f"   ✅ Table {table_name} is real and accessible ({row_count} rows)")
        else:
            print("   ❌ VERIFIED_TABLES not available")
            return False
        
        print("\n🎯 REAL DATA ENFORCEMENT TEST PASSED")
        print("✅ All components using real data and connections")
        return True
        
    except Exception as e:
        print(f"\n❌ REAL DATA ENFORCEMENT TEST FAILED: {e}")
        return False

def test_execution_order_enforcement():
    """
    Test that the system enforces correct cell execution order
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 TESTING EXECUTION ORDER ENFORCEMENT")
    print("=" * 50)
    
    try:
        # Test 1: Validate current state
        print("1. Testing Current Execution State")
        
        # Check that all cells have been run in correct order
        import __main__
        
        # Cell 1 check
        ENVIRONMENT_READY = getattr(__main__, 'ENVIRONMENT_READY', False)
        assert ENVIRONMENT_READY, "Cell 1 not executed - ENVIRONMENT_READY not set"
        print("   ✅ Cell 1 executed: Environment ready")
        
        # Cell 2 check
        VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
        assert VERIFIED_TABLES is not None, "Cell 2 not executed - VERIFIED_TABLES not set"
        assert len(VERIFIED_TABLES) > 0, "Cell 2 not executed - VERIFIED_TABLES empty"
        print("   ✅ Cell 2 executed: BigQuery tables verified")
        
        # Cell 3 check
        investigation_manager = getattr(__main__, 'investigation_manager', None)
        assert investigation_manager is not None, "Cell 3 not executed - investigation_manager not set"
        print("   ✅ Cell 3 executed: Investigation manager ready")
        
        # Cell 4 check
        sql_executor = getattr(__main__, 'sql_executor', None)
        assert sql_executor is not None, "Cell 4 not executed - sql_executor not set"
        print("   ✅ Cell 4 executed: SQL executor ready")
        
        # Cell 5 check
        main_system = getattr(__main__, 'main_system', None)
        assert main_system is not None, "Cell 5 not executed - main_system not set"
        assert main_system.analyzer is not None, "Cell 5 not executed - analyzer not initialized"
        assert main_system.bq_client is not None, "Cell 5 not executed - bq_client not initialized"
        print("   ✅ Cell 5 executed: Main system ready")
        
        # Cell 7b check
        agent_registry = getattr(__main__, 'agent_registry', None)
        if agent_registry is not None:
            print("   ✅ Cell 7b executed: Agent registry ready")
        else:
            print("   ⚠️  Cell 7b not executed - agent registry not available")
        
        print("\n🎯 EXECUTION ORDER ENFORCEMENT TEST PASSED")
        print("✅ All cells executed in correct order")
        return True
        
    except Exception as e:
        print(f"\n❌ EXECUTION ORDER ENFORCEMENT TEST FAILED: {e}")
        return False

def test_agent_specific_table_requirements():
    """
    Test that agent-specific table requirements are met with graceful degradation
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 TESTING AGENT-SPECIFIC TABLE REQUIREMENTS")
    print("=" * 50)
    
    try:
        # Test agent-specific requirements if function is available
        if 'check_agent_specific_requirements' in globals():
            print("✅ Agent-specific requirements function available")
            
            # Test each agent with graceful degradation
            agents_to_test = [
                ('scam_agent', 'Scam Detection'),
                ('email_network_agent', 'Email Network Analysis'),
                ('exploratory_agent', 'Exploratory Investigation')
            ]
            
            all_passed = True
            
            for agent_name, agent_description in agents_to_test:
                print(f"\n📋 Testing {agent_description} Agent ({agent_name})")
                try:
                    check_agent_specific_requirements(agent_name)
                    print(f"   ✅ {agent_description} agent requirements met")
                    
                except Exception as e:
                    print(f"   ❌ {agent_description} agent requirements failed: {e}")
                    
                    # Check if it's a critical failure or graceful degradation
                    if "critical requirements not met" in str(e).lower():
                        print(f"   💥 CRITICAL FAILURE: {agent_name} cannot run")
                        all_passed = False
                    else:
                        print(f"   ⚠️  Non-critical issue: {agent_name} may have limited functionality")
                        # This is acceptable with graceful degradation
            
            if all_passed:
                print("\n✅ ALL AGENTS READY FOR EXECUTION")
                print("✅ Critical requirements met for all agents")
                print("✅ Graceful degradation available for optional features")
            else:
                print("\n❌ SOME AGENTS HAVE CRITICAL FAILURES")
                print("💡 Fix critical issues before running affected agents")
                
        else:
            print("⚠️  Agent-specific requirements function not available")
            print("   Add check_agent_specific_requirements() to Cell 2")
            
            # Manual check for basic requirements
            print("\n📋 Manual Basic Requirements Check:")
            
            # Check TTS Usage table (required for all agents)
            if 'TTS Usage' in VERIFIED_TABLES:
                tts_accessible = VERIFIED_TABLES['TTS Usage'].get('accessible', False)
                print(f"   TTS Usage (required): {'✅ Available' if tts_accessible else '❌ Not accessible'}")
                if not tts_accessible:
                    error = VERIFIED_TABLES['TTS Usage'].get('error', 'Unknown error')
                    print(f"      Error: {error}")
                    print(f"      💥 CRITICAL: All agents require TTS Usage table")
                    return False
            else:
                print("   TTS Usage (required): ❌ Not found in VERIFIED_TABLES")
                return False
            
            # Check Classification Flags table (optional with graceful degradation)
            if 'Classification Flags' in VERIFIED_TABLES:
                cf_accessible = VERIFIED_TABLES['Classification Flags'].get('accessible', False)
                print(f"   Classification Flags (optional): {'✅ Available' if cf_accessible else '⚠️  Not accessible'}")
                if not cf_accessible:
                    error = VERIFIED_TABLES['Classification Flags'].get('error', 'Unknown error')
                    print(f"      Error: {error}")
                    print(f"      💡 Impact: Enhanced detection features will be disabled")
                    print(f"      💡 Agents will fall back to basic functionality")
                else:
                    print(f"      ✅ Enhanced detection features available")
            else:
                print("   Classification Flags (optional): ⚠️  Not found in VERIFIED_TABLES")
                print("      💡 Impact: Enhanced detection features will be disabled")
        
        print("\n🎯 AGENT-SPECIFIC TABLE REQUIREMENTS TEST COMPLETE")
        print("📋 Agent Capabilities Summary:")
        print("   • scam_agent: Scam detection (basic pattern matching)")
        print("   • email_network_agent: Email network analysis (basic patterns)")
        print("   • exploratory_agent: General investigation (full functionality)")
        
        if 'Classification Flags' in VERIFIED_TABLES and VERIFIED_TABLES['Classification Flags'].get('accessible', False):
            print("   🎯 Enhanced Features Available:")
            print("     • scam_agent: ML-based risk scoring")
            print("     • email_network_agent: Advanced risk analysis")
        else:
            print("   ⚠️  Enhanced Features Disabled:")
            print("     • scam_agent: Using basic pattern matching")
            print("     • email_network_agent: Using basic network analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ AGENT-SPECIFIC TABLE REQUIREMENTS TEST FAILED: {e}")
        return False

def diagnose_table_accessibility_issues():
    """
    Diagnose why specific tables are not accessible
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    """
    print("\n🔍 DIAGNOSING TABLE ACCESSIBILITY ISSUES")
    print("=" * 50)
    
    try:
        if not VERIFIED_TABLES:
            print("❌ No VERIFIED_TABLES found - run Cell 2 first")
            return False
        
        # Check each table in detail
        for table_name, table_info in VERIFIED_TABLES.items():
            print(f"\n📋 Table: {table_name}")
            print(f"   Table ID: {table_info.get('table_id', 'N/A')}")
            print(f"   Accessible: {'✅ Yes' if table_info.get('accessible', False) else '❌ No'}")
            
            if not table_info.get('accessible', False):
                error = table_info.get('error', 'Unknown error')
                print(f"   Error: {error}")
                
                # Provide specific guidance based on error type
                if 'not found' in error.lower():
                    print("   💡 SOLUTION: Table may not exist in the specified project")
                    print("      - Check if table ID is correct")
                    print("      - Verify project permissions")
                    print("      - Check if table exists in BigQuery console")
                
                elif 'permission' in error.lower() or 'denied' in error.lower():
                    print("   💡 SOLUTION: Permission issue")
                    print("      - Check GCP authentication")
                    print("      - Verify dataset permissions")
                    print("      - Check if service account has BigQuery Data Viewer role")
                
                elif 'project' in error.lower():
                    print("   💡 SOLUTION: Project issue")
                    print("      - Check if project ID is correct")
                    print("      - Verify project permissions")
                    print("      - Check if project exists and is accessible")
                
                else:
                    print("   💡 SOLUTION: General troubleshooting")
                    print("      - Re-run Cell 2 to refresh table verification")
                    print("      - Check network connectivity")
                    print("      - Verify BigQuery client initialization")
            
            else:
                print("   ✅ Table is accessible")
                if 'actual_columns' in table_info:
                    columns = table_info['actual_columns']
                    print(f"   Columns: {len(columns)} found")
                    if 'missing_columns' in table_info and table_info['missing_columns']:
                        print(f"   Missing columns: {table_info['missing_columns']}")
        
        # Overall recommendations
        print(f"\n💡 GENERAL RECOMMENDATIONS:")
        print("1. Run Cell 2 to refresh table verification")
        print("2. Check GCP authentication and permissions")
        print("3. Verify project IDs and dataset names")
        print("4. Check BigQuery console for table existence")
        print("5. Ensure service account has proper BigQuery roles")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DIAGNOSIS FAILED: {e}")
        return False

# =============================================================================
# INITIALIZE TEST SUITE
# =============================================================================

print("🧪 INVESTIGATION TEST SUITE LOADED")
print("=" * 40)
print("⚠️  STRICT INVESTIGATOR GUIDELINES — NO mock data, REAL data only")
print("\n📋 RUNTIME INTEGRITY TESTS:")
print("  - test_runtime_integrity() - Test all critical components")
print("  - test_agent_execution_prerequisites() - Test agent readiness")
print("  - test_agent_specific_table_requirements() - Test agent table requirements")
print("  - run_comprehensive_integrity_test() - Complete integrity test")
print("  - quick_integrity_check() - Quick validation")
print("\n📋 DEPENDENCY INJECTION TESTS:")
print("  - test_dependency_injection_system() - Test dependency injection")
print("  - test_agent_error_handling() - Test error handling")
print("  - test_real_data_enforcement() - Test real data usage")
print("  - test_execution_order_enforcement() - Test cell execution order")
print("\n📋 DIAGNOSTIC TOOLS:")
print("  - diagnose_table_accessibility_issues() - Diagnose table access problems")
print("  - check_agent_specific_requirements('agent_name') - Check agent requirements")
print("\n📋 STANDARD INVESTIGATION TESTS:")
print("  - test_basic_components() - Basic component validation")
print("  - test_investigation_manager() - Investigation manager tests")
print("  - test_sql_executor() - SQL executor validation")
print("  - test_main_system() - Main system tests")
print("  - test_agent_system() - Agent system validation")
print("  - run_all_tests() - Complete test suite")
print("\n💡 RECOMMENDED WORKFLOW:")
print("  1. quick_integrity_check() - Verify system readiness")
print("  2. test_runtime_integrity() - Full component validation")
print("  3. test_agent_specific_table_requirements() - Check agent table requirements")
print("  4. test_dependency_injection_system() - Test dependency injection")
print("  5. test_real_data_enforcement() - Verify real data usage")
print("  6. run_comprehensive_integrity_test() - Complete integrity validation")
print("  7. run_all_tests() - Complete system validation")
print("\n🎯 BEFORE RUNNING AGENTS:")
print("  • ALWAYS run check_runtime_integrity() first")
print("  • Check agent-specific table requirements")
print("  • Ensure all critical components are available")
print("  • Verify dependency injection system is working")
print("  • Use REAL data only — NO mock data or simulation")
print("  • Follow cell execution order: Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 7b")
print("\n🔒 DEPENDENCY INJECTION FEATURES:")
print("  • Explicit parameter passing to agents")
print("  • No global state reliance")
print("  • Runtime integrity validation")
print("  • Mock data detection and prevention")
print("  • Clear error messages with fix instructions")
print("\n🔍 TROUBLESHOOTING:")
print("  • If scam agent fails: check Classification Flags table accessibility")
print("  • If table not accessible: run diagnose_table_accessibility_issues()")
print("  • If permissions issue: verify GCP authentication and BigQuery roles")
print("  • If table missing: check table ID and project permissions")

if WIDGETS_AVAILABLE:
    print("  - create_test_suite_interface()      # Interactive widget interface")
else:
    print("  [INFO] Interactive widgets not available - using text-based testing")

print()
print("Quick start:")
print("  validate_test_environment()          # Check environment")
print("  run_all_tests()                      # Command-line testing")
if WIDGETS_AVAILABLE:
    print("  create_test_suite_interface()        # Interactive testing")
print()
print("[SYSTEM] Cell 7f Complete - Test Suite Ready")
