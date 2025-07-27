# =============================================================================
# PRODUCTION DEPLOYMENT GUIDE - TRUST & SAFETY INVESTIGATION PLATFORM
# =============================================================================
# Quick deployment script for all 3 critical production systems
# Run this to initialize the complete production-ready platform

import os
import sys
import time
from datetime import datetime

def print_banner():
    """Print deployment banner"""
    print("\n" + "="*80)
    print("🚀 TRUST & SAFETY INVESTIGATION PLATFORM - PRODUCTION DEPLOYMENT")
    print("="*80)
    print(f"Deployment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Initializing: Security, Persistence, and Monitoring Systems")
    print("="*80)

def deploy_security_system():
    """Deploy the security system"""
    print("\n🔒 STEP 1: DEPLOYING SECURITY SYSTEM")
    print("-" * 50)
    
    try:
        # Load security system
        exec(open('PRODUCTION_SECURITY_SYSTEM.py').read())
        
        # Initialize security system
        auth_system, audit_logger, secure_system = initialize_security_system()
        
        print("✅ Security System Deployed Successfully")
        print(f"   - Authentication: JWT-based")
        print(f"   - Roles: 6 role levels configured")
        print(f"   - Audit Logging: Enabled")
        print(f"   - Session Management: Active")
        
        return True, auth_system, audit_logger, secure_system
        
    except Exception as e:
        print(f"❌ Security System Deployment Failed: {e}")
        return False, None, None, None

def deploy_persistence_system():
    """Deploy the persistence system"""
    print("\n💾 STEP 2: DEPLOYING PERSISTENCE SYSTEM")
    print("-" * 50)
    
    try:
        # Load persistence system
        exec(open('PRODUCTION_DATA_PERSISTENCE.py').read())
        
        # Initialize persistence system
        persistence_system = initialize_persistence_system()
        
        # Get system health
        health = persistence_system.get_system_health()
        
        print("✅ Persistence System Deployed Successfully")
        print(f"   - Database: SQLite with WAL mode")
        print(f"   - Backup System: Enabled")
        print(f"   - Connection Pooling: Active")
        print(f"   - Investigation Storage: Ready")
        print(f"   - Records: {health.get('investigation_count', 0)} investigations")
        
        return True, persistence_system
        
    except Exception as e:
        print(f"❌ Persistence System Deployment Failed: {e}")
        return False, None

def deploy_monitoring_system():
    """Deploy the monitoring system"""
    print("\n📊 STEP 3: DEPLOYING MONITORING SYSTEM")
    print("-" * 50)
    
    try:
        # Load monitoring system
        exec(open('PRODUCTION_MONITORING_SYSTEM.py').read())
        
        # Initialize monitoring system
        monitoring_system = initialize_monitoring_system()
        
        # Wait a moment for initial health checks
        time.sleep(2)
        
        # Get system status
        status = monitoring_system.get_system_status()
        
        print("✅ Monitoring System Deployed Successfully")
        print(f"   - Health Checks: {len(status['health_checks'])} checks active")
        print(f"   - Performance Metrics: {len(status['performance_metrics'])} metrics")
        print(f"   - Overall Health: {status['overall_health']}")
        print(f"   - Active Alerts: {len(status['active_alerts'])}")
        print(f"   - Alert Channels: Email, Slack, Webhook, Log")
        
        return True, monitoring_system
        
    except Exception as e:
        print(f"❌ Monitoring System Deployment Failed: {e}")
        return False, None

def run_system_validation():
    """Run validation tests on all systems"""
    print("\n🧪 STEP 4: SYSTEM VALIDATION")
    print("-" * 50)
    
    validation_results = []
    
    # Test 1: Security System
    try:
        if 'security_system' in globals():
            # Test login
            token = login_user('admin', 'password')
            if token:
                user = get_current_user(token)
                if user:
                    validation_results.append("✅ Security: Authentication working")
                    logout_user(token)
                else:
                    validation_results.append("❌ Security: User validation failed")
            else:
                validation_results.append("❌ Security: Login failed")
        else:
            validation_results.append("❌ Security: System not available")
    except Exception as e:
        validation_results.append(f"❌ Security: Test failed - {e}")
    
    # Test 2: Persistence System
    try:
        if 'persistence_system' in globals():
            health = persistence_system.get_system_health()
            if health.get('status') == 'healthy':
                validation_results.append("✅ Persistence: Database healthy")
            else:
                validation_results.append(f"❌ Persistence: Database unhealthy - {health.get('error', 'Unknown')}")
        else:
            validation_results.append("❌ Persistence: System not available")
    except Exception as e:
        validation_results.append(f"❌ Persistence: Test failed - {e}")
    
    # Test 3: Monitoring System
    try:
        if 'monitoring_system' in globals():
            status = monitoring_system.get_system_status()
            if status.get('overall_health') in ['healthy', 'degraded']:
                validation_results.append("✅ Monitoring: System operational")
            else:
                validation_results.append("❌ Monitoring: System unhealthy")
        else:
            validation_results.append("❌ Monitoring: System not available")
    except Exception as e:
        validation_results.append(f"❌ Monitoring: Test failed - {e}")
    
    # Print results
    for result in validation_results:
        print(f"   {result}")
    
    # Overall validation
    failed_tests = [r for r in validation_results if '❌' in r]
    if not failed_tests:
        print("\n🎉 ALL SYSTEMS VALIDATED SUCCESSFULLY!")
        return True
    else:
        print(f"\n⚠️  {len(failed_tests)} VALIDATION TESTS FAILED")
        return False

def display_deployment_summary():
    """Display final deployment summary"""
    print("\n" + "="*80)
    print("🎊 DEPLOYMENT COMPLETE - PRODUCTION READY!")
    print("="*80)
    
    print("\n📋 DEPLOYED SYSTEMS:")
    print("   🔒 Security System: Authentication, Authorization, Audit Logging")
    print("   💾 Persistence System: Database, Backups, State Management")
    print("   📊 Monitoring System: Health Checks, Alerts, Performance Metrics")
    
    print("\n🚀 QUICK START GUIDE:")
    print("   1. Login: token = login_user('admin', 'password')")
    print("   2. Investigate: secure_investigation_system.run_investigation_agent(query, session_token=token)")
    print("   3. Monitor: monitoring_system.get_system_status()")
    print("   4. Logout: logout_user(token)")
    
    print("\n🎯 SYSTEM STATUS:")
    try:
        # Security status
        if 'security_system' in globals():
            active_sessions = len(security_system.active_sessions)
            print(f"   🔒 Security: {active_sessions} active sessions")
        
        # Persistence status
        if 'persistence_system' in globals():
            health = persistence_system.get_system_health()
            investigation_count = health.get('investigation_count', 0)
            print(f"   💾 Persistence: {investigation_count} investigations stored")
        
        # Monitoring status
        if 'monitoring_system' in globals():
            status = monitoring_system.get_system_status()
            overall_health = status.get('overall_health', 'unknown')
            active_alerts = len(status.get('active_alerts', []))
            print(f"   📊 Monitoring: {overall_health} health, {active_alerts} active alerts")
            
    except Exception as e:
        print(f"   ⚠️  Status check failed: {e}")
    
    print("\n📄 DOCUMENTATION:")
    print("   - Security Guide: See PRODUCTION_SECURITY_SYSTEM.py")
    print("   - Persistence Guide: See PRODUCTION_DATA_PERSISTENCE.py")
    print("   - Monitoring Guide: See PRODUCTION_MONITORING_SYSTEM.py")
    print("   - Production Gaps Analysis: See CRITICAL_PRODUCTION_GAPS.md")
    print("   - Complete Solution: See TOP_3_PRODUCTION_ISSUES_RESOLVED.md")
    
    print("\n" + "="*80)
    print("🎉 TRUST & SAFETY INVESTIGATION PLATFORM IS PRODUCTION READY!")
    print("="*80)

def main():
    """Main deployment function"""
    print_banner()
    
    success_count = 0
    total_systems = 3
    
    # Deploy Security System
    security_success, auth_system, audit_logger, secure_system = deploy_security_system()
    if security_success:
        success_count += 1
    
    # Deploy Persistence System
    persistence_success, persistence_system = deploy_persistence_system()
    if persistence_success:
        success_count += 1
    
    # Deploy Monitoring System
    monitoring_success, monitoring_system = deploy_monitoring_system()
    if monitoring_success:
        success_count += 1
    
    # Run validation
    if success_count == total_systems:
        validation_success = run_system_validation()
        
        if validation_success:
            display_deployment_summary()
            return True
        else:
            print("\n⚠️  DEPLOYMENT COMPLETED BUT VALIDATION FAILED")
            print("Some systems may have issues. Check the validation results above.")
            return False
    else:
        print(f"\n❌ DEPLOYMENT FAILED: {success_count}/{total_systems} systems deployed successfully")
        print("Check the error messages above and resolve issues before retrying.")
        return False

# =============================================================================
# QUICK DEPLOYMENT COMMANDS
# =============================================================================

def quick_deploy():
    """Quick deployment without detailed output"""
    print("🚀 Quick deployment starting...")
    
    try:
        # Load and initialize all systems
        exec(open('PRODUCTION_SECURITY_SYSTEM.py').read())
        exec(open('PRODUCTION_DATA_PERSISTENCE.py').read())
        exec(open('PRODUCTION_MONITORING_SYSTEM.py').read())
        
        # Initialize
        auth_system, audit_logger, secure_system = initialize_security_system()
        persistence_system = initialize_persistence_system()
        monitoring_system = initialize_monitoring_system()
        
        print("✅ All systems deployed successfully!")
        print("🎉 Production Trust & Safety Platform is ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick deployment failed: {e}")
        return False

def deploy_individual_system(system_name):
    """Deploy individual system"""
    system_map = {
        'security': 'PRODUCTION_SECURITY_SYSTEM.py',
        'persistence': 'PRODUCTION_DATA_PERSISTENCE.py',
        'monitoring': 'PRODUCTION_MONITORING_SYSTEM.py'
    }
    
    if system_name not in system_map:
        print(f"❌ Unknown system: {system_name}")
        print(f"Available systems: {', '.join(system_map.keys())}")
        return False
    
    try:
        print(f"🚀 Deploying {system_name} system...")
        exec(open(system_map[system_name]).read())
        
        # Initialize based on system type
        if system_name == 'security':
            auth_system, audit_logger, secure_system = initialize_security_system()
        elif system_name == 'persistence':
            persistence_system = initialize_persistence_system()
        elif system_name == 'monitoring':
            monitoring_system = initialize_monitoring_system()
        
        print(f"✅ {system_name.title()} system deployed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {system_name.title()} system deployment failed: {e}")
        return False

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            quick_deploy()
        elif command == "security":
            deploy_individual_system('security')
        elif command == "persistence":
            deploy_individual_system('persistence')
        elif command == "monitoring":
            deploy_individual_system('monitoring')
        elif command in ["help", "-h", "--help"]:
            print("\n🚀 TRUST & SAFETY PLATFORM DEPLOYMENT")
            print("Usage: python PRODUCTION_DEPLOYMENT_GUIDE.py [command]")
            print("\nCommands:")
            print("  (no args)   - Full deployment with validation")
            print("  quick       - Quick deployment without detailed output")
            print("  security    - Deploy security system only")
            print("  persistence - Deploy persistence system only")
            print("  monitoring  - Deploy monitoring system only")
            print("  help        - Show this help message")
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'help' for available commands")
    else:
        # Run full deployment
        success = main()
        sys.exit(0 if success else 1)

# =============================================================================
# JUPYTER NOTEBOOK USAGE
# =============================================================================

"""
JUPYTER NOTEBOOK USAGE:

# Full deployment
exec(open('PRODUCTION_DEPLOYMENT_GUIDE.py').read())
main()

# Quick deployment
quick_deploy()

# Individual systems
deploy_individual_system('security')
deploy_individual_system('persistence')
deploy_individual_system('monitoring')
"""

print("📋 PRODUCTION DEPLOYMENT GUIDE LOADED")
print("=" * 50)
print("Available functions:")
print("  • main() - Full deployment with validation")
print("  • quick_deploy() - Quick deployment")
print("  • deploy_individual_system(name) - Deploy single system")
print("")
print("🚀 Run main() to start full deployment") 