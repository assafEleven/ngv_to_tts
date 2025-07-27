# @title Cell 1: Environment Setup & Validation
# Foundation setup for Vertex AI investigation environment

import os
import sys
import warnings
from datetime import datetime
import logging

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def validate_vertex_environment():
    """Validate we're running in Vertex AI environment"""
    print("🔍 Validating Vertex AI Environment")
    print("=" * 40)

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Python version may be too old for some BigQuery features")
    else:
        print("✅ Python version compatible")

    # Check if we're in Colab/Vertex
    try:
        import google.colab
        print("✅ Google Colab environment detected")
        environment = "colab"
    except ImportError:
        try:
            # Check for Vertex AI specific indicators
            if '/opt/conda' in sys.executable or 'vertex' in os.environ.get('HOSTNAME', '').lower():
                print("✅ Vertex AI environment detected")
                environment = "vertex"
            else:
                print("⚠️  Environment type unclear - assuming Vertex AI")
                environment = "vertex"
        except:
            environment = "unknown"

    return environment

def install_required_packages():
    """Install and validate required packages"""
    print("\n📦 Installing Required Packages")
    print("=" * 40)

    required_packages = [
        'google-cloud-bigquery>=3.0.0',
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'ipywidgets>=7.0.0'
    ]

    for package in required_packages:
        try:
            package_name = package.split('>=')[0]
            print(f"Checking {package_name}...")

            if package_name == 'google-cloud-bigquery':
                import google.cloud.bigquery
                print(f"✅ {package_name}: {google.cloud.bigquery.__version__}")
            elif package_name == 'pandas':
                import pandas as pd
                print(f"✅ {package_name}: {pd.__version__}")
            elif package_name == 'numpy':
                import numpy as np
                print(f"✅ {package_name}: {np.__version__}")
            elif package_name == 'ipywidgets':
                import ipywidgets
                print(f"✅ {package_name}: {ipywidgets.__version__}")

        except ImportError:
            print(f"❌ {package_name} not found - attempting enterprise-friendly installation...")
            try:
                install_success = enterprise_install_package(package, package_name)
                if install_success:
                    print(f"✅ {package_name} installed successfully")
                else:
                    print(f"⚠️  {package_name} installation failed - continuing without it")
            except Exception as e:
                print(f"⚠️  {package_name} installation failed: {str(e)}")
                print(f"   Enterprise environments may have package installation restrictions")
        except Exception as e:
            print(f"⚠️  {package_name} check failed: {str(e)}")

def enterprise_install_package(package, package_name):
    """Enterprise-friendly package installation with multiple fallback methods"""
    
    # Method 1: Standard pip install (works in most environments)
    try:
        print(f"   Trying standard pip install for {package_name}...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"   Standard pip failed: {result.stderr}")
    except Exception as e:
        print(f"   Standard pip failed: {str(e)}")
    
    # Method 2: User install (for restricted environments)
    try:
        print(f"   Trying user install for {package_name}...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--user", package, "--quiet"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"   User install failed: {result.stderr}")
    except Exception as e:
        print(f"   User install failed: {str(e)}")
    
    # Method 3: No dependencies install (for strict environments)
    try:
        print(f"   Trying no-deps install for {package_name}...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", package, "--quiet"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"   No-deps install failed: {result.stderr}")
    except Exception as e:
        print(f"   No-deps install failed: {str(e)}")
    
    # Method 4: Try importing again (might have been installed in different location)
    try:
        print(f"   Checking if {package_name} is now available...")
        if package_name == 'google.cloud.bigquery':
            from google.cloud import bigquery
            return True
        elif package_name == 'pandas':
            import pandas
            return True
        elif package_name == 'numpy':
            import numpy
            return True
        elif package_name == 'ipywidgets':
            import ipywidgets
            return True
    except ImportError:
        pass
    
    print(f"   All installation methods failed for {package_name}")
    return False

def validate_enterprise_package_environment():
    """Validate package installation capabilities in enterprise environment"""
    print("\n🏢 ENTERPRISE PACKAGE VALIDATION")
    print("=" * 50)
    
    # Check if pip is available
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ pip is available")
        else:
            print("❌ pip is not available")
            return False
    except Exception as e:
        print(f"❌ pip check failed: {str(e)}")
        return False
    
    # Check installation permissions
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Package listing works")
        else:
            print("⚠️  Package listing restricted")
    except Exception as e:
        print(f"⚠️  Package listing failed: {str(e)}")
    
    # Check if we can install a test package
    try:
        print("🧪 Testing package installation capability...")
        import subprocess
        # Try to install a very small, safe package
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "six"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Package installation should work")
        else:
            print("⚠️  Package installation may be restricted")
    except Exception as e:
        print(f"⚠️  Package installation test failed: {str(e)}")
    
    return True

def setup_logging():
    """Configure logging for investigation workflow"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Suppress noisy warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    print("✅ Logging configured")

# =============================================================================
# CORE IMPORTS
# =============================================================================

def import_core_libraries():
    """Import and validate core libraries"""
    print("\n📚 Importing Core Libraries")
    print("=" * 40)

    try:
        # Essential imports
        global pd, np, datetime, timedelta, bigquery, widgets, display, HTML, clear_output

        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from google.cloud import bigquery
        from google.api_core import exceptions
        import ipywidgets as widgets
        from IPython.display import display, HTML, clear_output

        print("✅ Core libraries imported successfully")

        # Display versions
        print(f"   Pandas: {pd.__version__}")
        print(f"   NumPy: {np.__version__}")
        print(f"   BigQuery: {bigquery.__version__}")

        return True

    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

# =============================================================================
# CREDENTIAL VALIDATION
# =============================================================================

def check_credentials():
    """Check for required credentials without exposing them"""
    print("\n🔐 Checking Credentials")
    print("=" * 40)

    credentials_status = {}

    # Check OpenAI API key
    if 'OPENAI_API_KEY' in os.environ:
        key = os.environ['OPENAI_API_KEY']
        if key.startswith('sk-') and len(key) > 20:
            print("✅ OpenAI API key found and formatted correctly")
            credentials_status['openai'] = True
        else:
            print("⚠️  OpenAI API key found but format unclear")
            credentials_status['openai'] = False
    else:
        print("⚠️  OpenAI API key not found in environment")
        credentials_status['openai'] = False

    # Check for Google Cloud credentials
    gcp_methods = []

    # Method 1: Service account key file
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        gcp_methods.append("Service account key file")

    # Method 2: Service account JSON in env
    if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
        gcp_methods.append("Service account JSON")

    # Method 3: Try default credentials
    try:
        from google.auth import default
        credentials, project = default()
        gcp_methods.append("Default credentials")
    except Exception:
        pass

    if gcp_methods:
        print(f"✅ Google Cloud credentials found via: {', '.join(gcp_methods)}")
        credentials_status['gcp'] = True
    else:
        print("⚠️  Google Cloud credentials not detected")
        credentials_status['gcp'] = False

    return credentials_status

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

def display_system_info():
    """Display system information for debugging"""
    print("\n💻 System Information")
    print("=" * 40)

    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")

    # Check available memory (if psutil available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        print(f"Memory usage: {memory.percent}%")
    except ImportError:
        print("Memory info: psutil not available")

    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB")
    except Exception:
        print("Disk info: not available")

# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_investigation_environment():
    """Complete environment setup for investigation notebook"""
    print("🚀 Investigation Environment Setup")
    print("=" * 60)

    # Step 1: Validate environment
    environment = validate_vertex_environment()

    # Step 2: Install packages
    install_required_packages()

    # Step 3: Import libraries
    import_success = import_core_libraries()

    if not import_success:
        print("❌ Critical import failure - cannot continue")
        return False

    # Step 4: Setup logging
    setup_logging()

    # Step 5: Check credentials
    credentials = check_credentials()

    # Step 6: Display system info
    display_system_info()

    # Step 7: Summary
    print("\n📋 Setup Summary")
    print("=" * 30)
    print(f"Environment: {environment}")
    print(f"Core imports: {'✅ Success' if import_success else '❌ Failed'}")
    print(f"OpenAI credentials: {'✅ Ready' if credentials['openai'] else '⚠️ Missing'}")
    print(f"GCP credentials: {'✅ Ready' if credentials['gcp'] else '⚠️ Missing'}")

    if import_success and credentials['gcp']:
        print("\n🎯 Environment ready for BigQuery operations")
        print("   Next: Run Cell 2 to initialize BigQuery clients")
    else:
        print("\n⚠️  Environment setup incomplete")
        print("   Fix credential issues before proceeding")

    return import_success and credentials['gcp']

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Run environment setup
setup_success = setup_investigation_environment()

# Create global status for other cells
ENVIRONMENT_READY = setup_success

print("\n" + "=" * 60)
print("Cell 1 Complete - Environment Setup")
print(f"Status: {'✅ Ready' if ENVIRONMENT_READY else '❌ Issues detected'}")
print("Global variable: ENVIRONMENT_READY")