# Trust & Safety Investigation System - Cell Usage Guide

## 📋 Cell Execution Order

Your investigative agent system should be run in the following order:

### 1. **Cell 1: Environment Setup** (`cell_1_environment_setup.py`)
```bash
python cell_1_environment_setup.py
```
**Purpose:** Sets up the environment, validates dependencies, and configures logging and timezone

### 2. **Cell 2: BigQuery Configuration** (`integrated_cell_2_bigquery_config.py`)
```python
import integrated_cell_2_bigquery_config
```
**Purpose:** Configures BigQuery connections and verifies table access

### 3. **Cell 3: Investigation Management** (`cell_3_investigation_management.py`)
```python
import cell_3_investigation_management
```
**Purpose:** Provides investigation creation and management capabilities

### 4. **Cell 4: SQL Interface** (`cell_4_sql_interface.py`)
```python
import cell_4_sql_interface
```
**Purpose:** SQL query execution and database interface

### 5. **Cell 5: Main Investigation System** (`cell_5_main_investigation_system.py`)
```python
import cell_5_main_investigation_system
```
**Purpose:** Complete investigation system with UI

### 6. **Cell 7 Components** (Agent System)
- `cell_7b_agent_launcher.py` - Main agent launcher
- `cell_7c_agent_runtime_manager.py` - Runtime management
- `cell_7d_agent_test_debug.py` - Debug tools
- `cell_7e_investigation_summary_dashboard.py` - Dashboard
- `cell_7f_investigation_test_suite.py` - Test suite

## 🔧 **Dependencies**

**Cell 1 provides:**
- All standard imports (pandas, numpy, json, etc.)
- Logger configuration
- Timezone setup (PDT)
- Environment validation
- `CELL_1_INITIALIZED` flag for other cells

**Cell 2 requires:**
- Cell 1 initialization
- OpenAI API key (will prompt if missing)
- Google Cloud credentials

## 🚀 **Quick Start**

1. **Run Cell 1 first:**
   ```bash
   python cell_1_environment_setup.py
   ```

2. **Then import other cells as needed:**
   ```python
   import integrated_cell_2_bigquery_config
   import cell_3_investigation_management
   # ... etc
   ```

## 📝 **Notes**

- Cell 1 is now properly tracked in git
- All cells have datetime serialization fix integration
- Environment validation happens automatically
- Cell 2 will verify Cell 1 initialization before proceeding 