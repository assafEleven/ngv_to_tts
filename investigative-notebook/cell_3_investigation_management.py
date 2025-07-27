# ==============================================================
# CELL 03: Investigation Manager
# Purpose: Investigation lifecycle and logging
# Dependencies: Global config, BigQuery auth, upstream env setup
# ==============================================================

# @title Cell 3: Investigation Management - Core Investigation State Management 
# Trust & Safety Investigation Management with datetime serialization

import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import sys
import os

# Basic datetime serialization functions
def deep_serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: deep_serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deep_serialize_datetime(item) for item in obj]
    else:
        return obj

def safe_json_dumps(obj, **kwargs):
    return json.dumps(obj, default=str, **kwargs)

def serialize_agent_result(obj):
    return obj

# =============================================================================
# INVESTIGATION DATA STRUCTURES
# =============================================================================

@dataclass
class Investigation:
    investigation_id: str
    title: str
    description: str
    investigator: str
    created_at: datetime
    status: str = "active"  # active, completed, suspended
    risk_level: str = "medium"  # low, medium, high, critical
    findings: List[str] = None
    actions_taken: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []
        if self.actions_taken is None:
            self.actions_taken = []

class InvestigationManager:
    """Manages investigation lifecycle and data with comprehensive validation"""
    
    def __init__(self):
        self.current_investigation: Optional[Investigation] = None
        self.investigation_history: List[Investigation] = []
        
    def create_investigation(self, title: str, description: str, investigator: str = "analyst", 
                           risk_level: str = "medium") -> Investigation:
        """
        Create a new investigation with comprehensive validation
        
        Args:
            title: Investigation title (required, 3-200 chars)
            description: Investigation description (required, 10-1000 chars)
            investigator: Investigator name (optional, defaults to "analyst")
            risk_level: Risk level (optional, must be: low, medium, high, critical)
            
        Returns:
            Investigation: Created investigation object
            
        Raises:
            ValueError: If validation fails
        """
        # ROBUST VALIDATION - No cyclic debugging
        if not title or not isinstance(title, str):
            raise ValueError("Investigation title is required and must be a string")
        
        if not description or not isinstance(description, str):
            raise ValueError("Investigation description is required and must be a string")
        
        if not investigator or not isinstance(investigator, str):
            raise ValueError("Investigator name is required and must be a string")
        
        # Validate string lengths
        if len(title.strip()) < 3:
            raise ValueError("Investigation title must be at least 3 characters long")
        if len(title.strip()) > 200:
            raise ValueError("Investigation title must not exceed 200 characters")
        
        if len(description.strip()) < 10:
            raise ValueError("Investigation description must be at least 10 characters long")
        if len(description.strip()) > 1000:
            raise ValueError("Investigation description must not exceed 1000 characters")
        
        # Validate risk level
        valid_risk_levels = ["low", "medium", "high", "critical"]
        if risk_level not in valid_risk_levels:
            raise ValueError(f"Risk level must be one of: {', '.join(valid_risk_levels)}")
        
        # Clean and normalize inputs
        title = title.strip()
        description = description.strip()
        investigator = investigator.strip()
        risk_level = risk_level.lower()
        
        # Generate readable investigation ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        investigation_id = f"INV_{timestamp}_{str(uuid.uuid4())[:8]}"
        
        try:
            investigation = Investigation(
                investigation_id=investigation_id,
                title=title,
                description=description,
                investigator=investigator,
                created_at=datetime.now(),
                status="active",
                risk_level=risk_level,
                findings=[],
                actions_taken=[],
                notes=""
            )
            
            self.current_investigation = investigation
            self.investigation_history.append(investigation)
            
            # Log creation to BigQuery with error handling
            try:
                self._log_to_bigquery(
                    investigation_id=investigation.investigation_id,
                    action_type="creation",
                    details=f"Investigation created: {title}",
                    notes=f"Risk Level: {risk_level}, Description: {description}"
                )
            except Exception as e:
                print(f"WARNING: Failed to log investigation creation to BigQuery: {e}")
                # Continue execution - logging failure shouldn't break investigation creation
            
            print(f"SUCCESS: Investigation created: {investigation.investigation_id}")
            print(f"   Title: {title}")
            print(f"   Risk Level: {risk_level}")
            print(f"   Investigator: {investigator}")
            
            return investigation
            
        except Exception as e:
            print(f"ERROR: Failed to create investigation: {e}")
            raise ValueError(f"Investigation creation failed: {str(e)}")
    
    def add_finding(self, finding, evidence: Optional[str] = None):
        """
        Add a finding to current investigation with comprehensive validation
        
        Args:
            finding: Finding description (str) or finding dict with 'description' key
            evidence: Supporting evidence (optional, max 1000 chars)
            
        Raises:
            ValueError: If validation fails
        """
        # ROBUST VALIDATION - No cyclic debugging
        if self.current_investigation is None:
            raise ValueError("No active investigation found. Create an investigation first.")
        
        # Handle both string and dict inputs
        if isinstance(finding, dict):
            if 'description' not in finding:
                raise ValueError("Finding dict must contain 'description' key. "
                               "Example: {'description': 'User violated policy', 'type': 'policy_violation', 'risk_level': 'high'}")
            finding_text = finding['description']
            # Extract evidence from dict if provided
            if evidence is None and 'evidence' in finding:
                evidence = finding['evidence']
        elif isinstance(finding, str):
            finding_text = finding
        else:
            raise ValueError("Finding must be a string or dict with 'description' key. "
                           "Examples: 'User violated policy' or {'description': 'User violated policy', 'type': 'policy_violation'}")
        
        if not finding_text or not isinstance(finding_text, str):
            raise ValueError("Finding description is required and must be a string")
        
        if len(finding_text.strip()) < 5:
            raise ValueError("Finding description must be at least 5 characters long")
        if len(finding_text.strip()) > 500:
            raise ValueError("Finding description must not exceed 500 characters")
        
        if evidence and not isinstance(evidence, str):
            raise ValueError("Evidence must be a string")
        if evidence and len(evidence.strip()) > 1000:
            raise ValueError("Evidence must not exceed 1000 characters")
        
        # Clean inputs
        finding_text = finding_text.strip()
        evidence = evidence.strip() if evidence else None
        
        try:
            # Safely serialize inputs
            safe_finding = deep_serialize_datetime(finding_text)
            safe_evidence = deep_serialize_datetime(evidence) if evidence else None
            
            self.current_investigation.findings.append(safe_finding)
            
            details = {"finding": safe_finding}
            if safe_evidence:
                details["evidence"] = safe_evidence
                
            # Log finding to BigQuery with error handling
            try:
                self._log_to_bigquery(
                    investigation_id=self.current_investigation.investigation_id,
                    action_type="finding",
                    details=deep_serialize_datetime(safe_finding),
                    notes=f"Evidence: {details.get('evidence', 'None')}" if isinstance(details, dict) else str(details)
                )
            except Exception as e:
                print(f"WARNING: Failed to log finding to BigQuery: {e}")
                # Continue execution - logging failure shouldn't break finding addition
            
            print(f"SUCCESS: Finding added to investigation {self.current_investigation.investigation_id}")
            print(f"  Finding: {safe_finding}")
            if safe_evidence:
                print(f"  Evidence: {safe_evidence}")
                
        except Exception as e:
            print(f"ERROR: Failed to add finding: {e}")
            raise ValueError(f"Finding addition failed: {str(e)}")
    
    def add_action(self, action: str, details: Optional[str] = None):
        """Add an action to current investigation with safe datetime serialization"""
        if self.current_investigation is None:
            print("ERROR: No active investigation")
            return
            
        # Safely serialize inputs
        safe_action = deep_serialize_datetime(action)
        safe_details = deep_serialize_datetime(details) if details else None
        
        self.current_investigation.actions_taken.append(safe_action)
        
        action_details = {"action": safe_action}
        if safe_details:
            action_details["details"] = safe_details
            
        self._log_to_bigquery(
            investigation_id=self.current_investigation.investigation_id,
            action_type="action",
            details=deep_serialize_datetime(safe_action),
        )
        
        print(f"SUCCESS: Action logged for investigation {self.current_investigation.investigation_id}")
        print(f"  Action: {safe_action}")
    
    def update_status(self, status: str, notes: Optional[str] = None):
        """Update investigation status"""
        if self.current_investigation is None:
            print("ERROR: No active investigation")
            return
        
        valid_statuses = ["active", "completed", "suspended"]
        if status not in valid_statuses:
            print(f"ERROR: Invalid status. Must be one of: {valid_statuses}")
            return
        
        old_status = self.current_investigation.status
        self.current_investigation.status = status
        
        if notes:
            self.current_investigation.notes = notes
        
        self._log_to_bigquery(
            investigation_id=self.current_investigation.investigation_id,
            action_type="status_update",
            details=f"Status changed from {old_status} to {status}",
            notes=notes
        )
        
        print(f"SUCCESS: Status updated to {status}")
        
    def get_investigation_summary(self) -> Dict[str, Any]:
        """Get summary of current investigation"""
        if not self.current_investigation:
            return {}
        
        return {
            'investigation_id': self.current_investigation.investigation_id,
            'title': self.current_investigation.title,
            'status': self.current_investigation.status,
            'risk_level': self.current_investigation.risk_level,
            'investigator': self.current_investigation.investigator,
            'created_at': self.current_investigation.created_at.isoformat(),
            'findings': self.current_investigation.findings,
            'actions_taken': self.current_investigation.actions_taken,
            'notes': self.current_investigation.notes
        }

    def _log_to_bigquery(self, investigation_id: str, action_type: str, 
                        details: str, user_id: Optional[str] = None, 
                        notes: Optional[str] = None, details_dict: Optional[Dict] = None):
        """Log action to BigQuery investigation logs table"""
        try:
            # Check if Investigation Logs table is available
            import __main__
            VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
            if VERIFIED_TABLES and 'Investigation Logs' in VERIFIED_TABLES:
                table_info = VERIFIED_TABLES['Investigation Logs']
                
                if table_info['accessible']:
                    client = table_info['client']
                    table_id = table_info['table_id']
                    
                    # Prepare log entry (convert datetime to ISO string for JSON serialization)
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'investigation_id': investigation_id,
                        'user_id': user_id or 'system',
                        'action_type': action_type,
                        'details': details,
                        'notes': notes or '',
                        'investigation_type': 'trust_safety',  # Required field for BigQuery table
                        'findings': json.dumps(details_dict.get('findings', []) if details_dict else []),  # Convert list to JSON string
                        'actions_taken': json.dumps(details_dict.get('actions_taken', []) if details_dict else []),  # Convert list to JSON string
                        'status': details_dict.get('status', 'active') if details_dict else 'active',  # Investigation status
                        'risk_level': details_dict.get('risk_level', 'medium') if details_dict else 'medium'  # Risk level
                    }
                    
                    # Convert to DataFrame and insert
                    df = pd.DataFrame([log_entry])
                    
                    # Debug: Print the DataFrame columns to verify all required fields are included
                    print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
                    print(f"DEBUG: investigation_type value: {df['investigation_type'].iloc[0] if 'investigation_type' in df.columns else 'MISSING'}")
                    print(f"DEBUG: findings field present: {'findings' in df.columns}, type: {type(df['findings'].iloc[0]) if 'findings' in df.columns else 'N/A'}")
                    print(f"DEBUG: actions_taken field present: {'actions_taken' in df.columns}, type: {type(df['actions_taken'].iloc[0]) if 'actions_taken' in df.columns else 'N/A'}")
                    print(f"DEBUG: status field present: {'status' in df.columns}")
                    print(f"DEBUG: risk_level field present: {'risk_level' in df.columns}")
                    
                    # Ensure timestamp column is properly typed as datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Insert into BigQuery with explicit schema to ensure investigation_type is included
                    job_config = bigquery.LoadJobConfig(
                        write_disposition="WRITE_APPEND",
                        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
                        # Define explicit schema to match BigQuery table structure
                        schema=[
                            bigquery.SchemaField("timestamp", "TIMESTAMP"),
                            bigquery.SchemaField("investigation_id", "STRING"),
                            bigquery.SchemaField("user_id", "STRING"),
                            bigquery.SchemaField("action_type", "STRING"),
                            bigquery.SchemaField("details", "STRING"),
                            bigquery.SchemaField("notes", "STRING"),
                            bigquery.SchemaField("investigation_type", "STRING"),
                            bigquery.SchemaField("findings", "STRING"),  # JSON string (converted from list)
                            bigquery.SchemaField("actions_taken", "STRING"),  # JSON string (converted from list)
                            bigquery.SchemaField("status", "STRING"),
                            bigquery.SchemaField("risk_level", "STRING"),
                        ]
                    )
                    
                    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
                    job.result()  # Wait for completion
                    
                else:
                    print("WARNING: Investigation Logs table not accessible - logging to local only")
            else:
                print("WARNING: Investigation Logs table not verified - logging to local only")
                
        except Exception as e:
            print(f"ERROR: Logging error: {str(e)}")
            # Provide specific guidance for common issues
            if "JSON serializable" in str(e):
                print("   TIP: Datetime objects need .isoformat() conversion for JSON")
            elif "permission" in str(e).lower():
                print("   TIP: Check BigQuery write permissions for investigation logs table")
            elif "schema" in str(e).lower() or "field" in str(e).lower():
                print("   TIP: Check BigQuery table schema matches log entry fields")
                print("   TIP: Ensure all required fields are included: investigation_type, findings (JSON), actions_taken (JSON), status, risk_level")
            elif "pyarrow" in str(e).lower():
                print("   TIP: PyArrow conversion error - complex data types converted to JSON strings")
                print("   TIP: Lists and objects are now stored as JSON strings in BigQuery")
            else:
                print("   TIP: Check BigQuery connection and table configuration")
    
    def get_findings(self, investigation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get findings for an investigation
        
        Args:
            investigation_id: Investigation ID (uses current if None)
            
        Returns:
            List of findings with metadata
        """
        if investigation_id:
            # Find specific investigation
            target_investigation = None
            for inv in self.investigation_history:
                if inv.investigation_id == investigation_id:
                    target_investigation = inv
                    break
            if not target_investigation:
                return []
        else:
            # Use current investigation
            target_investigation = self.current_investigation
            
        if not target_investigation:
            return []
        
        findings_list = []
        for i, finding in enumerate(target_investigation.findings):
            finding_data = {
                'finding_id': f"{target_investigation.investigation_id}_F{i+1}",
                'investigation_id': target_investigation.investigation_id,
                'description': finding,
                'timestamp': target_investigation.created_at,  # Placeholder - we don't track individual finding timestamps
                'risk_level': target_investigation.risk_level,
                'agent_name': 'manual',  # Default for manually added findings
                'summary': finding[:100] + '...' if len(finding) > 100 else finding,
                'full_content': finding
            }
            findings_list.append(finding_data)
        
        return findings_list

    def list_investigations(self):
        """List all investigations"""
        if not self.investigation_history:
            print("No investigations found")
            return
        
        print("INVESTIGATION HISTORY")
        print("=" * 50)
        
        for inv in self.investigation_history:
            status_indicator = "[ACTIVE]" if inv == self.current_investigation else "[CLOSED]"
            print(f"{status_indicator} {inv.investigation_id}")
            print(f"   Title: {inv.title}")
            print(f"   Status: {inv.status}")
            print(f"   Created: {inv.created_at}")
            print(f"   Findings: {len(inv.findings)}")
            print()

# =============================================================================
# GLOBAL INVESTIGATION MANAGER
# =============================================================================

def create_investigation_from_query(query_description: str, risk_level: str = "medium") -> Investigation:
    """Quick investigation creation from query description"""
    title = f"Investigation: {query_description[:50]}..."
    description = f"Automated investigation based on query: {query_description}"
    
    return investigation_manager.create_investigation(
        title=title,
        description=description,
        risk_level=risk_level
    )

def get_current_investigation_id() -> Optional[str]:
    """Get ID of current investigation"""
    if investigation_manager.current_investigation:
        return investigation_manager.current_investigation.investigation_id
    return None

def get_investigation_findings(investigation_id: Optional[str] = None) -> pd.DataFrame:
    """Get investigation findings as DataFrame"""
    
    # Use investigation manager to get findings
    if 'investigation_manager' in globals():
        findings = investigation_manager.get_findings(investigation_id)
        if findings:
            return pd.DataFrame(findings)
    
    # Fallback: try to query BigQuery directly if available
    import __main__
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
    if VERIFIED_TABLES and 'Investigation Logs' in VERIFIED_TABLES:
        table_info = VERIFIED_TABLES['Investigation Logs']
        if table_info['accessible']:
            client = table_info['client']
            table_id = table_info['table_id']
            
            if investigation_id:
                query = f"""
                SELECT * FROM `{table_id}`
                WHERE investigation_id = '{investigation_id}'
                AND action_type = 'finding'
                ORDER BY timestamp DESC
                """
            else:
                query = f"""
                SELECT * FROM `{table_id}`
                WHERE action_type = 'finding'
                ORDER BY timestamp DESC
                LIMIT 100
                """
            
            try:
                return client.query(query).to_dataframe()
            except Exception as e:
                print(f"ERROR: Failed to query findings: {e}")
                
    return pd.DataFrame()

def show_agent_findings(investigation_id: Optional[str] = None):
    """Display agent findings for the active investigation in text mode"""
    findings = investigation_manager.get_findings(investigation_id)
    
    if not findings:
        if investigation_id:
            print(f"No findings found for investigation {investigation_id}")
        else:
            print("No findings found for active investigation")
        return
    
    print("\nAGENT FINDINGS SUMMARY")
    print("=" * 60)
    
    if investigation_id:
        print(f"Investigation ID: {investigation_id}")
    else:
        current = investigation_manager.current_investigation
        if current:
            print(f"Investigation: {current.title}")
            print(f"Investigation ID: {current.investigation_id}")
    
    print(f"Total Findings: {len(findings)}")
    print()
    
    # Group by risk level
    risk_groups = {}
    for finding in findings:
        risk = finding.get('risk_level', 'unknown')
        if risk not in risk_groups:
            risk_groups[risk] = []
        risk_groups[risk].append(finding)
    
    # Display by risk level (highest first)
    risk_order = ['critical', 'high', 'medium', 'low', 'unknown']
    for risk_level in risk_order:
        if risk_level in risk_groups:
            findings_for_risk = risk_groups[risk_level]
            print(f"\n{risk_level.upper()} RISK ({len(findings_for_risk)} findings):")
            print("-" * 40)
            
            for finding in findings_for_risk:
                print(f"ID: {finding['finding_id']}")
                print(f"Agent: {finding['agent_name']}")
                print(f"Time: {finding['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(finding['timestamp'], 'strftime') else finding['timestamp']}")
                print(f"Summary: {finding['summary']}")
                print()

def show_investigation_dashboard():
    """Display investigation dashboard"""
    print("\nInvestigation Dashboard")
    print("=" * 60)
    
    if investigation_manager.current_investigation:
        inv = investigation_manager.current_investigation
        print(f"Current Investigation: {inv.title}")
        print(f"ID: {inv.investigation_id}")
        print(f"Status: {inv.status}")
        print(f"Risk Level: {inv.risk_level}")
        print(f"Created: {inv.created_at}")
        print(f"Investigator: {inv.investigator}")
        print()
        print(f"Progress:")
        print(f"  Findings: {len(inv.findings)}")
        print(f"  Actions: {len(inv.actions_taken)}")
        print()
        
        if inv.findings:
            print("Recent Findings:")
            for finding in inv.findings[-3:]:  # Show last 3
                print(f"  - {finding}")
            print()
    else:
        print("No active investigation")
        print("Create an investigation to get started")

# Initialize global investigation manager
investigation_manager = InvestigationManager()

print("SUCCESS: Investigation Management System initialized")
print("Available functions:")
print("  - investigation_manager.create_investigation(title, description)")
print("  - investigation_manager.add_finding(finding, evidence)")
print("  - investigation_manager.add_action(action, details)")
print("  - investigation_manager.update_status(status)")
print("  - show_investigation_dashboard()")
print("  - show_agent_findings()")
print("  - investigation_manager.list_investigations()")

# =============================================================================
# NOTEBOOK INTEGRATION - SHARED SYSTEM AWARENESS
# =============================================================================

print("\n🔗 NOTEBOOK INTEGRATION CHECK...")

# Check if we're in a unified notebook system
if 'NOTEBOOK_STATE' in globals():
    print("✅ Unified notebook system detected")
    
    # Integrate with shared system
    if 'setup_investigation_manager' in globals():
        print("✅ Integrating with shared notebook system...")
        setup_investigation_manager()
        print("✅ Investigation manager integrated with shared system")
        
        # Update global reference to use shared system
        if 'investigation_manager' in NOTEBOOK_STATE.get('shared_variables', {}):
            investigation_manager = NOTEBOOK_STATE['shared_variables']['investigation_manager']
            print("✅ Using shared investigation manager")
        
    else:
        print("⚠️  Shared system functions not available")
        print("   For full integration, run: exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")
        
else:
    print("⚠️  Running in standalone mode")
    print("   For full notebook integration, run:")
    print("   exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")

print("\n🎯 CELL 3 READY - Investigation Management available")
print("   • Standalone: investigation_manager.*")
print("   • Integrated: Uses shared investigation_manager") 