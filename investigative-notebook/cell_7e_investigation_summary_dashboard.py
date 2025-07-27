# @title Cell 7e: Investigation Summary Generator and Dashboard Renderer
# Comprehensive investigation reporting and dashboard for Trust & Safety workflow

import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Try to import ipywidgets - graceful fallback if not available
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

# =============================================================================
# INVESTIGATION SUMMARY GENERATION
# =============================================================================

def generate_investigation_summary(investigation_id: str = None):
    """Generate comprehensive investigation summary"""

    # Get investigation
    if investigation_id:
        # For now, only support current investigation
        # Future: implement investigation history lookup
        if not investigation_manager.current_investigation or investigation_manager.current_investigation.investigation_id != investigation_id:
            print("ERROR: Investigation ID not found or not current")
            return None
        investigation = investigation_manager.current_investigation
    else:
        investigation = investigation_manager.current_investigation

    if investigation is None:
        print("ERROR: No active investigation found")
        return None

    # Generate summary
    summary = {
        'investigation_id': investigation.investigation_id,
        'title': investigation.title,
        'description': investigation.description,
        'status': investigation.status,
        'risk_level': investigation.risk_level,
        'created_at': investigation.created_at.isoformat(),
        'investigator': investigation.investigator,
        'findings_count': len(investigation.findings),
        'actions_count': len(investigation.actions_taken),
        'findings': investigation.findings,
        'actions_taken': investigation.actions_taken,
        'notes': investigation.notes,
        'summary_generated_at': datetime.now().isoformat()
    }

    # Display summary
    print("\nINVESTIGATION SUMMARY")
    print("=" * 60)
    print(f"Investigation: {investigation.title}")
    print(f"ID: {investigation.investigation_id}")
    print(f"Status: {investigation.status} | Risk Level: {investigation.risk_level}")
    print(f"Created: {investigation.created_at}")
    print(f"Investigator: {investigation.investigator}")
    print(f"Description: {investigation.description}")

    print(f"\nFindings ({len(investigation.findings)}):")
    for i, finding in enumerate(investigation.findings, 1):
        print(f"  {i}. {finding}")

    print(f"\nActions Taken ({len(investigation.actions_taken)}):")
    for i, action in enumerate(investigation.actions_taken, 1):
        print(f"  {i}. {action}")

    if investigation.notes:
        print(f"\nNotes: {investigation.notes}")

    print(f"\nSummary generated at: {datetime.now()}")

    return summary

def create_investigation_dashboard():
    """Create a comprehensive investigation dashboard"""
    print("\nINVESTIGATION DASHBOARD")
    print("=" * 60)

    # Current investigation
    if investigation_manager.current_investigation is not None:
        inv = investigation_manager.current_investigation
        print(f"CURRENT INVESTIGATION:")
        print(f"  Title: {inv.title}")
        print(f"  ID: {inv.investigation_id}")
        print(f"  Status: {inv.status}")
        print(f"  Risk Level: {inv.risk_level}")
        print(f"  Created: {inv.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Findings: {len(inv.findings)}")
        print(f"  Actions: {len(inv.actions_taken)}")

        # Risk level indicator
        risk_indicators = {
            'low': 'INFO',
            'medium': 'WARNING',
            'high': 'URGENT',
            'critical': 'CRITICAL'
        }
        risk_indicator = risk_indicators.get(inv.risk_level, 'UNKNOWN')
        print(f"  Risk Status: {risk_indicator}")

        # Recent activity
        if inv.findings:
            print(f"\nRECENT FINDINGS:")
            for finding in inv.findings[-3:]:  # Show last 3 findings
                print(f"  - {finding}")

        if inv.actions_taken:
            print(f"\nRECENT ACTIONS:")
            for action in inv.actions_taken[-3:]:  # Show last 3 actions
                print(f"  - {action}")
    else:
        print("No active investigation")

    # Investigation history
    print(f"\nINVESTIGATION HISTORY ({len(investigation_manager.investigation_history)} total):")
    for inv in investigation_manager.investigation_history[-5:]:  # Show last 5
        status_marker = "ACTIVE" if inv.status == "active" else inv.status.upper()
        print(f"  {status_marker}: {inv.title} ({inv.risk_level})")

    # Quick actions
    print(f"\nQUICK ACTIONS:")
    print("  - generate_investigation_summary(investigation_id)")
    print("  - create_investigation_dashboard()")
    print("  - export_investigation_data(investigation_id)")
    print("  - show_investigation_metrics()")

    print("\nDashboard updated at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def export_investigation_data(investigation_id: str = None, format: str = 'json'):
    """Export investigation data in specified format"""
    if investigation_manager.current_investigation is None:
        print("ERROR: No active investigation")
        return None

    inv = investigation_manager.current_investigation

    if format.lower() == 'json':
        data = {
            'investigation_id': inv.investigation_id,
            'title': inv.title,
            'description': inv.description,
            'status': inv.status,
            'risk_level': inv.risk_level,
            'created_at': inv.created_at.isoformat(),
            'investigator': inv.investigator,
            'findings': inv.findings,
            'actions_taken': inv.actions_taken,
            'notes': inv.notes,
            'export_timestamp': datetime.now().isoformat()
        }

        json_str = json.dumps(data, indent=2)
        print(f"Investigation data exported as JSON:")
        print(json_str)
        return json_str

    elif format.lower() == 'csv':
        # Create flat structure for CSV
        rows = []

        # Add findings
        for finding in inv.findings:
            rows.append({
                'investigation_id': inv.investigation_id,
                'type': 'finding',
                'content': finding,
                'timestamp': inv.created_at.isoformat()
            })

        # Add actions
        for action in inv.actions_taken:
            rows.append({
                'investigation_id': inv.investigation_id,
                'type': 'action',
                'content': action,
                'timestamp': inv.created_at.isoformat()
            })

        df = pd.DataFrame(rows)
        print(f"Investigation data exported as CSV:")
        print(df.to_csv(index=False))
        return df

    else:
        print(f"ERROR: Unsupported format '{format}'. Use 'json' or 'csv'")
        return None

def show_investigation_metrics():
    """Display investigation system metrics"""
    print("\nINVESTIGATION SYSTEM METRICS")
    print("=" * 40)

    # Current investigation metrics
    if investigation_manager.current_investigation is not None:
        inv = investigation_manager.current_investigation
        print(f"CURRENT INVESTIGATION:")
        print(f"  Duration: {datetime.now() - inv.created_at}")
        print(f"  Findings Rate: {len(inv.findings)}/investigation")
        print(f"  Actions Rate: {len(inv.actions_taken)}/investigation")
        print(f"  Risk Level: {inv.risk_level}")

    # Historical metrics
    total_investigations = len(investigation_manager.investigation_history)
    active_investigations = len([inv for inv in investigation_manager.investigation_history if inv.status == 'active'])
    completed_investigations = len([inv for inv in investigation_manager.investigation_history if inv.status == 'completed'])

    print(f"\nHISTORICAL METRICS:")
    print(f"  Total Investigations: {total_investigations}")
    print(f"  Active: {active_investigations}")
    print(f"  Completed: {completed_investigations}")

    if total_investigations > 0:
        total_findings = sum(len(inv.findings) for inv in investigation_manager.investigation_history)
        total_actions = sum(len(inv.actions_taken) for inv in investigation_manager.investigation_history)

        print(f"  Average Findings per Investigation: {total_findings / total_investigations:.1f}")
        print(f"  Average Actions per Investigation: {total_actions / total_investigations:.1f}")

    # Risk level distribution
    risk_levels = [inv.risk_level for inv in investigation_manager.investigation_history]
    risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}

    print(f"\nRISK LEVEL DISTRIBUTION:")
    for level, count in risk_counts.items():
        print(f"  {level}: {count}")

    print(f"\nMetrics generated at: {datetime.now()}")

# =============================================================================
# ADVANCED DASHBOARD FEATURES
# =============================================================================

def create_interactive_dashboard():
    """Create an interactive investigation dashboard with widgets"""
    if not WIDGETS_AVAILABLE:
        print("WARNING: ipywidgets not available - using text dashboard")
        create_investigation_dashboard()
        return

    try:
        print("[WIDGET] Creating Interactive Investigation Dashboard...")

        # Dashboard state
        dashboard_state = {'current_view': 'overview'}

        # Create widgets
        view_dropdown = widgets.Dropdown(
            options=['overview', 'current_investigation', 'history', 'metrics'],
            value='overview',
            description='View:'
        )

        refresh_button = widgets.Button(
            description='Refresh Dashboard',
            button_style='success'
        )

        export_button = widgets.Button(
            description='Export Data',
            button_style='info'
        )

        output_area = widgets.Output()

        # Widget interactions
        def update_dashboard(change=None):
            with output_area:
                clear_output()

                if view_dropdown.value == 'overview':
                    create_investigation_dashboard()
                elif view_dropdown.value == 'current_investigation':
                    if investigation_manager.current_investigation:
                        generate_investigation_summary(investigation_manager.current_investigation.investigation_id)
                    else:
                        print("No active investigation")
                elif view_dropdown.value == 'history':
                    investigation_manager.list_investigations()
                elif view_dropdown.value == 'metrics':
                    show_investigation_metrics()

        def handle_refresh(button):
            update_dashboard()

        def handle_export(button):
            with output_area:
                clear_output()
                if investigation_manager.current_investigation:
                    export_investigation_data(investigation_manager.current_investigation.investigation_id)
                else:
                    print("No active investigation to export")

        # Connect widgets
        view_dropdown.observe(update_dashboard, names='value')
        refresh_button.on_click(handle_refresh)
        export_button.on_click(handle_export)

        # Initial update
        update_dashboard()

        # Display dashboard
        dashboard = widgets.VBox([
            widgets.HBox([view_dropdown, refresh_button, export_button]),
            output_area
        ])

        display(dashboard)

        print("[WIDGET] Interactive dashboard created successfully!")

    except Exception as e:
        print(f"ERROR: Failed to create interactive dashboard - {str(e)}")
        print("Falling back to text dashboard...")
        create_investigation_dashboard()

def create_summary_widget():
    """Create a widget for generating investigation summaries"""
    if not WIDGETS_AVAILABLE:
        print("WARNING: ipywidgets not available - use generate_investigation_summary() function")
        return

    try:
        # Create widgets
        investigation_id_input = widgets.Text(
            placeholder='Enter investigation ID',
            description='Investigation ID:'
        )

        generate_button = widgets.Button(
            description='Generate Summary',
            button_style='primary'
        )

        output_area = widgets.Output()

        # Widget interaction
        def generate_summary(button):
            with output_area:
                clear_output()

                if investigation_id_input.value:
                    generate_investigation_summary(investigation_id_input.value)
                elif investigation_manager.current_investigation:
                    generate_investigation_summary(investigation_manager.current_investigation.investigation_id)
                else:
                    print("Please enter an investigation ID or create an active investigation")

        generate_button.on_click(generate_summary)

        # Display widget
        widget = widgets.VBox([
            widgets.HBox([investigation_id_input, generate_button]),
            output_area
        ])

        display(widget)

        print("[WIDGET] Summary widget created successfully!")

    except Exception as e:
        print(f"ERROR: Failed to create summary widget - {str(e)}")
        print("Use generate_investigation_summary() function instead")

def create_multi_view_dashboard():
    """Create a dashboard with multiple views"""
    print("\nMULTI-VIEW INVESTIGATION DASHBOARD")
    print("=" * 60)

    # View 1: Executive Summary
    print("\n1. EXECUTIVE SUMMARY")
    print("-" * 20)

    if investigation_manager.current_investigation is not None:
        inv = investigation_manager.current_investigation
        print(f"Investigation: {inv.title}")
        print(f"Status: {inv.status} | Risk: {inv.risk_level}")
        print(f"Progress: {len(inv.findings)} findings, {len(inv.actions_taken)} actions")

        # Risk assessment
        risk_score = len(inv.findings) * 10 + len(inv.actions_taken) * 5
        print(f"Activity Score: {risk_score}")

        # Recommendations
        if inv.risk_level in ['high', 'critical']:
            print("RECOMMENDED ACTION: Immediate review required")
        elif len(inv.findings) > 5:
            print("RECOMMENDED ACTION: Consider escalation")
        else:
            print("RECOMMENDED ACTION: Continue monitoring")

    # View 2: Technical Details
    print("\n2. TECHNICAL DETAILS")
    print("-" * 20)

    if investigation_manager.current_investigation is not None:
        inv = investigation_manager.current_investigation
        print(f"Investigation ID: {inv.investigation_id}")
        print(f"Created: {inv.created_at}")
        print(f"Investigator: {inv.investigator}")
        print(f"Description: {inv.description}")

        if inv.notes:
            print(f"Notes: {inv.notes}")

    # View 3: Activity Timeline
    print("\n3. ACTIVITY TIMELINE")
    print("-" * 20)

    if investigation_manager.current_investigation is not None:
        inv = investigation_manager.current_investigation

        # Simulate timeline (in real implementation, would have timestamps)
        timeline = []
        timeline.append(f"Investigation created: {inv.created_at}")

        for finding in inv.findings:
            timeline.append(f"Finding added: {finding}")

        for action in inv.actions_taken:
            timeline.append(f"Action taken: {action}")

        # Show recent timeline items
        for item in timeline[-5:]:
            print(f"  {item}")

    print(f"\nDashboard generated at: {datetime.now()}")

def create_risk_assessment_view():
    """Create a risk assessment focused view"""
    print("\nRISK ASSESSMENT DASHBOARD")
    print("=" * 50)

    if investigation_manager.current_investigation is None:
        print("ERROR: No active investigation")
        return

    inv = investigation_manager.current_investigation

    # Risk level analysis
    risk_levels = {
        'low': 1,
        'medium': 2,
        'high': 3,
        'critical': 4
    }

    current_risk_score = risk_levels.get(inv.risk_level, 0)
    print(f"CURRENT RISK LEVEL: {inv.risk_level.upper()}")
    print(f"Risk Score: {current_risk_score}/4")

    # Risk indicators
    print("\nRISK INDICATORS:")
    print(f"  Findings Count: {len(inv.findings)} {'(HIGH)' if len(inv.findings) > 10 else '(NORMAL)'}")
    print(f"  Actions Taken: {len(inv.actions_taken)} {'(HIGH)' if len(inv.actions_taken) > 5 else '(NORMAL)'}")

    # Calculate investigation age
    investigation_age = datetime.now() - inv.created_at
    age_days = investigation_age.days
    print(f"  Investigation Age: {age_days} days {'(STALE)' if age_days > 7 else '(FRESH)'}")

    # Risk recommendations
    print("\nRISK RECOMMENDATIONS:")
    if inv.risk_level == 'critical':
        print("  - IMMEDIATE ACTION REQUIRED")
        print("  - Escalate to senior investigator")
        print("  - Consider emergency response procedures")
    elif inv.risk_level == 'high':
        print("  - Priority investigation")
        print("  - Review within 24 hours")
        print("  - Consider additional resources")
    elif inv.risk_level == 'medium':
        print("  - Monitor closely")
        print("  - Review within 48 hours")
        print("  - Standard investigation procedures")
    else:
        print("  - Routine monitoring")
        print("  - Standard review cycle")

    # Activity summary
    print("\nACTIVITY SUMMARY:")
    print(f"  Total Findings: {len(inv.findings)}")
    print(f"  Total Actions: {len(inv.actions_taken)}")
    
    if inv.findings:
        print(f"  Latest Finding: {inv.findings[-1]}")
    
    if inv.actions_taken:
        print(f"  Latest Action: {inv.actions_taken[-1]}")

    print(f"\nRisk assessment updated at: {datetime.now()}")

def generate_investigation_report():
    """Generate a comprehensive investigation report"""
    print("\nCOMPREHENSIVE INVESTIGATION REPORT")
    print("=" * 60)

    if investigation_manager.current_investigation is None:
        print("ERROR: No active investigation")
        return None

    inv = investigation_manager.current_investigation

    # Report header
    print(f"INVESTIGATION REPORT")
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Investigation ID: {inv.investigation_id}")
    print(f"Title: {inv.title}")
    print(f"Investigator: {inv.investigator}")
    print(f"Status: {inv.status}")
    print(f"Risk Level: {inv.risk_level}")
    print(f"Created: {inv.created_at}")
    print(f"Duration: {datetime.now() - inv.created_at}")
    print()

    # Executive summary
    print("EXECUTIVE SUMMARY")
    print("-" * 20)
    print(f"Description: {inv.description}")
    print(f"Current Status: {inv.status}")
    print(f"Risk Assessment: {inv.risk_level}")
    print(f"Investigation Progress: {len(inv.findings)} findings, {len(inv.actions_taken)} actions")
    print()

    # Detailed findings
    print("DETAILED FINDINGS")
    print("-" * 20)
    if inv.findings:
        for i, finding in enumerate(inv.findings, 1):
            print(f"{i}. {finding}")
    else:
        print("No findings recorded.")
    print()

    # Actions taken
    print("ACTIONS TAKEN")
    print("-" * 20)
    if inv.actions_taken:
        for i, action in enumerate(inv.actions_taken, 1):
            print(f"{i}. {action}")
    else:
        print("No actions recorded.")
    print()

    # Notes and observations
    if inv.notes:
        print("NOTES AND OBSERVATIONS")
        print("-" * 20)
        print(inv.notes)
        print()

    # Risk assessment
    print("RISK ASSESSMENT")
    print("-" * 20)
    risk_score = len(inv.findings) * 10 + len(inv.actions_taken) * 5
    print(f"Activity Score: {risk_score}")
    print(f"Risk Level: {inv.risk_level}")
    
    if inv.risk_level in ['high', 'critical']:
        print("URGENT: This investigation requires immediate attention")
    
    print()

    # Recommendations
    print("RECOMMENDATIONS")
    print("-" * 20)
    if inv.risk_level == 'critical':
        print("1. Immediate escalation required")
        print("2. Consider emergency response procedures")
        print("3. Assign additional resources")
    elif inv.risk_level == 'high':
        print("1. Priority investigation status")
        print("2. Review within 24 hours")
        print("3. Consider additional investigative resources")
    elif len(inv.findings) > 5:
        print("1. Consider escalation due to high activity")
        print("2. Review investigation scope")
        print("3. Monitor for pattern development")
    else:
        print("1. Continue standard investigation procedures")
        print("2. Regular monitoring and review")
        print("3. Document any new developments")

    print()
    print("END OF REPORT")
    print("=" * 60)

    return {
        'investigation_id': inv.investigation_id,
        'report_generated': datetime.now().isoformat(),
        'findings_count': len(inv.findings),
        'actions_count': len(inv.actions_taken),
        'risk_level': inv.risk_level,
        'status': inv.status
    }

# =============================================================================
# CELL INITIALIZATION
# =============================================================================

print("SUCCESS: Investigation Summary & Dashboard System initialized")
print()
print("Available functions:")
print("  - generate_investigation_summary(investigation_id)")
print("  - create_investigation_dashboard()")
print("  - export_investigation_data(investigation_id, format='json')")
print("  - show_investigation_metrics()")
print("  - create_multi_view_dashboard()")
print("  - create_risk_assessment_view()")
print("  - generate_investigation_report()")

if WIDGETS_AVAILABLE:
    print("  - create_interactive_dashboard()  # Interactive widgets")
    print("  - create_summary_widget()  # Summary generator widget")
else:
    print("  [INFO] Interactive widgets not available - using text-based dashboards")

print()
print("Quick start:")
print("  create_investigation_dashboard()")
print("  generate_investigation_summary()")
print("  create_multi_view_dashboard()")
print()
print("Cell 7e Complete - Investigation Summary & Dashboard Ready")
