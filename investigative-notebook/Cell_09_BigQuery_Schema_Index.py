# ================================================================
# 📦 Cell 09: BigQuery Schema Index Builder
# Purpose: Dynamically build GLOBAL_SCHEMA_INDEX across all projects
# Depends on: BigQuery auth (Cell 02)
# ================================================================

import pandas as pd
from google.cloud import bigquery
from google.auth import default
from typing import Optional
from datetime import datetime

GLOBAL_SCHEMA_INDEX: pd.DataFrame = pd.DataFrame()
GLOBAL_SCHEMA_LAST_UPDATED: Optional[datetime] = None

def build_global_schema_index(force_refresh: bool = False) -> pd.DataFrame:
    """
    Build comprehensive schema index by scanning all available BigQuery datasets
    Returns:
        pd.DataFrame with columns: project, dataset, table, column, data_type
    """
    global GLOBAL_SCHEMA_INDEX, GLOBAL_SCHEMA_LAST_UPDATED
    if not force_refresh and not GLOBAL_SCHEMA_INDEX.empty and GLOBAL_SCHEMA_LAST_UPDATED:
        time_since_update = (datetime.now() - GLOBAL_SCHEMA_LAST_UPDATED).total_seconds()
        if time_since_update < 300:  # 5 minutes cache
            print(f"✅ Using cached schema index ({len(GLOBAL_SCHEMA_INDEX)} columns)")
            return GLOBAL_SCHEMA_INDEX
    print("🔍 BUILDING GLOBAL SCHEMA INDEX")
    print("=" * 60)
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
            client = bigquery.Client(project=project_id)
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
    if all_schema_data:
        GLOBAL_SCHEMA_INDEX = pd.concat(all_schema_data, ignore_index=True)
        GLOBAL_SCHEMA_LAST_UPDATED = datetime.now()
        print(f"\n📊 GLOBAL SCHEMA INDEX SUMMARY:")
        print(f"   Total projects: {GLOBAL_SCHEMA_INDEX['project'].nunique()}")
        print(f"   Total datasets: {GLOBAL_SCHEMA_INDEX['dataset'].nunique()}")
        print(f"   Total tables: {GLOBAL_SCHEMA_INDEX['table'].nunique()}")
        print(f"   Total columns: {len(GLOBAL_SCHEMA_INDEX)}")
        print(f"\n✅ GLOBAL_SCHEMA_INDEX built successfully")
        return GLOBAL_SCHEMA_INDEX
    else:
        print(f"\n❌ No schema data found - check project access and authentication")
        return pd.DataFrame() 