# @title Cell 2: BigQuery Configuration — Enterprise-Ready Multi-Project Setup
# Enterprise-grade BigQuery configuration with comprehensive table verification

import os
import logging
from typing import Dict, List, Optional, Any
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import time
from datetime import datetime
import sys
sys.path.append('.')  # Ensure current directory is in sys.path

# ==============================================================
# REGISTERED_CELLS: Core System Cell Registry
# ==============================================================
REGISTERED_CELLS = {
    "Cell_03_Investigation_Manager": "Investigation lifecycle and logging",
    "Cell_04_SQL_Abuse_Query_Executor": "Parameterized SQL execution for T&S use cases",
    "Cell_05_Trust_Safety_Analyzer": "OpenAI-based and fallback abuse classifier",
    "Cell_07a_System_Status_Dashboard": "Agent + system state visualizer",
    "Cell_07b_Agent_Launcher_Registry": "Agent definitions and natural language interface",
    "Cell_07c_Runtime_Manager": "Agent runtime executor and thread-safe status tracking"
}
print("\nREGISTERED_CELLS:")
for k, v in REGISTERED_CELLS.items():
    print(f"  {k}: {v}")
print()

# =============================================================================
# GLOBAL VARIABLES - RUNTIME DEPENDENCIES
# =============================================================================

# These will be set by this cell and checked by agent handlers
VERIFIED_TABLES: Dict[str, Dict[str, Any]] = {}
TABLES_VERIFIED: bool = False
# Note: ENVIRONMENT_READY is set by Cell 1, not Cell 2

# BigQuery clients (will be initialized)
bq_client: Optional[bigquery.Client] = None
bq_xilabs: Optional[bigquery.Client] = None
bq_analytics: Optional[bigquery.Client] = None
BIGQUERY_CLIENT: Optional[bigquery.Client] = None  # For Cell 4 compatibility

# =============================================================================
# TABLE SCHEMAS - CENTRALIZED COLUMN DEFINITIONS
# =============================================================================

# === AUTO-GENERATED TABLE_SCHEMAS (copy-paste into Cell 2) ===
TABLE_SCHEMAS = {
    "TTS Usage Partitioned": {
        "table_id": "xi-labs.xi_prod.tts_usage_partitioned",
        "columns": {
            "id": "id",
            "timestamp": "timestamp",
            "ip": "ip",
            "user_uid": "user_uid",
            "user_email": "user_email",
            "subscription_name": "subscription_name",
            "is_free": "is_free",
            "is_anonymous": "is_anonymous",
            "quota_left": "quota_left",
            "quota_limit": "quota_limit",
            "text": "text",
            "text_length": "text_length",
            "api_key": "api_key",
            "voice_id": "voice_id",
            "voice_name": "voice_name",
            "voice_category": "voice_category",
            "request_mode": "request_mode",
            "status": "status",
            "latency_ms": "latency_ms",
            "total_request_time_ms": "total_request_time_ms",
            "audio_seconds": "audio_seconds",
            "audio_url": "audio_url",
            "cost": "cost",
            "charged_cost": "charged_cost",
            "voice_shared_from_uid": "voice_shared_from_uid",
            "model_id": "model_id",
            "stability": "stability",
            "similarity_boost": "similarity_boost",
            "style": "style",
            "use_speaker_boost": "use_speaker_boost",
            "optimize_streaming_latency": "optimize_streaming_latency",
            "workspace_id": "workspace_id",
            "vc_input_audio_url": "vc_input_audio_url",
            "financial_reward": "financial_reward",
            "characters_reward": "characters_reward",
            "moderation_status": "moderation_status",
            "source": "source",
            "detail": "detail",
            "seed": "seed",
            "safety_status": "safety_status",
            "endpoint": "endpoint",
            "get_auth_latency_secs": "get_auth_latency_secs",
            "get_auth_latency_secs_since_start": "get_auth_latency_secs_since_start",
            "get_voice_latency_secs": "get_voice_latency_secs",
            "get_voice_latency_secs_since_start": "get_voice_latency_secs_since_start",
            "get_subscription_latency_secs": "get_subscription_latency_secs",
            "get_subscription_latency_secs_since_start": "get_subscription_latency_secs_since_start",
            "get_voice_multiplier_latency_secs": "get_voice_multiplier_latency_secs",
            "get_voice_multiplier_latency_secs_since_start": "get_voice_multiplier_latency_secs_since_start",
            "get_pronunciation_dictionaries_latency_secs": "get_pronunciation_dictionaries_latency_secs",
            "get_pronunciation_dictionaries_latency_secs_since_start": "get_pronunciation_dictionaries_latency_secs_since_start",
            "calculate_cost_latency_secs": "calculate_cost_latency_secs",
            "calculate_cost_latency_secs_since_start": "calculate_cost_latency_secs_since_start",
            "get_context_blocks_latency_secs": "get_context_blocks_latency_secs",
            "get_context_blocks_latency_secs_since_start": "get_context_blocks_latency_secs_since_start",
            "get_normalizer_latency_secs": "get_normalizer_latency_secs",
            "get_normalizer_latency_secs_since_start": "get_normalizer_latency_secs_since_start",
            "normalize_text_latency_secs": "normalize_text_latency_secs",
            "normalize_text_latency_secs_since_start": "normalize_text_latency_secs_since_start",
            "to_first_audio_chunk_latency_secs": "to_first_audio_chunk_latency_secs",
            "character_multiplier_total": "character_multiplier_total",
            "character_multiplier_voice": "character_multiplier_voice",
            "character_multiplier_model": "character_multiplier_model",
            "read_id": "read_id",
            "trace": "trace",
            "enterprise_background_moderation_enabled": "enterprise_background_moderation_enabled",
            "isolated_prod_environment": "isolated_prod_environment",
            "full_dialogue_audio_url": "full_dialogue_audio_url",
            "request_idx": "request_idx",
            "request_resource_type": "request_resource_type",
            "request_source": "request_source",
            "region": "region",
            "device_platform": "device_platform",
            "tls_fingerprint": "tls_fingerprint",
            "reader_subscription_plan": "reader_subscription_plan",
            "financial_reward_withheld_tax_amount": "financial_reward_withheld_tax_amount",
        },
        "required_columns": ['id', 'timestamp', 'ip'],
        "description": "Core TTS generation data (partitioned)"
    },
    "Stripe Customers": {
        "table_id": "analytics-dev-421514.dbt_jho_intermediate.dim_stripe_customers",
        "columns": {
            "customer_id": "customer_id",
            "workspace_id": "workspace_id",
            "userid": "userid",
            "email": "email",
            "customer_name": "customer_name",
            "customer_created_date": "customer_created_date",
            "currency": "currency",
            "address_city": "address_city",
            "address_state": "address_state",
            "address_country": "address_country",
            "address_postal_code": "address_postal_code",
            "total_sales": "total_sales",
            "total_tax": "total_tax",
            "total_net_sales": "total_net_sales",
            "total_refunds": "total_refunds",
            "first_sale_date": "first_sale_date",
            "most_recent_sale_date": "most_recent_sale_date",
            "total_sales_usage_based_lifetime": "total_sales_usage_based_lifetime",
            "total_quantity_usage_based_lifetime": "total_quantity_usage_based_lifetime",
            "num_subscriptions": "num_subscriptions",
            "latest_subscription_tier": "latest_subscription_tier",
            "latest_stripe_ARR": "latest_stripe_ARR",
            "latest_stripe_subscription_ARR": "latest_stripe_subscription_ARR",
            "latest_subscription_created_date": "latest_subscription_created_date",
            "latest_subscription_status": "latest_subscription_status",
            "latest_subscription_user_cancellation_date": "latest_subscription_user_cancellation_date",
            "latest_subscription_end_date": "latest_subscription_end_date",
            "earliest_subscription_tier": "earliest_subscription_tier",
            "earliest_subscription_created_date": "earliest_subscription_created_date",
            "uncollectible_invoices_count": "uncollectible_invoices_count",
            "uncollectible_invoices_amount_due": "uncollectible_invoices_amount_due",
            "overdue_invoices_count": "overdue_invoices_count",
            "overdue_invoices_amount_due": "overdue_invoices_amount_due",
            "days_overdue": "days_overdue",
            "most_recent_billing_option": "most_recent_billing_option",
            "open_invoices_count": "open_invoices_count",
            "open_invoices_amount_due": "open_invoices_amount_due",
            "ever_enterprise_customer_flag": "ever_enterprise_customer_flag",
            "character_limit": "character_limit",
            "monthly_character_limit_prorated": "monthly_character_limit_prorated",
            "recurring_interval": "recurring_interval",
            "total_enterprise_invoices_paid": "total_enterprise_invoices_paid",
            "can_extend_character_limit": "can_extend_character_limit",
            "can_access_api_key": "can_access_api_key",
            "can_extend_voice_limit": "can_extend_voice_limit",
            "can_use_instant_voice_cloning": "can_use_instant_voice_cloning",
            "can_use_professional_voice_cloning": "can_use_professional_voice_cloning",
            "concurrency": "concurrency",
            "priority": "priority",
            "professional_voice_limit": "professional_voice_limit",
            "voice_limit": "voice_limit",
            "usage_based_price_per_k": "usage_based_price_per_k",
            "usage_based_price_per_mil": "usage_based_price_per_mil",
            "included_characters_price_per_mil": "included_characters_price_per_mil",
            "enterprise_subscription_end_date": "enterprise_subscription_end_date",
        },
        "required_columns": ['customer_id', 'workspace_id', 'userid'],
        "description": "Stripe customer dimension table"
    },
    "Stripe Charges": {
        "table_id": "analytics-dev-421514.dbt_jho_staging.stg_stripe__charge",
        "columns": {
            "id": "id",
            "amount": "amount",
            "amount_refunded": "amount_refunded",
            "application": "application",
            "application_fee_amount": "application_fee_amount",
            "balance_transaction_id": "balance_transaction_id",
            "bank_account_id": "bank_account_id",
            "billing_detail_address_city": "billing_detail_address_city",
            "billing_detail_address_country": "billing_detail_address_country",
            "billing_detail_address_line_1": "billing_detail_address_line_1",
            "billing_detail_address_line_2": "billing_detail_address_line_2",
            "billing_detail_address_postal_code": "billing_detail_address_postal_code",
            "billing_detail_address_state": "billing_detail_address_state",
            "billing_detail_email": "billing_detail_email",
            "billing_detail_name": "billing_detail_name",
            "billing_detail_phone": "billing_detail_phone",
            "calculated_statement_descriptor": "calculated_statement_descriptor",
            "captured": "captured",
            "card_id": "card_id",
            "connected_account_id": "connected_account_id",
            "created": "created",
            "currency": "currency",
            "customer_id": "customer_id",
            "description": "description",
            "destination": "destination",
            "failure_code": "failure_code",
            "failure_message": "failure_message",
            "fraud_details_stripe_report": "fraud_details_stripe_report",
            "fraud_details_user_report": "fraud_details_user_report",
            "invoice_id": "invoice_id",
            "livemode": "livemode",
            "metadata": "metadata",
            "on_behalf_of": "on_behalf_of",
            "outcome_network_status": "outcome_network_status",
            "outcome_reason": "outcome_reason",
            "outcome_risk_level": "outcome_risk_level",
            "outcome_risk_score": "outcome_risk_score",
            "outcome_seller_message": "outcome_seller_message",
            "outcome_type": "outcome_type",
            "paid": "paid",
            "payment_intent_id": "payment_intent_id",
            "payment_method_id": "payment_method_id",
            "receipt_email": "receipt_email",
            "receipt_number": "receipt_number",
            "receipt_url": "receipt_url",
            "refunded": "refunded",
            "shipping_address_city": "shipping_address_city",
            "shipping_address_country": "shipping_address_country",
            "shipping_address_line_1": "shipping_address_line_1",
            "shipping_address_line_2": "shipping_address_line_2",
            "shipping_address_postal_code": "shipping_address_postal_code",
            "shipping_address_state": "shipping_address_state",
            "shipping_carrier": "shipping_carrier",
            "shipping_name": "shipping_name",
            "shipping_phone": "shipping_phone",
            "shipping_tracking_number": "shipping_tracking_number",
            "source_id": "source_id",
            "source_transfer": "source_transfer",
            "statement_descriptor": "statement_descriptor",
            "status": "status",
            "transfer_data_destination": "transfer_data_destination",
            "transfer_group": "transfer_group",
            "transfer_id": "transfer_id",
            "rule_rule": "rule_rule",
        },
        "required_columns": ['id', 'amount', 'amount_refunded'],
        "description": "Stripe charge staging table"
    },
    "Device Fingerprint Cleaned": {
        "table_id": "eleven-team-safety.device_fingerprint_dataset.device_fingerprint_cleaned",
        "columns": {
            "browser_name": "browser_name",
            "browser_version": "browser_version",
            "platform": "platform",
            "device_fingerprint": "device_fingerprint",
            "first_seen_at_unix": "first_seen_at_unix",
            "last_seen_at_unix": "last_seen_at_unix",
            "raw_key_path": "raw_key_path",
            "user_id": "user_id",
            "key_fingerprint": "key_fingerprint",
        },
        "required_columns": ['browser_name', 'browser_version', 'platform'],
        "description": "Cleaned device fingerprint data"
    },
    "Reader Customer Event": {
        "table_id": "xi-labs.xi_prod.reader_customer_event",
        "columns": {
            "id": "id",
            "user_id": "user_id",
            "type": "type",
            "timestamp_unix": "timestamp_unix",
            "data_json": "data_json",
            "customer_signed_up_data": "customer_signed_up_data",
            "subscription_started_data": "subscription_started_data",
            "subscription_renewed_data": "subscription_renewed_data",
            "subscription_expired_data": "subscription_expired_data",
            "credits_allocated_data": "credits_allocated_data",
            "credits_spent_data": "credits_spent_data",
            "one_time_credits_purchased_data": "one_time_credits_purchased_data",
            "one_time_credits_purchase_cancelled_data": "one_time_credits_purchase_cancelled_data",
            "purchases_transfer_data": "purchases_transfer_data",
        },
        "required_columns": ['id', 'user_id', 'type'],
        "description": "Reader customer event data"
    },
    "Reader Explore Search": {
        "table_id": "xi-labs.xi_prod.reader_explore_search",
        "columns": {
            "id": "id",
            "timestamp": "timestamp",
            "ip": "ip",
            "user_id": "user_id",
            "workspace_id": "workspace_id",
            "page_size": "page_size",
            "search": "search",
            "page": "page",
            "result_count": "result_count",
        },
        "required_columns": ['id', 'timestamp', 'ip'],
        "description": "Reader explore search data"
    },
    "Reader Firestore Reads": {
        "table_id": "xi-labs.xi_prod.reader_firestore_reads",
        "columns": {
            "read_id": "read_id",
            "language": "language",
            "source_field": "source_field",
            "genres": "genres",
            "publishing_state": "publishing_state",
            "title": "title",
            "author": "author",
            "description": "description",
            "created_at_unix": "created_at_unix",
            "updated_at_unix": "updated_at_unix",
        },
        "required_columns": ['read_id', 'language', 'source_field'],
        "description": "Reader Firestore reads data"
    },
    "Reader Listened Audio Duration": {
        "table_id": "xi-labs.xi_prod.reader_listened_audio_duration",
        "columns": {
            "id": "id",
            "timestamp": "timestamp",
            "user_id": "user_id",
            "workspace_id": "workspace_id",
            "read_id": "read_id",
            "voice_id": "voice_id",
            "is_global": "is_global",
            "audio_duration": "audio_duration",
            "read_type": "read_type",
        },
        "required_columns": ['id', 'timestamp', 'user_id'],
        "description": "Reader listened audio duration data"
    },
    "Reader Payout": {
        "table_id": "xi-labs.xi_prod.reader_payout",
        "columns": {
            "timestamp": "timestamp",
            "net_payout_usd": "net_payout_usd",
            "tax_withheld_usd": "tax_withheld_usd",
            "user_id": "user_id",
            "author_id": "author_id",
            "user_country_code": "user_country_code",
            "author_country_code": "author_country_code",
            "payout_type": "payout_type",
        },
        "required_columns": ['timestamp', 'net_payout_usd', 'tax_withheld_usd'],
        "description": "Reader payout data"
    },
    "Reader Publishing Engaged User": {
        "table_id": "xi-labs.xi_prod.reader_publishing_engaged_user",
        "columns": {
            "timestamp": "timestamp",
            "read_id": "read_id",
            "user_id": "user_id",
            "user_email_domain": "user_email_domain",
            "user_ip_addresses": "user_ip_addresses",
            "user_device_ids": "user_device_ids",
            "user_agent": "user_agent",
        },
        "required_columns": ['timestamp', 'read_id', 'user_id'],
        "description": "Reader publishing engaged user data"
    },
    "Reader Read Daily Usage": {
        "table_id": "xi-labs.xi_prod.reader_read_daily_usage",
        "columns": {
            "id": "id",
            "timestamp": "timestamp",
            "read_id": "read_id",
            "listened_audio_duration": "listened_audio_duration",
            "net_new_users": "net_new_users",
            "net_new_engaged_users": "net_new_engaged_users",
            "net_new_payouts_usd": "net_new_payouts_usd",
            "net_new_discovered_users": "net_new_discovered_users",
        },
        "required_columns": ['id', 'timestamp', 'read_id'],
        "description": "Reader read daily usage data"
    },
    "Reader Revenuecat Event": {
        "table_id": "xi-labs.xi_prod.reader_revenuecat_event",
        "columns": {
            "id": "id",
            "event_timestamp_ms": "event_timestamp_ms",
            "product_id": "product_id",
            "period_type": "period_type",
            "purchased_at_ms": "purchased_at_ms",
            "environment": "environment",
            "transaction_id": "transaction_id",
            "original_transaction_id": "original_transaction_id",
            "app_user_id": "app_user_id",
            "aliases": "aliases",
            "original_app_user_id": "original_app_user_id",
            "currency": "currency",
            "price": "price",
            "price_in_purchased_currency": "price_in_purchased_currency",
            "subscriber_attributes_json": "subscriber_attributes_json",
            "store": "store",
            "type": "type",
            "entitlement_id": "entitlement_id",
            "entitlement_ids": "entitlement_ids",
            "presented_offering_id": "presented_offering_id",
            "expiration_at_ms": "expiration_at_ms",
            "offer_code": "offer_code",
            "app_id": "app_id",
            "country_code": "country_code",
            "is_family_share": "is_family_share",
            "auto_resume_at_ms": "auto_resume_at_ms",
            "cancel_reason": "cancel_reason",
            "expiration_reason": "expiration_reason",
            "tax_percentage": "tax_percentage",
            "commission_percentage": "commission_percentage",
            "new_product_id": "new_product_id",
            "transferred_from": "transferred_from",
            "transferred_to": "transferred_to",
            "is_trial_conversion": "is_trial_conversion",
            "takehome_percentage": "takehome_percentage",
            "created_at_unix": "created_at_unix",
            "processed_at_unix": "processed_at_unix",
        },
        "required_columns": ['id', 'event_timestamp_ms', 'product_id'],
        "description": "Reader RevenueCat event data"
    },
    "Reader Sent Audio Duration": {
        "table_id": "xi-labs.xi_prod.reader_sent_audio_duration",
        "columns": {
            "id": "id",
            "timestamp": "timestamp",
            "user_id": "user_id",
            "workspace_id": "workspace_id",
            "read_id": "read_id",
            "voice_id": "voice_id",
            "cache_hit": "cache_hit",
            "is_global": "is_global",
            "audio_duration": "audio_duration",
            "read_type": "read_type",
        },
        "required_columns": ['id', 'timestamp', 'user_id'],
        "description": "Reader sent audio duration data"
    },
    "Reader Sent Character Length": {
        "table_id": "xi-labs.xi_prod.reader_sent_character_length",
        "columns": {
            "id": "id",
            "timestamp": "timestamp",
            "user_id": "user_id",
            "workspace_id": "workspace_id",
            "read_id": "read_id",
            "voice_id": "voice_id",
            "cache_hit": "cache_hit",
            "is_global": "is_global",
            "character_length": "character_length",
            "read_type": "read_type",
        },
        "required_columns": ['id', 'timestamp', 'user_id'],
        "description": "Reader sent character length data"
    },
    "Reader User Config": {
        "table_id": "xi-labs.xi_prod.reader_user_config",
        "columns": {
            "user_id": "user_id",
            "timezone": "timezone",
            "daily_goal_minutes": "daily_goal_minutes",
            "version": "version",
            "timestamp": "timestamp",
        },
        "required_columns": ['user_id', 'timezone', 'daily_goal_minutes'],
        "description": "Reader user config data"
    },
}
# === END AUTO-GENERATED TABLE_SCHEMAS ===

# Copy and replace your TABLE_SCHEMAS in Cell 2 with the above block to ensure schemas match actual tables.

# Helper functions for schema consistency
def get_table_schema(table_name: str) -> dict:
    """Get schema definition for a table"""
    return TABLE_SCHEMAS.get(table_name, {})

def get_column_name(table_name: str, column_key: str) -> str:
    """Get the correct column name for a table"""
    schema = get_table_schema(table_name)
    return schema.get("columns", {}).get(column_key, column_key)

def get_required_columns(table_name: str) -> list:
    """Get required columns for a table"""
    schema = get_table_schema(table_name)
    return schema.get("required_columns", [])

def build_select_clause(table_name: str, columns: list = None) -> str:
    """Build a SELECT clause with correct column names"""
    schema = get_table_schema(table_name)
    if not schema:
        return "*"
    
    if columns is None:
        columns = schema.get("required_columns", ["*"])
    
    # Fix: If columns is empty, return '*'
    if not columns or "*" in columns:
        return "*"
    
    # Map column keys to actual column names
    actual_columns = []
    for col in columns:
        if col in schema.get("columns", {}):
            actual_columns.append(schema["columns"][col])
        else:
            actual_columns.append(col)  # Use as-is if not in schema
    
    return ", ".join(actual_columns)

# Make schema functions globally available
globals()['TABLE_SCHEMAS'] = TABLE_SCHEMAS
globals()['get_table_schema'] = get_table_schema
globals()['get_column_name'] = get_column_name
globals()['get_required_columns'] = get_required_columns
globals()['build_select_clause'] = build_select_clause

# =============================================================================
# AUTHENTICATION METHODS
# =============================================================================

def try_env_auth():
    """Try environment variable authentication"""
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        print("✅ Found GOOGLE_APPLICATION_CREDENTIALS")
        return True
    raise Exception("GOOGLE_APPLICATION_CREDENTIALS not found")

def try_default_auth():
    """Try application default credentials"""
    print("🔍 Trying application default credentials...")
    try:
        # Test with a simple client creation
        test_client = bigquery.Client()
        test_client.project  # This will fail if auth is not working
        print("✅ Application default credentials working")
        return True
    except Exception as e:
        raise Exception(f"Application default credentials failed: {e}")

def try_service_account_auth():
    """Try service account key authentication"""
    # Look for service account key files
    possible_paths = [
        "service-account-key.json",
        "key.json",
        "credentials.json",
        os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found service account key: {path}")
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
            return True
    
    raise Exception("No service account key files found")

def try_colab_auth():
    """Try Google Colab authentication"""
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Google Colab authentication successful")
        return True
    except ImportError:
        raise Exception("Not running in Google Colab")
    except Exception as e:
        raise Exception(f"Colab authentication failed: {e}")

# =============================================================================
# BIGQUERY CLIENT INITIALIZATION
# =============================================================================

def initialize_bigquery_clients():
    """Initialize BigQuery clients with enterprise-ready authentication"""
    global bq_client, bq_xilabs, bq_analytics, BIGQUERY_CLIENT
    
    try:
        print("🔗 Testing GCP connectivity (Enterprise Mode)...")
        
        # Try multiple authentication methods in order of preference
        auth_methods = [
            ("Environment Variables", try_env_auth),
            ("Service Account Key", try_service_account_auth),
            ("Application Default Credentials", try_default_auth),
            ("Colab Authentication", try_colab_auth)
        ]
        
        authenticated = False
        for auth_name, auth_func in auth_methods:
            try:
                print(f"   Trying {auth_name}...")
                auth_func()
                authenticated = True
                print(f"✅ {auth_name} successful")
                break
            except Exception as e:
                print(f"   ⚠️  {auth_name} failed: {str(e)}")
                continue
        
        if not authenticated:
            raise Exception("All authentication methods failed")
            
        # Initialize clients for different projects with correct locations
        bq_client = bigquery.Client(project="eleven-team-safety", location="us-central1")
        bq_xilabs = bigquery.Client(project="xi-labs", location="us-central1")
        bq_analytics = bigquery.Client(project="analytics-dev-421514", location="us-central1")
        
        # Quick connectivity test
        test_query = "SELECT 1 as test_value"
        bq_client.query(test_query).result()
        
        print("✅ GCP connectivity confirmed")
        print("✅ BigQuery clients initialized")
        
        # Run comprehensive table verification
        verify_all_tables()
        
        # Set global BIGQUERY_CLIENT for Cell 4 compatibility
        try:
            BIGQUERY_CLIENT = bq_client
            print("✅ BIGQUERY_CLIENT available globally")
        except Exception as e:
            print(f"⚠️ Could not set BIGQUERY_CLIENT globally: {e}")
        
    except Exception as e:
        print(f"❌ BigQuery initialization failed: {e}")
        print("💡 Enterprise troubleshooting:")
        print("   • Check GCP project permissions")
        print("   • Verify service account has BigQuery access")
        print("   • Ensure proper authentication setup")
        print("   • Check network connectivity to GCP")
        
        # Set BIGQUERY_CLIENT to None for Cell 4 compatibility
        try:
            BIGQUERY_CLIENT = None
            print("✅ BIGQUERY_CLIENT set to None globally")
        except Exception as e:
            print(f"⚠️ Could not set BIGQUERY_CLIENT to None: {e}")
        raise

# =============================================================================
# COMPREHENSIVE TABLE VERIFICATION
# =============================================================================

def verify_all_tables():
    """Verify all BigQuery tables and create VERIFIED_TABLES global dictionary"""
    global VERIFIED_TABLES, TABLES_VERIFIED
    
    print("\n🔍 COMPREHENSIVE TABLE VERIFICATION")
    print("=" * 50)
    
    # Clear existing verification
    VERIFIED_TABLES = {}
    
    # Use schema definitions for table configurations
    for table_name, schema_info in TABLE_SCHEMAS.items():
        config = {
            "name": table_name,
            "table_id": schema_info["table_id"],
            "critical": table_name in ["TTS Usage", "Classification Flags"],
            "description": schema_info["description"],
            "schema": schema_info,  # Include schema information
            "client": "bq_analytics" if "analytics-dev-421514" in schema_info["table_id"] else "bq_client"
        }
        
        print(f"\n📋 Verifying: {config['name']}")
        print(f"   Table ID: {config['table_id']}")
        print(f"   Required columns: {schema_info['required_columns']}")
        
        try:
            # Get the appropriate client
            client = globals().get(config['client'])
            if client is None:
                raise Exception(f"Client {config['client']} not available")
            
            # Test basic table access
            table = client.get_table(config['table_id'])
            print(f"   ✅ Table exists: {table.num_rows} rows")
            
            # Verify required columns exist
            actual_columns = [field.name for field in table.schema]
            missing_columns = []
            for required_col in schema_info['required_columns']:
                if required_col not in actual_columns:
                    missing_columns.append(required_col)
            
            if missing_columns:
                print(f"   ⚠️  Missing columns: {missing_columns}")
                print(f"   📋 Available columns: {actual_columns}")
            else:
                print(f"   ✅ All required columns present")
            
            # Test a simple query
            test_query = f"""
            SELECT {build_select_clause(table_name)}
            FROM `{config['table_id']}`
            LIMIT 1
            """
            
            query_job = client.query(test_query)
            results = query_job.result()
            print(f"   ✅ Query test successful")
            
            # Add to verified tables
            VERIFIED_TABLES[config['name']] = {
                'table_id': config['table_id'],
                'accessible': True,
                'client': client,
                'description': config['description'],
                'schema': schema_info,
                'actual_columns': actual_columns,
                'missing_columns': missing_columns
            }
            
        except Exception as e:
            print(f"   ❌ Verification failed: {str(e)}")
            
            # Add failed table to VERIFIED_TABLES with accessible=False
            VERIFIED_TABLES[config['name']] = {
                'table_id': config['table_id'],
                'accessible': False,
                'client': None,
                'description': config['description'],
                'schema': schema_info,
                'error': str(e)
            }
    
    # Summary
    accessible_tables = [name for name, info in VERIFIED_TABLES.items() if info['accessible']]
    failed_tables = [name for name, info in VERIFIED_TABLES.items() if not info['accessible']]
    
    print(f"\n📊 VERIFICATION SUMMARY")
    print(f"   ✅ Accessible tables: {len(accessible_tables)}")
    print(f"   ❌ Failed tables: {len(failed_tables)}")
    
    if accessible_tables:
        print(f"   Available for queries: {', '.join(accessible_tables)}")
    
    if failed_tables:
        print(f"   ⚠️  Unavailable: {', '.join(failed_tables)}")
    
    # Set global flag
    TABLES_VERIFIED = len(accessible_tables) > 0
    
    # Make VERIFIED_TABLES available globally
    globals()['VERIFIED_TABLES'] = VERIFIED_TABLES
    
    print(f"\n✅ Table verification complete - {len(accessible_tables)} tables ready")
    return VERIFIED_TABLES

# =============================================================================
# RUNTIME HEALTH CHECK FUNCTIONS
# =============================================================================

def test_runtime_integrity():
    """Test function to verify all required runtime dependencies are available"""
    print("\n🔍 RUNTIME INTEGRITY CHECK")
    print("=" * 50)
    
    # Import __main__ to check for variables in the main module scope
    import __main__
    
    # Test 1: Environment Ready
    try:
        environment_ready = getattr(__main__, 'ENVIRONMENT_READY', False)
        if environment_ready:
            print("✅ ENVIRONMENT_READY: True")
        else:
            print("❌ ENVIRONMENT_READY: False or not set")
            print("   💡 Fix: Run Cell 1 (Environment Setup)")
    except Exception as e:
        print(f"❌ ENVIRONMENT_READY: Error checking - {e}")
    
    # Test 2: VERIFIED_TABLES
    try:
        verified_tables = getattr(__main__, 'VERIFIED_TABLES', {})
        if verified_tables:
            accessible_count = sum(1 for t in verified_tables.values() if t.get('accessible', False))
            print(f"✅ VERIFIED_TABLES: {accessible_count}/{len(verified_tables)} tables accessible")
            
            # Check critical tables
            critical_tables = [name for name, info in verified_tables.items() 
                             if info.get('critical', False) and info.get('accessible', False)]
            print(f"✅ Critical tables accessible: {len(critical_tables)}")
        else:
            print("❌ VERIFIED_TABLES: None or empty")
            print("   💡 Fix: Run Cell 2 (BigQuery Configuration)")
    except Exception as e:
        print(f"❌ VERIFIED_TABLES: Error checking - {e}")
    
    # Test 3: BigQuery Clients
    try:
        clients = ['bq_client', 'bq_xilabs', 'bq_analytics']
        for client_name in clients:
            client = getattr(__main__, client_name, None)
            if client:
                print(f"✅ {client_name}: Available")
            else:
                print(f"❌ {client_name}: Not available")
                print("   💡 Fix: Run Cell 2 (BigQuery Configuration)")
    except Exception as e:
        print(f"❌ BigQuery clients: Error checking - {e}")
    
    # Test 4: main_system components
    try:
        main_system = getattr(__main__, 'main_system', None)
        if main_system:
            # Check bq_client
            if hasattr(main_system, 'bq_client') and main_system.bq_client:
                print("✅ main_system.bq_client: Available")
            else:
                print("❌ main_system.bq_client: Not set")
                print("   💡 Fix: Run Cell 2, then Cell 4 (SQL Interface)")
            
            # Check analyzer
            if hasattr(main_system, 'analyzer') and main_system.analyzer:
                print("✅ main_system.analyzer: Available")
            else:
                print("❌ main_system.analyzer: Not set")
                print("   💡 Fix: Run Cell 5 (Main Investigation System)")
        else:
            print("❌ main_system: Not available")
            print("   💡 Fix: Run Cell 5 (Main Investigation System)")
    except Exception as e:
        print(f"❌ main_system: Error checking - {e}")
    
    # Test 5: sql_executor
    try:
        sql_executor = getattr(__main__, 'sql_executor', None)
        if sql_executor:
            print("✅ sql_executor: Available")
        else:
            print("❌ sql_executor: Not available")
            print("   💡 Fix: Run Cell 4 (SQL Interface)")
    except Exception as e:
        print(f"❌ sql_executor: Error checking - {e}")
    
    # Overall assessment
    print(f"\n🎯 RUNTIME INTEGRITY ASSESSMENT")
    print("=" * 50)
    
    # Count issues
    issues_found = 0
    if not getattr(__main__, 'ENVIRONMENT_READY', False):
        issues_found += 1
    if not getattr(__main__, 'VERIFIED_TABLES', {}):
        issues_found += 1
    if not getattr(__main__, 'bq_client', None):
        issues_found += 1
    if not getattr(__main__, 'main_system', None):
        issues_found += 1
    if not getattr(__main__, 'sql_executor', None):
        issues_found += 1
    
    if issues_found == 0:
        print("🎉 ALL RUNTIME DEPENDENCIES HEALTHY")
        print("✅ System ready for agent execution")
    else:
        print(f"⚠️  {issues_found} RUNTIME ISSUES FOUND")
        print("❌ Agent execution may fail")
        print("\n💡 RECOMMENDED FIXES:")
        print("   1. Run Cell 1 (Environment Setup)")
        print("   2. Run Cell 2 (BigQuery Configuration)")
        print("   3. Run Cell 4 (SQL Interface)")
        print("   4. Run Cell 5 (Main Investigation System)")
        print("   5. Run Cell 7b (Agent Launcher)")
    
    return issues_found == 0

def check_runtime_health():
    """Quick runtime health check for agent execution"""
    issues = []
    warnings = []
    
    # Import __main__ to check for variables in the main module scope
    import __main__
    
    # Check ENVIRONMENT_READY (critical)
    if not getattr(__main__, 'ENVIRONMENT_READY', False):
        issues.append("ENVIRONMENT_READY not set - run Cell 1")
    
    # Check VERIFIED_TABLES (critical) 
    if not getattr(__main__, 'VERIFIED_TABLES', {}):
        issues.append("VERIFIED_TABLES not populated - run Cell 2")
    
    # Check BigQuery clients (critical)
    if not getattr(__main__, 'bq_client', None):
        issues.append("bq_client not available - run Cell 2")
    
    # Check BIGQUERY_CLIENT for Cell 4 compatibility (warning only)
    if not BIGQUERY_CLIENT:
        warnings.append("BIGQUERY_CLIENT not available - Cell 4 may show warnings")
    
    # Check main_system (critical)
    if not getattr(__main__, 'main_system', None):
        issues.append("main_system not available - run Cell 5")
    
    # Check sql_executor (critical)
    if not getattr(__main__, 'sql_executor', None):
        issues.append("sql_executor not available - run Cell 4")
    
    # Check INVESTIGATION_QUERY_TEMPLATES (warning only)
    if not getattr(__main__, 'INVESTIGATION_QUERY_TEMPLATES', None):
        warnings.append("INVESTIGATION_QUERY_TEMPLATES not available - Cell 6 testing may be limited")
    
    # Check agent_registry (warning only)
    if not getattr(__main__, 'agent_registry', None):
        warnings.append("agent_registry not available - agent features may be limited")
    
    # Return both issues and warnings
    return {'issues': issues, 'warnings': warnings}

def check_runtime_integrity():
    """
    STRICT RUNTIME INTEGRITY CHECK — DO NOT VIOLATE
    
    Ensures all critical components are present and valid before agent execution.
    Raises RuntimeError with specific instructions if any component is missing.
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    - Preserve system integrity — don't let errors silently pass
    """
    import __main__
    
    print("🔍 STRICT RUNTIME INTEGRITY CHECK")
    print("=" * 50)
    
    # Check 1: ENVIRONMENT_READY (Cell 1)
    ENVIRONMENT_READY = getattr(__main__, 'ENVIRONMENT_READY', False)
    if not ENVIRONMENT_READY:
        print("❌ CRITICAL: Environment not ready")
        raise RuntimeError("ENVIRONMENT_READY not set — run Cell 1 first")
    print("✅ Environment ready")
    
    # Check 2: VERIFIED_TABLES (Cell 2)
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
    if VERIFIED_TABLES is None:
        print("❌ CRITICAL: BigQuery tables not verified")
        raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
    
    if not isinstance(VERIFIED_TABLES, dict) or len(VERIFIED_TABLES) == 0:
        print("❌ CRITICAL: VERIFIED_TABLES empty or invalid")
        raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
    
    # Check critical tables are accessible
    critical_tables = ["TTS Usage"]  # Only TTS Usage is truly critical - Classification Flags is optional with graceful degradation
    for table_name in critical_tables:
        if table_name not in VERIFIED_TABLES:
            print(f"❌ CRITICAL: Missing table {table_name}")
            raise RuntimeError(f"Table {table_name} not verified — run Cell 2 first")
        
        table_info = VERIFIED_TABLES[table_name]
        if not table_info.get('accessible', False):
            print(f"❌ CRITICAL: Table {table_name} not accessible")
            error_msg = table_info.get('error', 'Unknown error')
            raise RuntimeError(f"Table {table_name} not accessible — {error_msg} — check Cell 2 verification")
    
    # Check optional tables (warning only)
    optional_tables = ["Classification Flags"]
    for table_name in optional_tables:
        if table_name not in VERIFIED_TABLES:
            print(f"⚠️  Optional table {table_name} not configured")
        else:
            table_info = VERIFIED_TABLES[table_name]
            if not table_info.get('accessible', False):
                print(f"⚠️  Optional table {table_name} not accessible")
                error_msg = table_info.get('error', 'Unknown error')
                print(f"    Error: {error_msg}")
                print(f"    Impact: Enhanced detection features will be disabled")
            else:
                print(f"✅ Optional table {table_name} accessible")
    
    print("✅ BigQuery tables verified and accessible")
    
    # Check 3: BigQuery clients (Cell 2)
    bq_client = getattr(__main__, 'bq_client', None)
    if bq_client is None:
        print("❌ CRITICAL: BigQuery client missing")
        raise RuntimeError("BigQuery client missing — run Cell 2 first")
    print("✅ BigQuery client available")
    
    # Check 4: main_system (Cell 5)
    main_system = getattr(__main__, 'main_system', None)
    if main_system is None:
        print("❌ CRITICAL: Main system missing")
        raise RuntimeError("Main investigation system missing — run Cell 5 first")
    
    # Check 4a: main_system.bq_client
    if not hasattr(main_system, "bq_client") or main_system.bq_client is None:
        print("❌ CRITICAL: main_system.bq_client missing")
        raise RuntimeError("BigQuery client missing in main_system — run Cell 5 first")
    print("✅ main_system.bq_client available")
    
    # Check 4b: main_system.analyzer
    if not hasattr(main_system, "analyzer") or main_system.analyzer is None:
        print("❌ CRITICAL: main_system.analyzer missing")
        raise RuntimeError("AI analyzer missing in main_system — run Cell 5 first")
    
    # Check 4c: main_system.analyzer.openai_client
    if not hasattr(main_system.analyzer, 'openai_client') or not main_system.analyzer.openai_client:
        print("❌ CRITICAL: OpenAI client missing in analyzer")
        raise RuntimeError("OpenAI client missing in analyzer — run Cell 5 first")
    print("✅ main_system.analyzer and OpenAI client available")
    
    # Check 5: sql_executor (Cell 4)
    sql_executor = getattr(__main__, 'sql_executor', None)
    if sql_executor is None:
        print("❌ CRITICAL: SQL executor missing")
        raise RuntimeError("SQL executor missing — run Cell 4 first")
    print("✅ SQL executor available")
    
    # Check 6: investigation_manager (Cell 3)
    investigation_manager = getattr(__main__, 'investigation_manager', None)
    if investigation_manager is None:
        print("❌ CRITICAL: Investigation manager missing")
        raise RuntimeError("Investigation manager missing — run Cell 3 first")
    print("✅ Investigation manager available")
    
    # Check 7: Schema system
    TABLE_SCHEMAS = getattr(__main__, 'TABLE_SCHEMAS', None)
    if TABLE_SCHEMAS is None:
        print("❌ CRITICAL: Schema system missing")
        raise RuntimeError("Schema system missing — run Cell 2 first")
    print("✅ Schema system available")
    
    # Check 8: agent_registry (Cell 7b)
    agent_registry = getattr(__main__, 'agent_registry', None)
    if agent_registry is None:
        print("❌ WARNING: Agent registry missing")
        print("⚠️  Agent features limited — run Cell 7b for full functionality")
    else:
        print("✅ Agent registry available")
    
    print("\n🎯 RUNTIME INTEGRITY CHECK PASSED")
    print("✅ All critical components verified and available")
    print("✅ System ready for REAL data investigation")
    
    return True

def check_agent_specific_requirements(agent_name: str):
    """
    Check agent-specific table and dependency requirements
    
    ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
    - NO mock data, fake records, or simulation
    - Ensure all execution is based on REAL data and logic
    - Validate agent-specific requirements
    """
    import __main__
    
    print(f"🔍 CHECKING AGENT-SPECIFIC REQUIREMENTS: {agent_name}")
    print("=" * 50)
    
    VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', {})
    
    # Define agent-specific table requirements with graceful degradation
    agent_requirements = {
        "scam_agent": {
            "required_tables": ["TTS Usage"],
            "optional_tables": ["Classification Flags"],
            "description": "Scam detection requires TTS data. Classification Flags enables enhanced detection with ML scores.",
            "graceful_degradation": True
        },
        "exploratory_agent": {
            "required_tables": ["TTS Usage"],
            "optional_tables": [],
            "description": "Exploratory analysis requires TTS data only",
            "graceful_degradation": False
        },
        "email_network_agent": {
            "required_tables": ["TTS Usage"],
            "optional_tables": ["Classification Flags"],
            "description": "Email network analysis requires TTS data. Classification Flags enables enhanced risk scoring.",
            "graceful_degradation": True
        }
    }
    
    if agent_name not in agent_requirements:
        print(f"⚠️  No specific requirements defined for {agent_name}")
        return True
    
    requirements = agent_requirements[agent_name]
    print(f"📋 Agent: {agent_name}")
    print(f"📋 Description: {requirements['description']}")
    print(f"📋 Graceful Degradation: {'✅ Yes' if requirements['graceful_degradation'] else '❌ No'}")
    
    missing_required = []
    inaccessible_required = []
    missing_optional = []
    inaccessible_optional = []
    
    # Check required tables
    for table_name in requirements["required_tables"]:
        if table_name not in VERIFIED_TABLES:
            missing_required.append(table_name)
            print(f"❌ Missing required table: {table_name}")
        else:
            table_info = VERIFIED_TABLES[table_name]
            if not table_info.get('accessible', False):
                inaccessible_required.append(table_name)
                error_msg = table_info.get('error', 'Unknown error')
                print(f"❌ Inaccessible required table: {table_name} - {error_msg}")
            else:
                print(f"✅ Available required table: {table_name}")
    
    # Check optional tables
    for table_name in requirements["optional_tables"]:
        if table_name not in VERIFIED_TABLES:
            missing_optional.append(table_name)
            print(f"⚠️  Missing optional table: {table_name}")
        else:
            table_info = VERIFIED_TABLES[table_name]
            if not table_info.get('accessible', False):
                inaccessible_optional.append(table_name)
                error_msg = table_info.get('error', 'Unknown error')
                print(f"⚠️  Inaccessible optional table: {table_name} - {error_msg}")
            else:
                print(f"✅ Available optional table: {table_name}")
    
    # Check for critical failures (missing required tables)
    if missing_required or inaccessible_required:
        print(f"\n❌ AGENT {agent_name.upper()} CRITICAL REQUIREMENTS NOT MET")
        
        if missing_required:
            print(f"Missing required tables: {', '.join(missing_required)}")
        
        if inaccessible_required:
            print(f"Inaccessible required tables: {', '.join(inaccessible_required)}")
        
        print("\n💡 CRITICAL SOLUTIONS:")
        print("1. Re-run Cell 2 (BigQuery Configuration) to verify tables")
        print("2. Check GCP permissions for the following required tables:")
        
        for table_name in missing_required + inaccessible_required:
            if table_name in TABLE_SCHEMAS:
                table_id = TABLE_SCHEMAS[table_name]["table_id"]
                print(f"   • {table_name}: {table_id}")
        
        raise RuntimeError(f"Agent {agent_name} critical requirements not met — missing required tables")
    
    # Check for optional table issues (warnings only)
    if missing_optional or inaccessible_optional:
        print(f"\n⚠️  OPTIONAL TABLES NOT AVAILABLE FOR {agent_name.upper()}")
        
        if missing_optional:
            print(f"Missing optional tables: {', '.join(missing_optional)}")
        
        if inaccessible_optional:
            print(f"Inaccessible optional tables: {', '.join(inaccessible_optional)}")
        
        if requirements['graceful_degradation']:
            print("\n✅ GRACEFUL DEGRADATION AVAILABLE")
            print("   Agent will automatically fall back to basic functionality")
            print("   Enhanced features will be disabled")
            
            # Provide specific degradation information
            if agent_name == "scam_agent":
                print("   • Enhanced detection: ❌ Disabled (uses basic pattern matching)")
                print("   • ML risk scores: ❌ Disabled")
                print("   • Basic scam detection: ✅ Available")
            elif agent_name == "email_network_agent":
                print("   • Enhanced risk scoring: ❌ Disabled")
                print("   • ML-based network analysis: ❌ Disabled")
                print("   • Basic network analysis: ✅ Available")
        else:
            print("\n⚠️  NO GRACEFUL DEGRADATION AVAILABLE")
            print("   Agent functionality may be limited")
        
        print("\n💡 OPTIONAL SOLUTIONS:")
        print("1. Check GCP permissions for the following optional tables:")
        
        for table_name in missing_optional + inaccessible_optional:
            if table_name in TABLE_SCHEMAS:
                table_id = TABLE_SCHEMAS[table_name]["table_id"]
                print(f"   • {table_name}: {table_id}")
        
        print("2. Contact your GCP administrator for access to optional tables")
        print("3. Agent will work with reduced functionality")
    
    print(f"\n✅ BASIC REQUIREMENTS MET FOR {agent_name.upper()}")
    print("✅ Agent ready for execution with REAL data")
    
    if missing_optional or inaccessible_optional:
        if requirements['graceful_degradation']:
            print("✅ Graceful degradation will handle missing optional tables")
        else:
            print("⚠️  Some functionality may be limited due to missing optional tables")
    
    return True

# Make functions globally available
globals()['check_runtime_integrity'] = check_runtime_integrity
globals()['check_agent_specific_requirements'] = check_agent_specific_requirements

# =============================================================================
# ENTERPRISE ENVIRONMENT VALIDATOR
# =============================================================================

def validate_enterprise_environment():
    """Validate enterprise environment setup"""
    print("\n🏢 ENTERPRISE ENVIRONMENT VALIDATION")
    print("=" * 50)
    
    checks = [
        ("Python Environment", check_python_env),
        ("Network Connectivity", check_network_access),
        ("GCP Authentication", check_gcp_auth),
        ("BigQuery Access", check_bigquery_access),
        ("Required Libraries", check_required_libraries)
    ]
    
    passed = 0
    for check_name, check_func in checks:
        try:
            print(f"\n🔍 {check_name}...")
            check_func()
            print(f"   ✅ {check_name} OK")
            passed += 1
        except Exception as e:
            print(f"   ❌ {check_name} FAILED: {e}")
    
    print(f"\n📊 Environment Check: {passed}/{len(checks)} passed")
    
    if passed == len(checks):
        print("🎉 Enterprise environment ready!")
        # Note: ENVIRONMENT_READY is managed by Cell 1, not Cell 2
    else:
        print("⚠️  Environment issues detected")
        print("💡 Resolve above issues before proceeding")

def check_python_env():
    """Check Python environment"""
    import sys
    if sys.version_info < (3, 7):
        raise Exception("Python 3.7+ required")

def check_network_access():
    """Check network connectivity"""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
    except OSError:
        raise Exception("Network connectivity issue")

def check_gcp_auth():
    """Check GCP authentication"""
    try:
        client = bigquery.Client()
        client.project  # This will fail if not authenticated
    except Exception as e:
        raise Exception(f"GCP authentication failed: {e}")

def check_bigquery_access():
    """Check BigQuery access"""
    try:
        client = bigquery.Client()
        query = "SELECT 1 as test"
        client.query(query).result()
    except Exception as e:
        raise Exception(f"BigQuery access failed: {e}")

def check_required_libraries():
    """Check required libraries"""
    required_libs = ['google.cloud.bigquery', 'pandas', 'numpy']
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            raise Exception(f"Required library missing: {lib}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_available_tables():
    """List all available tables with their status"""
    if not VERIFIED_TABLES:
        print("❌ No verified tables available. Run Cell 2 first.")
        return
    
    print("\n📊 AVAILABLE TABLES")
    print("=" * 60)
    
    for name, info in VERIFIED_TABLES.items():
        status = "✅" if info.get('accessible', False) else "❌"
        critical = "🔴" if info.get('critical', False) else "⚪"
        
        print(f"{status} {critical} {name}")
        print(f"   Table ID: {info.get('table_id', 'N/A')}")
        print(f"   Status: {'Accessible' if info.get('accessible', False) else 'Failed'}")
        
        if info.get('accessible', False):
            print(f"   Rows: {info.get('row_count', 'N/A'):,}")
            print(f"   Size: {info.get('size_mb', 'N/A'):.1f} MB")
        else:
            print(f"   Error: {info.get('error', 'Unknown')}")
        
        print(f"   Description: {info.get('description', 'N/A')}")
        print()

def get_table_info(table_name: str):
    """Get information about a specific table"""
    if table_name not in VERIFIED_TABLES:
        print(f"❌ Table '{table_name}' not found")
        print(f"Available tables: {list(VERIFIED_TABLES.keys())}")
        return None
    
    table_info = VERIFIED_TABLES[table_name]
    
    if not table_info.get('accessible', False):
        print(f"❌ Table '{table_name}' not accessible")
        print(f"Error: {table_info.get('error', 'Unknown')}")
        return None
    
    return table_info

def test_table_query(table_name: str):
    """Test a simple query on a table"""
    table_info = get_table_info(table_name)
    if not table_info:
        return False
    
    try:
        client = table_info['client']
        table_id = table_info['table_id']
        
        query = f"SELECT COUNT(*) as row_count FROM `{table_id}` LIMIT 1"
        result = client.query(query).result()
        
        for row in result:
            print(f"✅ Query successful - Row count: {row.row_count:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

# =============================================================================
# OPENAI API KEY MANAGEMENT
# =============================================================================

def setup_openai_key_only():
    """Setup OpenAI API key only (for cases where BigQuery is already configured)"""
    try:
        import getpass
        api_key = getpass.getpass("🔑 Enter OpenAI API Key: ")
        
        if api_key and api_key.strip():
            os.environ['OPENAI_API_KEY'] = api_key.strip()
            print("✅ OpenAI API key configured")
            
            # Test the API key
            try:
                from openai import OpenAI
                
                # Simple test with new OpenAI 1.0+ API format
                client = OpenAI(api_key=api_key.strip())
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                print("✅ OpenAI API key validated")
                
            except Exception as e:
                print(f"⚠️  OpenAI API key validation failed: {e}")
                print("   Key saved but may not be valid")
        else:
            print("❌ No API key provided")
            
    except Exception as e:
        print(f"❌ Failed to setup OpenAI API key: {e}")

def reset_api_key():
    """Reset OpenAI API key"""
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    print("🔑 OpenAI API key cleared. Re-run Cell 2 to enter new key.")

def show_current_api_status():
    """Show current API key status"""
    if 'OPENAI_API_KEY' in os.environ:
        key = os.environ['OPENAI_API_KEY']
        masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        print(f"✅ OpenAI API key configured: {masked_key}")
    else:
        print("❌ No OpenAI API key configured")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("🚀 CELL 2: BIGQUERY CONFIGURATION")
    print("=" * 60)
    
    # Check if credentials are already set
    if 'OPENAI_API_KEY' in os.environ:
        print("✅ All credentials found in environment")
        print("💡 To re-enter API key, run: reset_api_key() then re-run Cell 2")
        initialize_bigquery_clients()
    else:
        print("🔐 OpenAI API key needed - please complete setup below")
        print("💡 GCP BigQuery credentials assumed already configured")
        setup_openai_key_only()
    
    print("\n" + "=" * 60)
    print("Cell 2 Complete - BigQuery clients ready with verified table access")
    print("Global variables available: bq_client, bq_xilabs, bq_analytics, VERIFIED_TABLES")
    print("Run list_available_tables() to see all accessible tables")
    print("Run test_runtime_integrity() to verify all dependencies")
    print("\nAPI Key Management:")
    print("  • show_current_api_status() - Check current API key")
    print("  • reset_api_key() - Clear API key and re-prompt")
    print("\n💡 Quick Commands:")
    print("   • check_runtime_integrity() - STRICT validation of all components")
    print("   • check_notebook_runtime() - Validate notebook environment and shared state")
    print("   • validate_notebook_execution_order() - Check cell execution sequence")
    print("   • quick_system_status() - Check what's missing")
    print("   • check_runtime_health() - Detailed health check")
    print("   • get_table_schema('TTS Usage') - Get schema for any table")

# Run main initialization
if __name__ == "__main__":
    main()
else:
    # When imported/executed in notebook
    main()

# =============================================================================
# NOTEBOOK INTEGRATION - SHARED SYSTEM AWARENESS
# =============================================================================

print("\n🔗 NOTEBOOK INTEGRATION CHECK...")

# Check if we're in a unified notebook system
if 'NOTEBOOK_STATE' in globals():
    print("✅ Unified notebook system detected")
    
    # Integrate with shared system
    if 'setup_shared_bigquery' in globals():
        print("✅ Integrating with shared notebook system...")
        try:
            setup_shared_bigquery()
            print("✅ BigQuery clients integrated with shared system")
            
            # Update global references to use shared system
            if 'bq_client' in NOTEBOOK_STATE.get('shared_variables', {}):
                bq_client = NOTEBOOK_STATE['shared_variables']['bq_client']
                bq_xilabs = NOTEBOOK_STATE['shared_variables']['bq_xilabs']
                bq_analytics = NOTEBOOK_STATE['shared_variables']['bq_analytics']
                print("✅ Using shared BigQuery clients")
            
        except Exception as e:
            print(f"⚠️  Shared system integration failed: {e}")
            print("   Continuing with standalone BigQuery clients")
        
    else:
        print("⚠️  Shared system functions not available")
        print("   For full integration, run: exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")
        
else:
    print("⚠️  Running in standalone mode")
    print("   For full notebook integration, run:")
    print("   exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")

print("\n🎯 CELL 2 READY - BigQuery Configuration available")
print("   • Schema System: TABLE_SCHEMAS with centralized column definitions")
print("   • Helper Functions: get_table_schema(), get_column_name(), build_select_clause()")
print("   • Runtime Integrity: check_runtime_integrity() - STRICT system validation")
print("   • Standalone: bq_client, bq_xilabs, bq_analytics, VERIFIED_TABLES")
print("   • Integrated: Uses shared BigQuery clients")
print("   • Runtime Health: test_runtime_integrity() function available")
print("   • System Status: quick_system_status() function available")
print(f"   • BIGQUERY_CLIENT: {'✅ Available' if BIGQUERY_CLIENT else '❌ Not Available'}")
print("\n💡 Quick Commands:")
print("   • check_runtime_integrity() - STRICT validation of all components")
print("   • check_notebook_runtime() - Validate notebook environment and shared state")
print("   • validate_notebook_execution_order() - Check cell execution sequence")
print("   • quick_system_status() - Check what's missing")
print("   • check_runtime_health() - Detailed health check")
print("   • get_table_schema('TTS Usage') - Get schema for any table")
print("\n📋 Schema System: See SCHEMA_SYSTEM_GUIDE.md for complete documentation")
print("⚠️  STRICT INVESTIGATOR GUIDELINES — Use check_runtime_integrity() before agents")

def quick_system_status():
    """Quick system status check with actionable guidance"""
    import __main__
    
    print("🔍 SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # Check each component
    components = [
        ("ENVIRONMENT_READY", "Cell 1 - Environment Setup"),
        ("VERIFIED_TABLES", "Cell 2 - BigQuery Configuration"),
        ("bq_client", "Cell 2 - BigQuery Client"),
        ("BIGQUERY_CLIENT", "Cell 2 - BigQuery Client (compatibility)"),
        ("investigation_manager", "Cell 3 - Investigation Management"),
        ("sql_executor", "Cell 4 - SQL Interface"),
        ("INVESTIGATION_QUERY_TEMPLATES", "Cell 4 - Query Templates"),
        ("main_system", "Cell 5 - Main Investigation System"),
        ("agent_registry", "Cell 7b - Agent Launcher"),
        ("agent_runtime_manager", "Cell 7c - Runtime Manager"),
    ]
    
    missing_components = []
    
    for var_name, description in components:
        value = getattr(__main__, var_name, None)
        if value is None:
            status = "❌ Missing"
            missing_components.append((var_name, description))
        elif var_name == "VERIFIED_TABLES" and (not isinstance(value, dict) or len(value) == 0):
            status = "❌ Empty"
            missing_components.append((var_name, description))
        else:
            status = "✅ Available"
        
        print(f"{status:12} {var_name:25} ({description})")
    
    print()
    if missing_components:
        print("🚨 ACTION REQUIRED:")
        print("Run these cells in order to fix missing components:")
        
        cell_order = {
            "ENVIRONMENT_READY": "Cell 1",
            "VERIFIED_TABLES": "Cell 2", 
            "bq_client": "Cell 2",
            "BIGQUERY_CLIENT": "Cell 2",
            "investigation_manager": "Cell 3",
            "sql_executor": "Cell 4",
            "INVESTIGATION_QUERY_TEMPLATES": "Cell 4", 
            "main_system": "Cell 5",
            "agent_registry": "Cell 7b",
            "agent_runtime_manager": "Cell 7c",
        }
        
        cells_needed = sorted(set(cell_order[var] for var, _ in missing_components))
        for cell in cells_needed:
            print(f"  • {cell}")
    else:
        print("✅ All components available!")
        print("System is ready for investigation work.")

# =============================================================================
# NOTEBOOK RUNTIME VALIDATION
# =============================================================================

def check_notebook_runtime():
    """
    Comprehensive notebook runtime validation to ensure proper execution environment.
    
    This function validates:
    1. If we're in a notebook cell-based execution environment
    2. If global state is shared across cells
    3. If required cells are run in sequence and persistent
    4. Provides clear guidance if issues are detected
    
    Raises:
        RuntimeError: If not running in notebook environment
    """
    import sys
    import builtins
    
    print("🔍 NOTEBOOK RUNTIME VALIDATION")
    print("=" * 50)
    
    # 1. Check if running in notebook environment
    try:
        from IPython import get_ipython
        
        if get_ipython() is None:
            raise RuntimeError("❌ NOT_RUNNING_IN_NOTEBOOK: IPython not available")
        
        shell = get_ipython().__class__.__name__
        
        if shell == 'ZMQInteractiveShell':
            print("✅ Running in Jupyter Notebook environment")
        elif shell == 'TerminalInteractiveShell':
            print("⚠️  Running in IPython terminal - some features may not work")
        else:
            print(f"⚠️  Running in {shell} - may not be full notebook environment")
            
        # Check for Google Colab specifically
        try:
            import google.colab
            print("✅ Google Colab environment detected")
        except ImportError:
            print("ℹ️  Not in Google Colab (this is fine)")
            
    except ImportError:
        raise RuntimeError("❌ NOT_RUNNING_IN_NOTEBOOK: You're not in a notebook — some core functionality may break.")
    
    # 2. Check for shared global state
    print("\n🔍 CHECKING SHARED GLOBAL STATE...")
    
    # Import __main__ to check for globals across cells
    import __main__
    
    # Define required globals for each cell
    required_globals = {
        "Cell 1": ["ENVIRONMENT_READY"],
        "Cell 2": ["VERIFIED_TABLES", "bq_client", "BIGQUERY_CLIENT"],
        "Cell 3": ["investigation_manager"],
        "Cell 4": ["sql_executor"],
        "Cell 5": ["main_system"]
    }
    
    missing_by_cell = {}
    all_missing = []
    
    for cell_name, variables in required_globals.items():
        missing = []
        for var_name in variables:
            if not hasattr(__main__, var_name) or getattr(__main__, var_name) is None:
                missing.append(var_name)
                all_missing.append(var_name)
        
        if missing:
            missing_by_cell[cell_name] = missing
    
    # 3. Report results
    if not missing_by_cell:
        print("✅ All required global variables found across cells")
        print("✅ Shared state verified: Cell execution sequence appears correct")
        
        # Additional validation
        environment_ready = getattr(__main__, 'ENVIRONMENT_READY', False)
        verified_tables = getattr(__main__, 'VERIFIED_TABLES', {})
        
        if environment_ready:
            print("✅ ENVIRONMENT_READY: Cell 1 completed successfully")
        else:
            print("⚠️  ENVIRONMENT_READY is False - Cell 1 may need to be re-run")
            
        if verified_tables:
            print(f"✅ VERIFIED_TABLES: {len(verified_tables)} tables accessible")
        else:
            print("⚠️  VERIFIED_TABLES is empty - Cell 2 may need to be re-run")
            
    else:
        print("⚠️  Missing global state variables detected:")
        for cell_name, missing_vars in missing_by_cell.items():
            print(f"   {cell_name}: {missing_vars}")
        
        print("\n💡 RESOLUTION STEPS:")
        print("   1. Ensure you're running in a Jupyter notebook (.ipynb file)")
        print("   2. Run cells in the correct sequence:")
        print("      • Cell 1: Environment Setup")
        print("      • Cell 2: BigQuery Configuration")
        print("      • Cell 3: Investigation Management")
        print("      • Cell 4: SQL Interface")
        print("      • Cell 5: Main Investigation System")
        print("   3. Check for any error messages in cell outputs")
        print("   4. If issues persist, restart kernel and run cells again")
        
        # Specific guidance based on what's missing
        if "ENVIRONMENT_READY" in all_missing:
            print("\n🔧 SPECIFIC: Run Cell 1 first to set up environment")
        if any(var in all_missing for var in ["VERIFIED_TABLES", "bq_client"]):
            print("🔧 SPECIFIC: Run Cell 2 to configure BigQuery access")
        if "main_system" in all_missing:
            print("🔧 SPECIFIC: Run Cell 5 to initialize main investigation system")
    
    # 4. Additional runtime checks
    print("\n🔍 ADDITIONAL RUNTIME CHECKS...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor} (compatible)")
    else:
        print(f"⚠️  Python {python_version.major}.{python_version.minor} (may have compatibility issues)")
    
    # Check for required packages
    required_packages = ['google.cloud.bigquery', 'openai', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('.', '/').split('/')[0])
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {missing_packages}")
        print("   Run: pip install google-cloud-bigquery openai pandas numpy")
    else:
        print("✅ All required packages available")
    
    # 5. Memory and execution context
    print("\n🔍 EXECUTION CONTEXT...")
    
    # Check if variables are in the same namespace
    current_globals = set(globals().keys())
    main_globals = set(dir(__main__))
    
    shared_vars = current_globals.intersection(main_globals)
    if len(shared_vars) > 10:  # Reasonable threshold
        print(f"✅ Shared namespace detected ({len(shared_vars)} common variables)")
    else:
        print("⚠️  Limited shared namespace - may indicate import/execution issues")
    
    print("\n" + "=" * 50)
    
    if not missing_by_cell:
        print("🎯 NOTEBOOK RUNTIME VALIDATION PASSED")
        print("✅ Environment is ready for agent execution")
    else:
        print("⚠️  NOTEBOOK RUNTIME VALIDATION INCOMPLETE")
        print("   Follow resolution steps above before running agents")
    
    return len(missing_by_cell) == 0

def validate_notebook_execution_order():
    """
    Validate that cells were executed in the correct order.
    Returns True if order is correct, False otherwise.
    """
    import __main__
    
    print("🔍 VALIDATING CELL EXECUTION ORDER...")
    
    # Check execution order indicators
    order_checks = [
        ("Cell 1", "ENVIRONMENT_READY", "Environment setup"),
        ("Cell 2", "VERIFIED_TABLES", "BigQuery configuration"),
        ("Cell 3", "investigation_manager", "Investigation management"),
        ("Cell 4", "sql_executor", "SQL interface"),
        ("Cell 5", "main_system", "Main investigation system")
    ]
    
    execution_status = []
    
    for cell_name, check_var, description in order_checks:
        if hasattr(__main__, check_var) and getattr(__main__, check_var) is not None:
            execution_status.append(f"✅ {cell_name}: {description}")
        else:
            execution_status.append(f"❌ {cell_name}: {description} - NOT EXECUTED")
    
    for status in execution_status:
        print(f"   {status}")
    
    # Check if all cells up to Cell 5 are executed
    required_vars = ["ENVIRONMENT_READY", "VERIFIED_TABLES", "investigation_manager", "sql_executor", "main_system"]
    all_executed = all(hasattr(__main__, var) and getattr(__main__, var) is not None for var in required_vars)
    
    if all_executed:
        print("✅ All prerequisite cells executed in correct order")
    else:
        print("❌ Cells not executed in correct order")
        print("💡 Execute cells sequentially: Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5")
    
    return all_executed

# =============================================================================
# SCHEMA AUTO-GENERATION SCRIPT (for copy-paste)
# =============================================================================
def print_actual_table_schemas():
    """Print a ready-to-paste TABLE_SCHEMAS block for all accessible tables using actual columns."""
    if not VERIFIED_TABLES:
        print("❌ No verified tables available. Run Cell 2 first.")
        return
    print("\n# === AUTO-GENERATED TABLE_SCHEMAS (copy-paste into Cell 2) ===\nTABLE_SCHEMAS = {")
    for name, info in VERIFIED_TABLES.items():
        if not info.get('accessible', False):
            continue
        table_id = info['table_id']
        actual_columns = info.get('actual_columns', [])
        # Use first 3 columns as required_columns, or all if fewer
        required_columns = actual_columns[:3] if len(actual_columns) >= 3 else actual_columns
        print(f'    "{name}": {{')
        print(f'        "table_id": "{table_id}",')
        print(f'        "columns": {{')
        for col in actual_columns:
            print(f'            "{col}": "{col}",')
        print(f'        }},')
        print(f'        "required_columns": {required_columns},')
        print(f'        "description": "{info.get("description", "")}"')
        print(f'    }},')
    print("}\n# === END AUTO-GENERATED TABLE_SCHEMAS ===\n")
    print("# Copy and replace your TABLE_SCHEMAS in Cell 2 with the above block to ensure schemas match actual tables.") 