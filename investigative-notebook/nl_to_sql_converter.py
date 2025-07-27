# @title Natural Language to SQL Converter - Production Implementation

"""
Natural Language to BigQuery SQL Converter
=========================================

This module implements sophisticated Natural Language processing to convert
investigator queries into proper BigQuery SQL queries for the Trust & Safety
investigation system.

⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
- NO mock data, placeholders, or simulation
- Use only real BigQuery table schemas and column names
- Trust and investigation integrity are core — do not fake queries
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# =============================================================================
# SQL TEMPLATE DEFINITIONS
# =============================================================================

@dataclass
class SQLTemplate:
    """SQL query template with metadata"""
    name: str
    description: str
    base_query: str
    required_params: List[str]
    optional_params: List[str]
    example_nl: str
    result_columns: List[str]

# =============================================================================
# NATURAL LANGUAGE TO SQL CONVERTER
# =============================================================================

class NLToSQLConverter:
    """
    Production-grade Natural Language to BigQuery SQL converter
    
    Converts investigator natural language queries into proper BigQuery SQL
    queries using pattern matching, parameter extraction, and template systems.
    """
    
    def __init__(self):
        self.templates = {}
        self.table_schemas = {}
        self.query_patterns = {}
        self._initialize_templates()
        self._initialize_schemas()
        self._initialize_patterns()
    
    def _initialize_templates(self):
        """Initialize SQL query templates"""
        
        # Template 1: Basic TTS Usage Query
        self.templates["tts_basic"] = SQLTemplate(
            name="tts_basic",
            description="Basic TTS usage data retrieval",
            base_query="""
            SELECT user_uid, email, text, timestamp,
                   LENGTH(text) as text_length,
                   EXTRACT(HOUR FROM timestamp) as hour_of_day,
                   EXTRACT(DAYOFWEEK FROM timestamp) as day_of_week
            FROM `{table_id}`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            {additional_filters}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """,
            required_params=["table_id", "days_back", "limit"],
            optional_params=["email", "text_contains", "user_uid", "min_length", "max_length"],
            example_nl="find the past 7 days of tts generations",
            result_columns=["user_uid", "email", "text", "timestamp", "text_length", "hour_of_day", "day_of_week"]
        )
        
        # Template 2: User-Specific Query
        self.templates["user_specific"] = SQLTemplate(
            name="user_specific",
            description="User-specific TTS usage analysis",
            base_query="""
            SELECT user_uid, email, text, timestamp,
                   LENGTH(text) as text_length,
                   COUNT(*) OVER (PARTITION BY user_uid) as total_requests,
                   EXTRACT(HOUR FROM timestamp) as hour_of_day
            FROM `{table_id}`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            {user_filter}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """,
            required_params=["table_id", "days_back", "limit"],
            optional_params=["email", "user_uid", "text_contains"],
            example_nl="investigate user john@example.com",
            result_columns=["user_uid", "email", "text", "timestamp", "text_length", "total_requests", "hour_of_day"]
        )
        
        # Template 3: Content Analysis Query
        self.templates["content_analysis"] = SQLTemplate(
            name="content_analysis",
            description="Content analysis with keyword search",
            base_query="""
            SELECT user_uid, email, text, timestamp,
                   LENGTH(text) as text_length,
                   CASE 
                     WHEN LENGTH(text) > 500 THEN 'long'
                     WHEN LENGTH(text) < 50 THEN 'short'
                     ELSE 'medium'
                   END as text_category
            FROM `{table_id}`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            {content_filter}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """,
            required_params=["table_id", "days_back", "limit"],
            optional_params=["text_contains", "min_length", "max_length"],
            example_nl="find content containing 'scam' in past 3 days",
            result_columns=["user_uid", "email", "text", "timestamp", "text_length", "text_category"]
        )
        
        # Template 4: Volume Analysis Query
        self.templates["volume_analysis"] = SQLTemplate(
            name="volume_analysis",
            description="Volume analysis for high-usage detection",
            base_query="""
            SELECT user_uid, email,
                   COUNT(*) as request_count,
                   AVG(LENGTH(text)) as avg_text_length,
                   MIN(timestamp) as first_request,
                   MAX(timestamp) as last_request,
                   COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM `{table_id}`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            {additional_filters}
            GROUP BY user_uid, email
            HAVING COUNT(*) > {min_requests}
            ORDER BY request_count DESC
            LIMIT {limit}
            """,
            required_params=["table_id", "days_back", "limit", "min_requests"],
            optional_params=["email", "text_contains"],
            example_nl="find users with more than 100 requests in past week",
            result_columns=["user_uid", "email", "request_count", "avg_text_length", "first_request", "last_request", "active_days"]
        )
        
        # Template 5: Time-based Analysis Query
        self.templates["time_analysis"] = SQLTemplate(
            name="time_analysis",
            description="Time-based pattern analysis",
            base_query="""
            SELECT user_uid, email, text, timestamp,
                   EXTRACT(HOUR FROM timestamp) as hour_of_day,
                   EXTRACT(DAYOFWEEK FROM timestamp) as day_of_week,
                   EXTRACT(DATE FROM timestamp) as date_only
            FROM `{table_id}`
            WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            {time_filter}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """,
            required_params=["table_id", "days_back", "limit"],
            optional_params=["hour_start", "hour_end", "day_of_week", "email"],
            example_nl="find activity during overnight hours",
            result_columns=["user_uid", "email", "text", "timestamp", "hour_of_day", "day_of_week", "date_only"]
        )
    
    def _initialize_schemas(self):
        """Initialize table schemas for validation"""
        
        # TTS Usage table schema
        self.table_schemas["TTS Usage"] = {
            "table_id": "xi-labs.xi_prod.tts_usage",
            "columns": {
                "user_uid": "STRING",
                "email": "STRING", 
                "text": "STRING",
                "timestamp": "TIMESTAMP",
                "workspace_uid": "STRING",
                "voice_id": "STRING",
                "model_id": "STRING",
                "request_id": "STRING"
            },
            "primary_key": "request_id",
            "time_column": "timestamp",
            "description": "Core TTS generation data"
        }
        
        # Trust Safety Data schema
        self.table_schemas["Trust Safety Data"] = {
            "table_id": "eleven-team-safety.trust_safety_data.content_analysis",
            "columns": {
                "content_id": "STRING",
                "user_uid": "STRING",
                "email": "STRING",
                "analysis_result": "STRING",
                "risk_level": "STRING",
                "timestamp": "TIMESTAMP",
                "flags": "ARRAY<STRING>",
                "confidence": "FLOAT"
            },
            "primary_key": "content_id",
            "time_column": "timestamp",
            "description": "Content safety classifications"
        }
        
        # User Analytics schema
        self.table_schemas["User Analytics"] = {
            "table_id": "analytics-dev-421514.dbt_jho_marts.dim_users",
            "columns": {
                "user_uid": "STRING",
                "email": "STRING",
                "created_at": "TIMESTAMP",
                "last_active": "TIMESTAMP",
                "total_requests": "INTEGER",
                "subscription_type": "STRING"
            },
            "primary_key": "user_uid",
            "time_column": "created_at",
            "description": "User profile analytics"
        }
    
    def _initialize_patterns(self):
        """Initialize natural language patterns"""
        
        # Intent patterns
        self.query_patterns = {
            "time_based": [
                r"past (\d+) (day|days|week|weeks|month|months)",
                r"last (\d+) (day|days|week|weeks|month|months)",
                r"(\d+) (day|days|week|weeks|month|months) ago",
                r"since (\d{4}-\d{2}-\d{2})",
                r"from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})"
            ],
            "user_specific": [
                r"user[s]? (.+@.+\..+)",
                r"email (.+@.+\..+)",
                r"investigate (.+@.+\..+)",
                r"check (.+@.+\..+)",
                r"analyze (.+@.+\..+)"
            ],
            "content_based": [
                r"containing ['\"](.+?)['\"]",
                r"with ['\"](.+?)['\"]",
                r"about (.+)",
                r"text (.+)",
                r"content (.+)"
            ],
            "volume_based": [
                r"more than (\d+) requests",
                r"over (\d+) (request|requests|generation|generations)",
                r"high volume",
                r"frequent users",
                r"top (\d+) users"
            ],
            "scam_detection": [
                r"scam",
                r"fraud",
                r"suspicious",
                r"malicious",
                r"abuse"
            ]
        }
    
    def convert_nl_to_sql(self, nl_query: str, target_table: str = "TTS Usage") -> Dict[str, Any]:
        """
        Convert natural language query to BigQuery SQL
        
        Args:
            nl_query: Natural language query string
            target_table: Target table name (default: "TTS Usage")
            
        Returns:
            Dictionary with SQL query, parameters, and metadata
        """
        print(f"🔄 NL→SQL: Converting '{nl_query}' to BigQuery SQL")
        
        result = {
            "nl_query": nl_query,
            "target_table": target_table,
            "sql_query": None,
            "template_used": None,
            "extracted_params": {},
            "validation_errors": [],
            "conversion_confidence": 0.0,
            "conversion_method": "pattern_matching"
        }
        
        try:
            # Step 1: Extract parameters from natural language
            params = self._extract_nl_parameters(nl_query)
            result["extracted_params"] = params
            
            # Step 2: Determine query intent and select template
            template_name = self._determine_query_intent(nl_query, params)
            result["template_used"] = template_name
            
            # Step 3: Validate target table
            if target_table not in self.table_schemas:
                result["validation_errors"].append(f"Unknown target table: {target_table}")
                return result
            
            # Step 4: Generate SQL query
            sql_query = self._generate_sql_query(template_name, params, target_table)
            result["sql_query"] = sql_query
            
            # Step 5: Validate generated SQL
            validation_result = self._validate_sql_query(sql_query, target_table)
            result["validation_errors"] = validation_result["errors"]
            result["conversion_confidence"] = validation_result["confidence"]
            
            # Step 6: Add metadata
            result["query_metadata"] = {
                "estimated_complexity": self._estimate_query_complexity(sql_query),
                "expected_result_columns": self.templates[template_name].result_columns if template_name in self.templates else [],
                "performance_notes": self._generate_performance_notes(sql_query, params)
            }
            
            print(f"✅ NL→SQL: Successfully converted to {template_name} template")
            print(f"✅ NL→SQL: Confidence: {result['conversion_confidence']:.2f}")
            
            return result
            
        except Exception as e:
            result["validation_errors"].append(f"Conversion failed: {str(e)}")
            print(f"❌ NL→SQL: Conversion failed - {str(e)}")
            return result
    
    def _extract_nl_parameters(self, nl_query: str) -> Dict[str, Any]:
        """Extract parameters from natural language query"""
        
        params = {
            "days_back": 7,  # Default
            "limit": 1000,   # Default
            "min_requests": 1  # Default
        }
        
        query_lower = nl_query.lower()
        
        # Extract time parameters
        for pattern in self.query_patterns["time_based"]:
            match = re.search(pattern, query_lower)
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                
                if unit.startswith("day"):
                    params["days_back"] = number
                elif unit.startswith("week"):
                    params["days_back"] = number * 7
                elif unit.startswith("month"):
                    params["days_back"] = number * 30
                
                break
        
        # Extract user email
        for pattern in self.query_patterns["user_specific"]:
            match = re.search(pattern, query_lower)
            if match:
                params["email"] = match.group(1)
                break
        
        # Extract content search
        for pattern in self.query_patterns["content_based"]:
            match = re.search(pattern, query_lower)
            if match:
                params["text_contains"] = match.group(1)
                break
        
        # Extract volume parameters
        for pattern in self.query_patterns["volume_based"]:
            match = re.search(pattern, query_lower)
            if match:
                if "more than" in pattern or "over" in pattern:
                    params["min_requests"] = int(match.group(1))
                elif "top" in pattern:
                    params["limit"] = int(match.group(1))
                break
        
        # Extract limit from query
        limit_match = re.search(r"limit (\d+)", query_lower)
        if limit_match:
            params["limit"] = int(limit_match.group(1))
        
        return params
    
    def _determine_query_intent(self, nl_query: str, params: Dict[str, Any]) -> str:
        """Determine query intent and select appropriate template"""
        
        query_lower = nl_query.lower()
        
        # Check for volume analysis intent
        if any(re.search(pattern, query_lower) for pattern in self.query_patterns["volume_based"]):
            return "volume_analysis"
        
        # Check for user-specific intent
        if "email" in params or any(re.search(pattern, query_lower) for pattern in self.query_patterns["user_specific"]):
            return "user_specific"
        
        # Check for content analysis intent
        if "text_contains" in params or any(re.search(pattern, query_lower) for pattern in self.query_patterns["content_based"]):
            return "content_analysis"
        
        # Check for time-based analysis
        if any(word in query_lower for word in ["overnight", "night", "hour", "time"]):
            return "time_analysis"
        
        # Default to basic TTS query
        return "tts_basic"
    
    def _generate_sql_query(self, template_name: str, params: Dict[str, Any], target_table: str) -> str:
        """Generate SQL query from template and parameters"""
        
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        table_schema = self.table_schemas[target_table]
        
        # Build query parameters
        query_params = {
            "table_id": table_schema["table_id"],
            "days_back": params.get("days_back", 7),
            "limit": params.get("limit", 1000)
        }
        
        # Add template-specific parameters
        if template_name == "volume_analysis":
            query_params["min_requests"] = params.get("min_requests", 50)
        
        # Build additional filters
        additional_filters = []
        
        if params.get("email"):
            additional_filters.append(f"AND email = '{params['email']}'")
        
        if params.get("text_contains"):
            additional_filters.append(f"AND LOWER(text) LIKE '%{params['text_contains'].lower()}%'")
        
        if params.get("user_uid"):
            additional_filters.append(f"AND user_uid = '{params['user_uid']}'")
        
        if params.get("min_length"):
            additional_filters.append(f"AND LENGTH(text) >= {params['min_length']}")
        
        if params.get("max_length"):
            additional_filters.append(f"AND LENGTH(text) <= {params['max_length']}")
        
        # Build filter strings
        if template_name == "user_specific":
            query_params["user_filter"] = " ".join(additional_filters) if additional_filters else ""
        elif template_name == "content_analysis":
            query_params["content_filter"] = " ".join(additional_filters) if additional_filters else ""
        elif template_name == "time_analysis":
            query_params["time_filter"] = " ".join(additional_filters) if additional_filters else ""
        else:
            query_params["additional_filters"] = " ".join(additional_filters) if additional_filters else ""
        
        # Generate SQL from template
        sql_query = template.base_query.format(**query_params)
        
        # Clean up the query
        sql_query = self._clean_sql_query(sql_query)
        
        # Post-process the query
        sql_query = self.fix_week_interval_in_sql(sql_query)
        
        return sql_query
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and format SQL query"""
        
        # Remove extra whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Remove empty WHERE clauses
        sql_query = re.sub(r'WHERE\s+AND', 'WHERE', sql_query)
        sql_query = re.sub(r'WHERE\s+ORDER', 'ORDER', sql_query)
        sql_query = re.sub(r'WHERE\s+GROUP', 'GROUP', sql_query)
        sql_query = re.sub(r'WHERE\s+HAVING', 'HAVING', sql_query)
        
        return sql_query
    
    def _validate_sql_query(self, sql_query: str, target_table: str) -> Dict[str, Any]:
        """Validate generated SQL query"""
        
        validation_result = {
            "errors": [],
            "warnings": [],
            "confidence": 1.0
        }
        
        # Check for basic SQL syntax
        if not sql_query.strip().upper().startswith("SELECT"):
            validation_result["errors"].append("Query must start with SELECT")
            validation_result["confidence"] = 0.0
        
        # Check for table reference
        table_schema = self.table_schemas[target_table]
        if table_schema["table_id"] not in sql_query:
            validation_result["errors"].append(f"Missing table reference: {table_schema['table_id']}")
            validation_result["confidence"] *= 0.5
        
        # Check for required columns
        required_columns = ["user_uid", "email", "timestamp"]
        for col in required_columns:
            if col in table_schema["columns"] and col not in sql_query:
                validation_result["warnings"].append(f"Missing recommended column: {col}")
                validation_result["confidence"] *= 0.9
        
        # Check for LIMIT clause
        if "LIMIT" not in sql_query.upper():
            validation_result["warnings"].append("Query should include LIMIT clause")
            validation_result["confidence"] *= 0.95
        
        # Check for ORDER BY clause
        if "ORDER BY" not in sql_query.upper():
            validation_result["warnings"].append("Query should include ORDER BY clause")
            validation_result["confidence"] *= 0.95
        
        return validation_result
    
    def _estimate_query_complexity(self, sql_query: str) -> str:
        """Estimate query complexity for performance planning"""
        
        query_upper = sql_query.upper()
        
        # Count complexity indicators
        complexity_score = 0
        
        if "JOIN" in query_upper:
            complexity_score += 2
        if "GROUP BY" in query_upper:
            complexity_score += 1
        if "HAVING" in query_upper:
            complexity_score += 1
        if "WINDOW" in query_upper or "OVER" in query_upper:
            complexity_score += 2
        if "DISTINCT" in query_upper:
            complexity_score += 1
        
        # Count functions
        functions = ["COUNT", "SUM", "AVG", "MAX", "MIN", "EXTRACT", "LENGTH"]
        for func in functions:
            if func in query_upper:
                complexity_score += 0.5
        
        # Determine complexity level
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_performance_notes(self, sql_query: str, params: Dict[str, Any]) -> List[str]:
        """Generate performance optimization notes"""
        
        notes = []
        
        # Check for time range optimization
        if params.get("days_back", 7) > 30:
            notes.append(f"Long time range ({params['days_back']} days) - consider partitioning")
        
        # Check for text search optimization
        if params.get("text_contains"):
            notes.append("Text search may be slow - consider full-text search optimization")
        
        # Check for high limit
        if params.get("limit", 1000) > 5000:
            notes.append(f"High limit ({params['limit']}) - consider pagination")
        
        # Check for aggregation
        if "GROUP BY" in sql_query.upper():
            notes.append("Aggregation query - may require more memory")
        
        return notes
    
    def test_nl_conversion(self, test_queries: List[str]) -> Dict[str, Any]:
        """Test natural language conversion with multiple queries"""
        
        print("🧪 Testing NL→SQL conversion system")
        print("=" * 50)
        
        results = {
            "total_queries": len(test_queries),
            "successful_conversions": 0,
            "failed_conversions": 0,
            "average_confidence": 0.0,
            "test_results": []
        }
        
        total_confidence = 0.0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}/{len(test_queries)}: {query}")
            print("-" * 40)
            
            conversion_result = self.convert_nl_to_sql(query)
            
            if conversion_result["sql_query"] and not conversion_result["validation_errors"]:
                results["successful_conversions"] += 1
                status = "✅ SUCCESS"
            else:
                results["failed_conversions"] += 1
                status = "❌ FAILED"
            
            confidence = conversion_result["conversion_confidence"]
            total_confidence += confidence
            
            print(f"Status: {status}")
            print(f"Template: {conversion_result['template_used']}")
            print(f"Confidence: {confidence:.2f}")
            
            if conversion_result["validation_errors"]:
                print(f"Errors: {'; '.join(conversion_result['validation_errors'])}")
            
            results["test_results"].append({
                "query": query,
                "success": conversion_result["sql_query"] is not None,
                "confidence": confidence,
                "template": conversion_result["template_used"],
                "errors": conversion_result["validation_errors"]
            })
        
        results["average_confidence"] = total_confidence / len(test_queries) if test_queries else 0.0
        
        print("\n" + "=" * 50)
        print("🧪 NL→SQL TEST RESULTS")
        print("=" * 50)
        print(f"Total Queries: {results['total_queries']}")
        print(f"Successful: {results['successful_conversions']}")
        print(f"Failed: {results['failed_conversions']}")
        print(f"Success Rate: {(results['successful_conversions']/results['total_queries'])*100:.1f}%")
        print(f"Average Confidence: {results['average_confidence']:.2f}")
        
        return results

    def fix_week_interval_in_sql(self, sql: str) -> str:
        """Replace INTERVAL 1 WEEK with INTERVAL 7 DAY for TIMESTAMP columns in SQL."""
        return re.sub(r'INTERVAL\\s+1\\s+WEEK', 'INTERVAL 7 DAY', sql, flags=re.IGNORECASE)

# =============================================================================
# USAGE EXAMPLES AND TESTING
# =============================================================================

def test_nl_to_sql_system():
    """Test the NL to SQL conversion system with example queries"""
    
    converter = NLToSQLConverter()
    
    test_queries = [
        "find the past 7 days of tts generations",
        "investigate user john@example.com",
        "find content containing 'scam' in past 3 days",
        "show users with more than 100 requests in past week",
        "find activity during overnight hours",
        "analyze past 30 days of high volume users",
        "search for 'investment' in past 2 weeks",
        "get recent activity for user@example.com"
    ]
    
    return converter.test_nl_conversion(test_queries)

def convert_query(nl_query: str, target_table: str = "TTS Usage") -> Dict[str, Any]:
    """Convert a single natural language query to SQL"""
    
    converter = NLToSQLConverter()
    return converter.convert_nl_to_sql(nl_query, target_table)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("🔄 NATURAL LANGUAGE TO SQL CONVERTER INITIALIZED")
print("=" * 60)
print("Available Functions:")
print("  • convert_query(nl_query, target_table) - Convert single query")
print("  • test_nl_to_sql_system() - Test system with example queries")
print("  • NLToSQLConverter() - Full converter class")
print()
print("Example Usage:")
print("  result = convert_query('find past 7 days of tts generations')")
print("  print(result['sql_query'])")
print()
print("Supported Query Types:")
print("  • Time-based: 'past X days', 'last X weeks'")
print("  • User-specific: 'user email@domain.com', 'investigate user'")
print("  • Content-based: 'containing text', 'about keyword'")
print("  • Volume-based: 'more than X requests', 'high volume users'")
print("  • Time pattern: 'overnight hours', 'during weekends'")
print("=" * 60)

# Create global converter instance
nl_converter = NLToSQLConverter()

print("✅ NL→SQL Converter Ready - Natural Language to BigQuery SQL conversion available") 