# ==============================================================
# CELL 05: Trust Safety Analyzer
# Purpose: OpenAI-based and fallback abuse classifier
# Dependencies: SQL executor, investigation manager, config
# ==============================================================

# @title Cell 5: Main Investigation System with OpenAI Integration
# Core Trust & Safety investigation system with agentic analysis

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from google.cloud import bigquery
import re

# OpenAI integration (if available)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: OpenAI library not available - install with: pip install openai")

# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class TrustSafetyConfig:
    """Configuration for Trust & Safety analysis"""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    max_tokens: int = 1500
    temperature: float = 0.1
    analysis_batch_size: int = 10
    enable_content_analysis: bool = True
    enable_pattern_detection: bool = True
    enable_risk_scoring: bool = True

@dataclass
class AnalysisResult:
    """Result of Trust & Safety analysis"""
    user_id: str
    content: str
    risk_score: float  # 0.0 to 1.0
    risk_level: str    # "low", "medium", "high", "critical"
    flags: List[str]
    reasoning: str
    timestamp: datetime
    confidence: float
    recommendations: List[str]
    analysis_method: str

@dataclass
class InvestigationTask:
    """Task for investigation system"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    depends_on: Optional[str] = None

# =============================================================================
# TRUST & SAFETY ANALYZER
# =============================================================================

class TrustSafetyAnalyzer:
    """Synchronous trust & safety analyzer with OpenAI integration"""

    def __init__(self, config: TrustSafetyConfig):
        self.config = config
        self.openai_client = None
        self.analysis_history = []
        self.token_usage = {"total": 0, "prompt": 0, "completion": 0}
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and config.openai_api_key:
            self.openai_client = OpenAI(api_key=config.openai_api_key)
            print("SUCCESS: OpenAI client initialized")
        else:
            print("WARNING: OpenAI client not available - content analysis limited")

    def analyze_content_batch(self, content_items: List[Dict[str, Any]]) -> List[AnalysisResult]:
        """Analyze a batch of content items for Trust & Safety violations"""
        if not self.openai_client:
            print("WARNING: OpenAI not available - using rule-based analysis")
            return self._fallback_analysis(content_items)

        results = []
        batch_size = self.config.analysis_batch_size
        
        for i in range(0, len(content_items), batch_size):
            batch = content_items[i:i + batch_size]
            batch_results = self._analyze_batch_with_openai(batch)
            results.extend(batch_results)
            
            # Brief pause to respect rate limits
            time.sleep(0.1)
        
        return results

    def _analyze_batch_with_openai(self, batch: List[Dict[str, Any]]) -> List[AnalysisResult]:
        """Analyze batch using OpenAI"""
        try:
            # Prepare content for analysis
            content_text = ""
            for idx, item in enumerate(batch):
                content_text += f"Content {idx + 1}: {item.get('text', '')}\n"
                content_text += f"User: {item.get('user_id', 'unknown')}\n"
                content_text += f"Timestamp: {item.get('timestamp', '')}\n\n"

            # OpenAI prompt for Trust & Safety analysis
            prompt = f"""
            You are a Trust & Safety analyst. Analyze the following content for potential violations:

            {content_text}

            For each content item, provide:
            1. Risk score (0.0 to 1.0)
            2. Risk level (low, medium, high, critical)
            3. Specific flags (scam, hate_speech, self_harm, spam, fraud, etc.)
            4. Brief reasoning
            5. Confidence level (0.0 to 1.0)
            6. Recommendations

            Focus on:
            - Scam attempts and fraudulent content
            - Hate speech and harassment
            - Self-harm content
            - Spam and bulk abuse
            - Economic exploitation
            - Platform manipulation

            Return as JSON array with format:
            [{{"content_id": 1, "risk_score": 0.8, "risk_level": "high", "flags": ["scam", "fraud"], "reasoning": "...", "confidence": 0.9, "recommendations": ["immediate_review", "account_suspension"]}}]
            """

            response = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a Trust & Safety analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            # Track token usage
            if hasattr(response, 'usage'):
                self.token_usage["total"] += response.usage.total_tokens
                self.token_usage["prompt"] += response.usage.prompt_tokens
                self.token_usage["completion"] += response.usage.completion_tokens

            # Parse response
            response_text = response.choices[0].message.content
            analysis_data = json.loads(response_text)

            # Convert to AnalysisResult objects
            results = []
            for idx, item in enumerate(batch):
                if idx < len(analysis_data):
                    analysis = analysis_data[idx]
                    result = AnalysisResult(
                        user_id=item.get('user_id', 'unknown'),
                        content=item.get('text', ''),
                        risk_score=analysis.get('risk_score', 0.0),
                        risk_level=analysis.get('risk_level', 'low'),
                        flags=analysis.get('flags', []),
                        reasoning=analysis.get('reasoning', ''),
                        timestamp=datetime.now(),
                        confidence=analysis.get('confidence', 0.0),
                        recommendations=analysis.get('recommendations', []),
                        analysis_method="OpenAI"
                    )
                    results.append(result)
                    self.analysis_history.append(result)

            return results

        except Exception as e:
            print(f"ERROR: OpenAI analysis failed: {str(e)}")
            return self._fallback_analysis(batch)

    def _fallback_analysis(self, content_items: List[Dict[str, Any]]) -> List[AnalysisResult]:
        """
        FALLBACK ONLY: Basic rule-based pattern matching when OpenAI is unavailable
        WARNING: This is NOT AI analysis - only simple pattern matching
        Results are significantly less accurate than OpenAI-powered analysis
        """
        print("⚠️  WARNING: Using fallback rule-based analysis - OpenAI not available")
        print("⚠️  Results will be limited and less accurate than AI analysis")
        
        results = []
        
        # Basic rule-based patterns (NOT AI)
        risk_patterns = {
            "scam": [r"\bmoney\b.*\bquick\b", r"\bget\s+rich\b", r"\bfree\s+money\b", r"\binvestment\s+opportunity\b"],
            "hate_speech": [r"\bstupid\b.*\bpeople\b", r"\bhate\s+you\b", r"\bkill\s+yourself\b"],
            "spam": [r"\bclick\s+here\b", r"\bvisit\s+now\b", r"\bspecial\s+offer\b"],
            "fraud": [r"\bcredit\s+card\b.*\bfree\b", r"\bsocial\s+security\b", r"\bbank\s+account\b"]
        }
        
        for item in content_items:
            text = item.get('text', '').lower()
            flags = []
            risk_score = 0.0
            
            # Check for patterns
            for flag_type, patterns in risk_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        flags.append(flag_type)
                        risk_score += 0.3
            
            # Cap risk score at 1.0, but reduce confidence for rule-based
            risk_score = min(risk_score, 1.0) * 0.6  # Reduced confidence for rule-based
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            result = AnalysisResult(
                user_id=item.get('user_id', 'unknown'),
                content=text,
                risk_score=risk_score,
                risk_level=risk_level,
                flags=flags,
                analysis_method="FALLBACK: Rule-based pattern matching (NOT AI)",
                timestamp=datetime.now(),
                recommendations=["MANUAL REVIEW REQUIRED - Rule-based analysis only"] if risk_score > 0.3 else []
            )
            
            results.append(result)
            self.analysis_history.append(result)
        
        return results

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.analysis_history:
            return {"total_analyzed": 0}
        
        total = len(self.analysis_history)
        risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for result in self.analysis_history:
            risk_levels[result.risk_level] += 1
        
        return {
            "total_analyzed": total,
            "risk_distribution": risk_levels,
            "token_usage": self.token_usage,
            "average_risk_score": sum(r.risk_score for r in self.analysis_history) / total,
            "last_analysis": self.analysis_history[-1].timestamp.isoformat()
        }

# =============================================================================
# MAIN INVESTIGATION SYSTEM
# =============================================================================

class MainInvestigationSystem:
    """Main investigation system coordinating all components"""
    
    def __init__(self):
        self.config = TrustSafetyConfig()
        self.analyzer = None
        self.bq_client = None
        self.active_tasks = []
        self.completed_tasks = []
        
        # Initialize with OpenAI API key from environment
        if 'OPENAI_API_KEY' in os.environ:
            self.config.openai_api_key = os.environ['OPENAI_API_KEY']
            self.analyzer = TrustSafetyAnalyzer(self.config)
            print("✅ AI analyzer initialized with OpenAI API key")
        else:
            print("❌ WARNING: No OpenAI API key found - content analysis will be limited")
            print("   Set OPENAI_API_KEY environment variable or run Cell 1 first")
        
        # Initialize BigQuery client from global scope
        self._initialize_bigquery_client()
        
        # BigQuery client will be dynamically resolved when needed
        self._bq_client = None
    
    def _initialize_bigquery_client(self):
        """Initialize BigQuery client from global scope"""
        try:
            import __main__
            
            # Try to get BigQuery client from Cell 2
            bq_client = getattr(__main__, 'bq_client', None)
            if bq_client:
                self._bq_client = bq_client
                print("✅ BigQuery client initialized from Cell 2")
                return
            
            # Check if VERIFIED_TABLES exists and has accessible TTS Usage table
            VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
            if VERIFIED_TABLES and 'TTS Usage' in VERIFIED_TABLES:
                tts_table = VERIFIED_TABLES['TTS Usage']
                if tts_table.get('accessible') and tts_table.get('client'):
                    self._bq_client = tts_table['client']
                    print("✅ BigQuery client initialized from verified tables")
                    return
            
            print("❌ WARNING: BigQuery client not found - run Cell 2 first")
            print("   main_system.bq_client will be None until Cell 2 is run")
        
        except Exception as e:
            print(f"❌ WARNING: Error initializing BigQuery client: {e}")
            print("   Run Cell 2 first to initialize BigQuery client")
    
    @property
    def bq_client(self):
        """Dynamic property that resolves BigQuery client from global scope"""
        # If explicitly set, use that value
        if self._bq_client is not None:
            return self._bq_client
            
        # Otherwise, dynamically resolve from global scope
        try:
            import __main__
            sql_executor = getattr(__main__, 'sql_executor', None)
            if sql_executor:
                return sql_executor
            
            # Fallback: try to get BigQuery client directly from Cell 2
            bq_client = getattr(__main__, 'bq_client', None)
            if bq_client:
                return bq_client
            
            return None
        except:
            return None
    
    @bq_client.setter
    def bq_client(self, value):
        """Setter for bq_client (stores in private attribute)"""
        self._bq_client = value

    def run_comprehensive_investigation(self, investigation_type: str, **params) -> Dict[str, Any]:
        """Run a comprehensive investigation combining SQL queries and content analysis"""

        # Create investigation if none exists
        if investigation_manager.current_investigation is None:
            investigation_manager.create_investigation(
                title=f"Comprehensive {investigation_type} Investigation",
                description=f"Automated investigation for {investigation_type} patterns",
                risk_level="medium"
            )

        print(f"Starting comprehensive {investigation_type} investigation...")

        # Step 1: Execute SQL query to get data
        if investigation_type == "suspicious_content":
            df = sql_executor.execute_investigation_query("suspicious_content", **params)
        elif investigation_type == "bulk_usage":
            df = sql_executor.execute_investigation_query("bulk_tts_usage", **params)
        elif investigation_type == "payment_fraud":
            df = sql_executor.execute_investigation_query("payment_patterns", **params)
        else:
            print(f"ERROR: Unknown investigation type: {investigation_type}")
            return {"error": "Unknown investigation type"}

        if df.empty:
            print("No data found for investigation")
            return {"data_found": False}

        print(f"SUCCESS: Retrieved {len(df)} records for analysis")

        # Step 2: Analyze content with OpenAI (if available and relevant)
        analysis_results = []
        if self.analyzer and 'text' in df.columns:
            print("Analyzing content with Trust & Safety AI...")
            
            # Prepare content for analysis
            content_items = []
            for _, row in df.iterrows():
                content_items.append({
                    'text': row[text_col],
                    'user_id': row[user_id_col],
                    'timestamp': row[timestamp_col]
                })
            
            # Analyze in batches
            analysis_results = self.analyzer.analyze_content_batch(content_items)
            
            # Log high-risk findings
            high_risk_count = sum(1 for r in analysis_results if r.risk_level in ['high', 'critical'])
            if high_risk_count > 0:
                investigation_manager.add_finding(
                    f"Found {high_risk_count} high-risk content items",
                    f"Analysis identified {high_risk_count} items requiring immediate attention"
                )
        
        # Step 3: Generate investigation summary
        summary = {
            "investigation_type": investigation_type,
            "data_records": len(df),
            "analysis_completed": len(analysis_results),
            "high_risk_items": sum(1 for r in analysis_results if r.risk_level in ['high', 'critical']),
            "recommendations": self._generate_recommendations(df, analysis_results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 4: Update investigation status
        investigation_manager.add_action(
            f"Completed {investigation_type} investigation",
            f"Analyzed {len(df)} records, found {summary['high_risk_items']} high-risk items"
        )

        print(f"SUCCESS: Investigation completed")
        print(f"  Records analyzed: {len(df)}")
        print(f"  High-risk items: {summary['high_risk_items']}")

        return {
            "success": True,
            "summary": summary,
            "data": df,
            "analysis_results": analysis_results
        }

    def _generate_recommendations(self, data_df: pd.DataFrame, analysis_results: List[AnalysisResult]) -> List[str]:
        """Generate recommendations based on investigation results"""
        recommendations = []

        # Data-based recommendations
        if len(data_df) > 100:
            recommendations.append("Large dataset detected - consider focused analysis")

        # Analysis-based recommendations
        if analysis_results:
            critical_count = sum(1 for r in analysis_results if r.risk_level == 'critical')
            high_count = sum(1 for r in analysis_results if r.risk_level == 'high')
            
            if critical_count > 0:
                recommendations.append(f"URGENT: {critical_count} critical violations require immediate action")
            
            if high_count > 5:
                recommendations.append(f"High priority: {high_count} high-risk items need review")
            
            # Common flag patterns
            all_flags = [flag for result in analysis_results for flag in result.flags]
            if all_flags:
                flag_counts = {}
                for flag in all_flags:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1
                
                top_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                recommendations.append(f"Primary concerns: {', '.join(f[0] for f in top_flags)}")

        return recommendations

    def show_system_status(self):
        """Display system status dashboard"""
        print("\nMain Investigation System Status")
        print("=" * 60)

        # OpenAI status
        openai_status = "AVAILABLE" if self.analyzer else "NOT AVAILABLE"
        print(f"OpenAI Integration: {openai_status}")

        # Current investigation
        if 'investigation_manager' in globals() and investigation_manager.current_investigation is not None:
            inv = investigation_manager.current_investigation
            print(f"Current Investigation: {inv.title}")
            print(f"  Status: {inv.status}")
            print(f"  Risk Level: {inv.risk_level}")
            print(f"  Findings: {len(inv.findings)}")
        else:
            print("Current Investigation: None")

        # Analysis stats
        if self.analyzer:
            stats = self.analyzer.get_analysis_stats()
            print(f"Analysis Stats:")
            print(f"  Total Analyzed: {stats['total_analyzed']}")
            if stats['total_analyzed'] > 0:
                print(f"  Risk Distribution: {stats['risk_distribution']}")
                print(f"  Token Usage: {stats['token_usage']['total']}")

        # Available tables - use fallback for VERIFIED_TABLES
        if 'VERIFIED_TABLES' in globals():
            accessible_tables = sum(1 for t in VERIFIED_TABLES.values() if t.get('accessible', False))
            print(f"Accessible Tables: {accessible_tables}/{len(VERIFIED_TABLES)}")
        else:
            print("Accessible Tables: Configuration pending (run Cell 2)")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def quick_investigation(investigation_type: str, **params) -> Dict[str, Any]:
    """Quick function to run investigations"""
    return main_system.run_comprehensive_investigation(investigation_type, **params)

def analyze_user_content(user_id: str, days_back: int = 7) -> List[AnalysisResult]:
    """Analyze all content from a specific user"""
    if not main_system.analyzer:
        print("ERROR: Content analyzer not available")
        return []
    
    # Get user's TTS content using schema-aware column names
    table_name = "TTS Usage"
    schema = get_table_schema(table_name)
    
    if not schema:
        raise RuntimeError(f"Schema not found for {table_name}")
    
    verified_table = VERIFIED_TABLES.get(table_name)
    if not verified_table or not verified_table["accessible"]:
        raise RuntimeError(f"Table {table_name} not verified or accessible")
    
    # Get correct column names from schema
    user_id_col = get_column_name(table_name, "user_id")
    text_col = get_column_name(table_name, "text")
    timestamp_col = get_column_name(table_name, "timestamp")
    
    # Build analysis query
    query = f"""
    SELECT {user_id_col}, {text_col}, {timestamp_col}
    FROM `{verified_table['table_id']}`
    WHERE {user_id_col} = '{user_id}'
    AND DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    ORDER BY {timestamp_col} DESC
    LIMIT 1000
    """
    
    df = sql_executor.execute_custom_query(query, f"Content analysis for user {user_id}")
    
    if df.empty:
        print(f"No content found for user {user_id}")
        return []
    
    # Process results using schema-aware column names
    content_items = []
    for _, row in df.iterrows():
        content_items.append({
            'text': row[text_col],
            'user_id': row[user_id_col],
            'timestamp': row[timestamp_col]
        })
    
    # Analyze content
    results = main_system.analyzer.analyze_content_batch(content_items)
    
    print(f"SUCCESS: Analyzed {len(results)} content items for user {user_id}")
    return results

def show_system_status():
    """Display system status dashboard"""
    print("\nMain Investigation System Status")
    print("=" * 60)
    
    # OpenAI status
    openai_status = "AVAILABLE" if main_system.analyzer else "NOT AVAILABLE"
    print(f"OpenAI Integration: {openai_status}")
    
    # Current investigation
    if investigation_manager.current_investigation is not None:
        inv = investigation_manager.current_investigation
        print(f"Current Investigation: {inv.title}")
        print(f"  Status: {inv.status}")
        print(f"  Risk Level: {inv.risk_level}")
        print(f"  Findings: {len(inv.findings)}")
    else:
        print("Current Investigation: None")
    
    # Analysis stats
    if main_system.analyzer:
        stats = main_system.analyzer.get_analysis_stats()
        print(f"Analysis Stats:")
        print(f"  Total Analyzed: {stats['total_analyzed']}")
        if stats['total_analyzed'] > 0:
            print(f"  Risk Distribution: {stats['risk_distribution']}")
            print(f"  Token Usage: {stats['token_usage']['total']}")
    
    # Available tables
    accessible_tables = sum(1 for t in VERIFIED_TABLES.values() if t['accessible'])
    print(f"Accessible Tables: {accessible_tables}/{len(VERIFIED_TABLES)}")

# =============================================================================
# INITIALIZE MAIN SYSTEM
# =============================================================================

print("🚀 INITIALIZING MAIN INVESTIGATION SYSTEM")
print("=" * 50)

# Create global main investigation system
main_system = MainInvestigationSystem()

# Verify system integrity
print("\n🔍 SYSTEM INTEGRITY CHECK")
print("-" * 30)

# Check analyzer
if main_system.analyzer:
    print("✅ AI analyzer: Available")
    if hasattr(main_system.analyzer, 'openai_client') and main_system.analyzer.openai_client:
        print("✅ OpenAI client: Available")
    else:
        print("❌ OpenAI client: Missing")
else:
    print("❌ AI analyzer: Not available")

# Check BigQuery client
if main_system.bq_client:
    print("✅ BigQuery client: Available")
else:
    print("❌ BigQuery client: Not available")

# Check if system is ready for investigation
system_ready = (
    main_system.analyzer is not None and
    main_system.bq_client is not None and
    hasattr(main_system.analyzer, 'openai_client') and
    main_system.analyzer.openai_client is not None
)

print(f"\n🎯 SYSTEM STATUS: {'✅ READY' if system_ready else '❌ NOT READY'}")

if not system_ready:
    print("\n⚠️  CRITICAL COMPONENTS MISSING:")
    if not main_system.analyzer:
        print("   • AI analyzer missing - check OpenAI API key")
    elif not hasattr(main_system.analyzer, 'openai_client') or not main_system.analyzer.openai_client:
        print("   • OpenAI client missing - check API key configuration")
    if not main_system.bq_client:
        print("   • BigQuery client missing - run Cell 2 first")
    print("\n💡 REQUIRED ACTIONS:")
    print("   1. Ensure Cell 1 is run (sets OpenAI API key)")
    print("   2. Ensure Cell 2 is run (initializes BigQuery client)")
    print("   3. Re-run Cell 5 to reinitialize main_system")
else:
    print("✅ All critical components available")
    print("✅ System ready for REAL data investigation")

print("\n📋 AVAILABLE FUNCTIONS:")
print("  - quick_investigation(investigation_type, **params)")
print("  - analyze_user_content(user_id, days_back=7)")
print("  - show_system_status()")
print("  - main_system.run_comprehensive_investigation(type, **params)")
print("  - check_runtime_integrity() - Verify all components")

print(f"\n🎯 CELL 5 COMPLETE - Main Investigation System {'✅ READY' if system_ready else '❌ NEEDS SETUP'}")

# =============================================================================
# NOTEBOOK INTEGRATION - SHARED SYSTEM AWARENESS
# =============================================================================

print("\n🔗 NOTEBOOK INTEGRATION CHECK...")

# Check if we're in a unified notebook system
if 'NOTEBOOK_STATE' in globals():
    print("✅ Unified notebook system detected")
    
    # Integrate with shared system
    if 'setup_main_system' in globals():
        print("✅ Integrating with shared notebook system...")
        try:
            setup_main_system()
            print("✅ Main system integrated with shared system")
            
            # Update global reference to use shared system
            if 'run_investigation_query' in globals():
                print("✅ Using shared investigation query system")
            
        except Exception as e:
            print(f"⚠️  Shared system integration failed: {e}")
            print("   Continuing with standalone main system")
        
    else:
        print("⚠️  Shared system functions not available")
        print("   For full integration, run: exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")
        
else:
    print("⚠️  Running in standalone mode")
    print("   For full notebook integration, run:")
    print("   exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")

print("\n🎯 CELL 5 READY - Main Investigation System available")
print("   • Standalone: main_system.*")
print("   • Integrated: Uses shared investigation query system") 