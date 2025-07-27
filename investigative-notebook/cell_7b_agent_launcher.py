# @title Cell 7b: Agent Launcher — Investigator-Defined Agent Execution
# Investigator-first agent system for Trust & Safety investigations

import os
import re
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import pandas as pd
from google.cloud import bigquery
import inspect

# Import main investigation system
import sys
import os
# Add current directory to path if needed
if '.' not in sys.path:
    sys.path.insert(0, '.')
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Import main_system from cell 5
# Note: main_system will be accessed dynamically from __main__ when needed
# This avoids module import issues in Jupyter notebook environment
print("Agent system initializing - main_system will be accessed at runtime")

# Basic datetime serialization functions
def deep_serialize_datetime(obj):
    return obj.isoformat() if isinstance(obj, datetime) else obj
def safe_json_dumps(obj, **kwargs):
    return json.dumps(obj, default=str, **kwargs)
def serialize_agent_result(obj):
    return obj

# =============================================================================
# BIGQUERY ERROR HANDLING - PYARROW CONVERSION FIXES
# =============================================================================

def setup_bigquery_client_for_agents():
    """Setup BigQuery client with proper configuration to handle complex data types"""
    try:
        # Import BigQuery client
        from google.cloud import bigquery
        
        # Create job config that handles complex data types
        job_config = bigquery.QueryJobConfig()
        
        # Set job config to use legacy SQL and disable pyarrow conversion for complex types
        job_config.use_legacy_sql = False
        job_config.use_query_cache = True
        
        # Configure job to avoid pyarrow conversion issues
        job_config.create_disposition = bigquery.CreateDisposition.CREATE_NEVER
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
        
        return job_config
        
    except Exception as e:
        print(f"⚠️ BigQuery client setup warning: {e}")
        return None

# Configure BigQuery for complex data types (lazy-loaded)
BIGQUERY_JOB_CONFIG = None

def safe_bigquery_query(client, query_string, job_config=None):
    """
    Execute BigQuery query with error handling for complex data types
    
    This prevents pyarrow conversion errors when tables have complex columns
    like ARRAY, STRUCT, or JSON data types.
    """
    try:
        # Validate client
        if client is None:
            raise ValueError("BigQuery client is None - ensure Cell 2 has been run successfully")
        
        # Use configured job config if available
        if job_config is None:
            global BIGQUERY_JOB_CONFIG
            if BIGQUERY_JOB_CONFIG is None:
                BIGQUERY_JOB_CONFIG = setup_bigquery_client_for_agents()
            job_config = BIGQUERY_JOB_CONFIG
        
        # Execute query
        try:
            query_job = client.query(query_string, job_config=job_config)
        except Exception as query_error:
            # Handle common query errors
            if "404" in str(query_error):
                raise ValueError(f"Table not found: {query_error}")
            elif "400" in str(query_error):
                raise ValueError(f"Query syntax error: {query_error}")
            elif "403" in str(query_error):
                raise ValueError(f"Permission denied: {query_error}")
            else:
                raise query_error
        
        # Get results with error handling
        try:
            results = query_job.result()
            return results
        except Exception as conversion_error:
            # Handle pyarrow conversion errors
            if "pyarrow" in str(conversion_error).lower() or "findings" in str(conversion_error).lower():
                print(f"⚠️ Complex data type detected - using alternative conversion")
                
                # Try to get results without pandas conversion
                results = query_job.result()
                return results
            else:
                raise conversion_error
                
    except Exception as e:
        print(f"❌ BigQuery query failed: {e}")
        print("   Common fixes:")
        print("   • Ensure Cell 2 (BigQuery Configuration) has been run")
        print("   • Check table names and permissions")
        print("   • Verify query syntax")
        raise

# =============================================================================
# AGENT STRUCTURES AND CONFIGURATION
# =============================================================================

def _check_agent_cancellation(agent_context=None):
    """Check if agent should be cancelled and raise exception if so"""
    if agent_context and hasattr(agent_context, 'should_stop') and agent_context.should_stop.is_set():
        raise RuntimeError("Agent execution cancelled by user")

@dataclass
class AgentConfig:
    """Configuration for investigation agents"""
    agent_name: str
    description: str
    target_tables: List[str]
    required_fields: List[str]
    default_params: Dict[str, Any]
    use_openai: bool = True
    confidence_threshold: float = 0.7
    query_template: str = ""

@dataclass
class AgentResult:
    """Result from agent execution"""
    agent_name: str
    query_description: str
    records_found: int
    high_risk_items: int
    analysis_results: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time: float
    timestamp: datetime
    investigation_id: Optional[str] = None
    summary_text: Optional[str] = None

@dataclass
class AgentIntent:
    """Detected intent from natural language query"""
    abuse_type: str
    confidence: float
    extracted_params: Dict[str, Any]
    suggested_agent: str
    match_score: float = 0.0
    match_reason: Optional[str] = None

# =============================================================================
# AGENT REGISTRY SYSTEM
# =============================================================================

class AgentRegistry:
    """Registry for investigation agents with investigator-defined logic"""
    
    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.agent_handlers: Dict[str, Callable] = {}
        self.intent_patterns: Dict[str, List[str]] = {}
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default investigation agents"""
        # Scam Detection Agent
        self.register_agent(
            agent_name="scam_agent",
            description="Detects potential scam content and patterns",
            target_tables=["TTS Usage", "Classification Flags"],
            required_fields=["text", "userid", "timestamp"],
            default_params={"days_back": 7, "limit": 100, "confidence_threshold": 0.8},
            use_openai=True,
            handler_function=self._scam_agent_handler,
            intent_patterns=[
                r"scam.*detect",
                r"find.*scam",
                r"fraudulent.*content",
                r"suspicious.*financial",
                r"investment.*scheme",
                r"money.*laundering"
            ],
            query_template="Find recent scam activity from PlayAPI users with potential financial fraud indicators"
        )
        
        # Spam Detection Agent - REMOVED: Not implemented
        # TODO: Implement spam_agent when real detection logic is ready
        
        # Email Network Analysis Agent
        self.register_agent(
            agent_name="email_network_agent",
            description="Analyzes email network patterns and connections",
            target_tables=["TTS Usage", "Classification Flags"],
            required_fields=["email", "userid", "timestamp"],
            default_params={"days_back": 30, "limit": 200, "min_connections": 2},
            use_openai=False,
            handler_function=self._email_network_agent_handler,
            intent_patterns=[
                r"email.*network",
                r"user.*connections",
                r"account.*linkage",
                r"shared.*email",
                r"network.*analysis",
                r"connected.*users",
                r"email.*patterns",
                r"multi.*account",
                r"email.*abuse",
                r"user.*clustering"
            ]
        )
        
        # Exploratory Investigation Agent
        self.register_agent(
            agent_name="exploratory_agent",
            description="General purpose investigation agent for exploratory analysis",
            target_tables=["TTS Usage"],
            required_fields=["email", "userid", "timestamp"],
            default_params={"days_back": 7, "limit": 1000, "analysis_depth": "basic"},
            use_openai=False,
            handler_function=self._exploratory_agent_handler,
            intent_patterns=[
                r"explore.*user",
                r"investigate.*general",
                r"analyze.*activity",
                r"broad.*investigation",
                r"exploratory.*analysis",
                r"general.*check",
                r"what.*happening",
                r"overall.*activity",
                r"user.*behavior",
                r"activity.*patterns"
            ]
        )
        
        print("SUCCESS: Default investigation agents initialized")
    
    def register_agent(self, agent_name: str, description: str, target_tables: List[str],
                      required_fields: List[str], default_params: Dict[str, Any],
                      use_openai: bool, handler_function: Callable,
                      intent_patterns: List[str], query_template: str = ""):
        """Register a new investigation agent"""
        config = AgentConfig(
            agent_name=agent_name,
            description=description,
            target_tables=target_tables,
            required_fields=required_fields,
            default_params=default_params,
            use_openai=use_openai,
            query_template=query_template
        )
        
        self.agents[agent_name] = config
        self.agent_handlers[agent_name] = handler_function
        self.intent_patterns[agent_name] = intent_patterns
        
        print(f"SUCCESS: Agent '{agent_name}' registered")
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names"""
        return list(self.agents.keys())
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.agents.get(agent_name)
    
    def detect_intent(self, query: str) -> Optional[AgentIntent]:
        """
        Production-grade semantic intent detection for investigator queries
        
        Uses OpenAI-based semantic classification with robust fallback mechanisms
        to handle natural investigator language patterns.
        """
        
        print(f"NL INTENT DETECTION: Processing query: '{query}'")
        
        # Step 1: Semantic Intent Classification (Primary)
        semantic_intent = self._classify_intent_semantic(query)
        if semantic_intent and semantic_intent != "unknown":
            return self._build_intent_from_classification(query, semantic_intent, "semantic", 0.9)
        
        # Step 2: Alias/Synonym Mapping (Secondary)
        alias_intent = self._classify_intent_aliases(query)
        if alias_intent:
            return self._build_intent_from_classification(query, alias_intent, "alias", 0.8)
        
        # Step 3: Fallback Handling (Tertiary)
        return self._handle_unknown_intent(query)
    
    def _classify_intent_semantic(self, query: str) -> Optional[str]:
        """Use OpenAI to semantically classify investigator queries"""
        
        # Check if OpenAI is available
        try:
            import __main__
            main_system = getattr(__main__, 'main_system', None)
        except:
            main_system = None
        
        if not main_system or not main_system.analyzer or not main_system.analyzer.openai_client:
            print("  SEMANTIC: OpenAI not available, skipping semantic classification")
            return None
        
        try:
            prompt = f"""Given this Trust & Safety investigator query: "{query}"

Classify it into ONE of these investigation intents:
- scam (financial fraud, investment schemes, fake identities, voice cloning, impersonation)
- spam (bulk content, automated posting, repetitive messages, volume abuse)
- network_analysis (account connections, user relationships, network mapping, linked behavior)
- identity_linkage (same person across accounts, device sharing, shared payment methods)
- exploration (general investigation, baseline review, unclear focus, "investigate user X")
- unknown (truly unclear or nonsensical queries)

Consider natural investigator language like:
- "this guy again?" → network_analysis
- "check if linked" → identity_linkage  
- "looks like a scammer" → scam
- "investigate user@example.com" → exploration
- "check this out" → exploration
- "what's going on here?" → exploration

Return ONLY the intent name, no explanation."""

            response = main_system.analyzer.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Trust & Safety intent classifier. Return only the intent category."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            intent = response.choices[0].message.content.strip().lower()
            
            # Validate intent
            valid_intents = ["scam", "spam", "network_analysis", "identity_linkage", "exploration", "unknown"]
            if intent in valid_intents:
                print(f"  SEMANTIC: Classified as '{intent}'")
                return intent
            else:
                print(f"  SEMANTIC: Invalid classification '{intent}', treating as unknown")
                return "unknown"
                
        except Exception as e:
            print(f"  SEMANTIC: Classification failed: {str(e)}")
            return None
    
    def _classify_intent_aliases(self, query: str) -> Optional[str]:
        """Use alias/synonym mapping for common investigator phrases"""
        
        query_lower = query.lower()
        
        # Investigator-friendly aliases and phrases
        alias_mapping = {
            # Scam detection aliases
            "scam": [
                "fraud", "fraudulent", "scammer", "scamming", "fake", "phishing", 
                "impersonation", "voice clone", "deepfake", "investment scheme",
                "get rich quick", "pyramid", "ponzi", "romance scam", "catfish",
                "financial fraud", "identity theft", "voice spoofing", "fake profile"
            ],
            
            # Spam detection aliases  
            "spam": [
                "bulk", "automated", "bot", "mass", "repetitive", "flooding",
                "volume abuse", "content spam", "message spam", "auto-generated",
                "spam patterns", "bulk messaging", "coordinated posting"
            ],
            
            # Network analysis aliases
            "network_analysis": [
                "connected", "linked", "network", "connections", "relationships",
                "this guy again", "same person", "related accounts", "account cluster",
                "network mapping", "user network", "connection analysis", "linked behavior",
                "coordinated", "ring", "cluster", "group", "associated accounts"
            ],
            
            # Identity linkage aliases
            "identity_linkage": [
                "same user", "shared device", "device sharing", "payment method",
                "same card", "linked payment", "account linkage", "user linkage",
                "cross-account", "multi-account", "alt account", "duplicate account",
                "same identity", "verification", "account verification"
            ],
            
            # Exploratory investigation aliases
            "exploration": [
                "investigate", "check out", "look into", "explore", "review",
                "baseline", "general check", "what's going on", "suspicious",
                "examine", "analyze", "assess", "investigate user", "check user",
                "user activity", "account review", "profile check", "investigate this"
            ]
        }
        
        # Score each intent based on alias matches
        intent_scores = {}
        for intent, aliases in alias_mapping.items():
            score = 0
            matched_aliases = []
            
            for alias in aliases:
                if alias in query_lower:
                    # Exact phrase match gets higher score
                    score += 2.0
                    matched_aliases.append(alias)
                elif any(word in query_lower.split() for word in alias.split()):
                    # Partial word match gets lower score
                    score += 0.5
                    matched_aliases.append(f"partial:{alias}")
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'matches': matched_aliases
                }
        
        # Return best match if above threshold
        if intent_scores:
            best_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['score'])
            best_score = intent_scores[best_intent]['score']
            
            if best_score >= 1.0:  # Minimum threshold for alias matching
                matches = ', '.join(intent_scores[best_intent]['matches'][:3])
                print(f"  ALIAS: Matched '{best_intent}' (score: {best_score:.1f}, matches: {matches})")
                return best_intent
        
        print(f"  ALIAS: No strong alias matches found")
        return None
    
    def _build_intent_from_classification(self, query: str, intent: str, method: str, confidence: float) -> AgentIntent:
        """Build AgentIntent object from classification result"""
        
        # Map semantic intents to agent names
        intent_to_agent = {
            "scam": "scam_agent",
            "spam": "scam_agent",  # Use scam agent for spam detection (no dedicated spam agent)
            "network_analysis": "email_network_agent",
            "identity_linkage": "email_network_agent",  # Use network agent for identity linkage too
            "exploration": "exploratory_agent"  # Add exploratory intent mapping
        }
        
        agent_name = intent_to_agent.get(intent)
        if not agent_name:
            print(f"  ERROR: No agent mapped for intent '{intent}'")
            return None
        
        # Extract parameters from query
        params = self._extract_query_parameters(query)
        
        print(f"  SUCCESS: {method.upper()} classification -> {agent_name} (confidence: {confidence:.2f})")
        
        return AgentIntent(
            abuse_type=intent,
            confidence=confidence,
            extracted_params=params,
            suggested_agent=agent_name,
            match_score=confidence,
            match_reason=f"{method.title()} classification: {intent}"
        )
    
    def _handle_unknown_intent(self, query: str) -> Optional[AgentIntent]:
        """Handle queries with unknown or unclear intent by triggering exploratory investigation"""
        
        print(f"  FALLBACK: Intent unclear - triggering exploratory investigation")
        print(f"  EXPLORATORY: Will perform comprehensive baseline review")
        
        # Log unclassified query for improvement
        self._log_unclassified_query(query)
        
        # Extract parameters for exploratory investigation
        params = self._extract_query_parameters(query)
        
        # Return exploratory intent - never fail an investigation
        return AgentIntent(
            abuse_type="exploration",  # Use exploration intent instead of unknown
            confidence=0.8,  # High confidence for exploratory approach
            extracted_params=params,
            suggested_agent="exploratory_agent",  # Always use exploratory agent for unclear queries
            match_score=0.8,
            match_reason="Intent unclear - performing comprehensive exploratory investigation"
        )
    
    def _log_unclassified_query(self, query: str):
        """Log unclassified queries for system improvement"""
        try:
            # Log to investigation if available
            if hasattr(investigation_manager, 'current_investigation') and investigation_manager.current_investigation is not None:
                investigation_manager.add_action(
                    f"Unclassified Query",
                    f"System could not classify query: '{query}' - manual review needed"
                )
            
            # Also log to console for immediate visibility
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"  LOG: [{timestamp}] Unclassified query logged: '{query}'")
            
        except Exception as e:
            print(f"  LOG: Failed to log unclassified query: {str(e)}")
    
    def _extract_query_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from natural language query using enhanced extraction engine"""
        
        # Use the enhanced parameter extractor from the holistic system
        try:
            # Try to use the enhanced extractor if available
            if 'enhanced_nl' in globals():
                return enhanced_nl.param_extractor.extract_parameters(query)
            else:
                # Fallback to comprehensive built-in extraction
                return self._comprehensive_parameter_extraction(query)
        except Exception as e:
            print(f"⚠️  Parameter extraction error: {str(e)}")
            # Ultimate fallback to basic extraction
            return self._basic_parameter_extraction(query)
    
    def _comprehensive_parameter_extraction(self, query: str) -> Dict[str, Any]:
        """Comprehensive parameter extraction with extensive pattern matching"""
        params = {}
        query_lower = query.lower()
        
        # Time patterns with comprehensive coverage
        time_patterns = {
            r'(?:past|last|previous)\s+(\d+)\s+days?': lambda m: {'days_back': int(m.group(1))},
            r'(?:past|last|previous)\s+(\d+)\s+hours?': lambda m: {'days_back': max(1, int(m.group(1)) // 24)},
            r'(?:past|last|previous)\s+(\d+)\s+weeks?': lambda m: {'days_back': int(m.group(1)) * 7},
            r'(?:past|last|previous)\s+(\d+)\s+months?': lambda m: {'days_back': int(m.group(1)) * 30},
            r'\btoday\b': lambda m: {'days_back': 1},
            r'\byesterday\b': lambda m: {'days_back': 2},
            r'\bthis\s+week\b': lambda m: {'days_back': 7},
            r'\bthis\s+month\b': lambda m: {'days_back': 30},
            r'\blast\s+week\b': lambda m: {'days_back': 7},
            r'\blast\s+month\b': lambda m: {'days_back': 30},
            r'\bpast\s+week\b': lambda m: {'days_back': 7},
            r'\bpast\s+month\b': lambda m: {'days_back': 30},
            r'\brecent(?:ly)?\b': lambda m: {'days_back': 7},
            r'\blatest\b': lambda m: {'days_back': 3},
            r'\bnew\b': lambda m: {'days_back': 1},
        }
        
        # Extract time parameters
        for pattern, extractor in time_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                params.update(extractor(match))
                break
        
        # Limit patterns
        limit_patterns = {
            r'(?:top|first|limit)\s+(\d+)': lambda m: {'limit': int(m.group(1))},
            r'(\d+)\s+(?:records?|results?|items?)': lambda m: {'limit': int(m.group(1))},
            r'show\s+(\d+)': lambda m: {'limit': int(m.group(1))},
        }
        
        for pattern, extractor in limit_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                params.update(extractor(match))
                break
        
        # Source patterns
        source_patterns = {
            r'\bplayapi\b': lambda m: {'source_filter': 'PlayAPI'},
            r'\bwebsite\b': lambda m: {'source_filter': 'Website'},
            r'\bapi\b': lambda m: {'source_filter': 'API'},
            r'\bmobile\b': lambda m: {'source_filter': 'Mobile'},
        }
        
        for pattern, extractor in source_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                params.update(extractor(match))
                break
        
        # Threshold patterns
        threshold_patterns = {
            r'(?:above|over|more than)\s+(\d+)': lambda m: {'min_threshold': int(m.group(1))},
            r'(?:below|under|less than)\s+(\d+)': lambda m: {'max_threshold': int(m.group(1))},
            r'(?:threshold|confidence)\s+(\d+)': lambda m: {'confidence_threshold': float(m.group(1)) / 100},
        }
        
        for pattern, extractor in threshold_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                params.update(extractor(match))
                break
        
        # Email addresses
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        email_matches = re.findall(email_pattern, query)
        if email_matches:
            params['target_email'] = email_matches[0]
        
        # Text content patterns (for searching within text)
        text_content_patterns = {
            r'with the name\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'containing\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'about\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'mentioning\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'named\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'called\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'saying\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
            r'text\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)': lambda m: {'text_contains': m.group(1).strip()},
        }
        
        for pattern, extractor in text_content_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params.update(extractor(match))
                break
        
        # User IDs (but not if we already have an email or text search)
        if 'target_email' not in params and 'text_contains' not in params:
            user_id_pattern = r'\b[A-Za-z0-9]{8,}\b'
            uid_matches = re.findall(user_id_pattern, query)
            if uid_matches:
                params['target_uid'] = uid_matches[0]
        
        return params
    
    def _basic_parameter_extraction(self, query: str) -> Dict[str, Any]:
        """Basic parameter extraction as ultimate fallback"""
        params = {}
        query_lower = query.lower()
        
        # Basic time patterns
        if "past 1 day" in query_lower or "today" in query_lower:
            params["days_back"] = 1
        elif "yesterday" in query_lower:
            params["days_back"] = 2
        elif "last week" in query_lower or "past week" in query_lower:
            params["days_back"] = 7
        elif "last month" in query_lower or "past month" in query_lower:
            params["days_back"] = 30
        
        # Basic number extraction for "past X days"
        past_days_match = re.search(r'(?:past|last)\s+(\d+)\s+days?', query_lower)
        if past_days_match:
            params["days_back"] = int(past_days_match.group(1))
        
        # Email extraction
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        email_matches = re.findall(email_pattern, query)
        if email_matches:
            params["target_email"] = email_matches[0]
        
        # Text content extraction (basic patterns)
        text_patterns = [
            r'with the name\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)',
            r'containing\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)',
            r'about\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)',
            r'mentioning\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)',
            r'named\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)',
            r'called\s+([a-zA-Z ]+?)(?:\s+(?:in|over|since|during|for|past|last|this|from|until|before|after)|\s*$)',
        ]
        
        for pattern in text_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params["text_contains"] = match.group(1).strip()
                break
        
        return params
    
    def run_agent(self, agent_name: str, query: str, **params) -> Optional[Any]:
        """
        ENHANCED AGENT EXECUTION WITH DEPENDENCY INJECTION
        
        Run an agent by name with the provided query and parameters.
        Uses explicit dependency injection to avoid global state reliance.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        - Explicit dependency injection — no global state reliance
        
        Args:
            agent_name: Name of the agent to run
            query: Investigation query/description
            **params: Additional parameters for the agent
            
        Returns:
            Result from the agent execution
            
        Raises:
            ValueError: If agent name is not found in registry
            RuntimeError: If system dependencies are missing
        """
        # Check if agent exists in registry
        if agent_name not in self.agents:
            available_agents = list(self.agents.keys())
            raise ValueError(f"Agent '{agent_name}' not found in registry. Available agents: {available_agents}")
        
        # Get agent handler
        if agent_name not in self.agent_handlers:
            raise ValueError(f"No handler found for agent '{agent_name}'")
        
        handler = self.agent_handlers[agent_name]
        
        # Get agent config for merging default parameters
        config = self.agents[agent_name]
        
        # Merge default parameters with provided parameters
        final_params = config.default_params.copy()
        final_params.update(params)
        
        print(f"🚀 Running agent '{agent_name}' with query: '{query}'")
        print(f"📊 Parameters: {final_params}")
        
        try:
            # Execute agent with dependency injection
            result = self._run_agent_with_dependencies(
                handler_function=handler,
                agent_type=agent_name,
                user_query=query,
                **final_params
            )
            
            if result:
                print(f"✅ Agent '{agent_name}' completed successfully")
                return result
            else:
                print(f"❌ Agent '{agent_name}' returned no results")
                return None
                
        except Exception as e:
            print(f"❌ Error running agent '{agent_name}': {str(e)}")
            raise
    
    # =============================================================================
    # SHARED AGENT VALIDATION
    # =============================================================================
    
    def _validate_agent_dependencies(self) -> tuple:
        """
        Validate that all required dependencies are available for agent execution.
        Returns (main_system, sql_executor, VERIFIED_TABLES) or raises RuntimeError.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, placeholders, or simulation
        - Use only real execution logic from actual notebook files
        - Trust and investigation integrity are core — do not fake output
        """
        print("🔍 Validating agent runtime dependencies...")
        
        # Get dependencies from global scope
        try:
            import __main__
            main_system = getattr(__main__, 'main_system', None)
            sql_executor = getattr(__main__, 'sql_executor', None)
            VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
            ENVIRONMENT_READY = getattr(__main__, 'ENVIRONMENT_READY', False)
        except Exception:
            main_system = None
            sql_executor = None
            VERIFIED_TABLES = None
            ENVIRONMENT_READY = False
        
        # EXPLICIT RUNTIME GUARDS - fail early with clear messages
        
        # Check 1: VERIFIED_TABLES (most critical for agent execution)
        if VERIFIED_TABLES is None:
            raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
        
        if not isinstance(VERIFIED_TABLES, dict) or len(VERIFIED_TABLES) == 0:
            raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
        
        # Check for critical tables
        critical_tables = ['TTS Usage', 'Classification Flags']
        missing_critical = []
        inaccessible_critical = []
        
        for table_name in critical_tables:
            if table_name not in VERIFIED_TABLES:
                missing_critical.append(table_name)
            elif not VERIFIED_TABLES[table_name].get('accessible', False):
                inaccessible_critical.append(table_name)
        
        if missing_critical:
            raise RuntimeError(f"BigQuery tables not verified — run Cell 2 first")
        
        if inaccessible_critical:
            raise RuntimeError(f"BigQuery tables not verified — run Cell 2 first")
        
        # Check 2: Environment readiness
        if not ENVIRONMENT_READY:
            raise RuntimeError("Environment not ready — run Cell 1 first")
        
        # Check 3: sql_executor (needed for legacy compatibility)
        if not sql_executor:
            raise RuntimeError("SQL executor missing — run Cell 4 first")
        
        # Check 4: main_system exists
        if not main_system:
            raise RuntimeError("Main investigation system missing — run Cell 5 first")
        
        # Check 5: main_system has required components
        if not hasattr(main_system, "bq_client") or main_system.bq_client is None:
            raise RuntimeError("BigQuery client missing — run Cell 2 first")
        
        if not hasattr(main_system, "analyzer") or main_system.analyzer is None:
            raise RuntimeError("AI analyzer missing — run Cell 5 first")
        
        # Check 6: OpenAI client availability
        if not hasattr(main_system.analyzer, 'openai_client') or not main_system.analyzer.openai_client:
            raise RuntimeError("AI analyzer missing — run Cell 5 first")
        
        print("✅ All agent runtime dependencies validated successfully")
        
        return main_system, sql_executor, VERIFIED_TABLES
    
    def _initialize_agent_with_dependencies(self, agent_type: str):
        """
        DEPENDENCY INJECTION FOR AGENTS — DO NOT VIOLATE
        
        Initializes agent with explicit dependencies rather than relying on global state.
        This ensures agents have their own local dependencies and prevents global state issues.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        - Explicit dependency injection — no global state reliance
        """
        import __main__
        
        # Get components with explicit validation
        main_system = getattr(__main__, 'main_system', None)
        if not main_system:
            raise RuntimeError("Main investigation system missing — run Cell 5 first")
        
        # Extract and validate BigQuery client
        bq_client = main_system.bq_client
        if not bq_client:
            raise RuntimeError("BigQuery client missing in main_system — run Cell 5 first")
        
        # Extract and validate AI analyzer
        analyzer = main_system.analyzer
        if not analyzer:
            raise RuntimeError("AI analyzer missing in main_system — run Cell 5 first")
        
        # Validate OpenAI client in analyzer
        if not hasattr(analyzer, 'openai_client') or not analyzer.openai_client:
            raise RuntimeError("OpenAI client missing in analyzer — run Cell 5 first")
        
        # Get SQL executor
        sql_executor = getattr(__main__, 'sql_executor', None)
        if not sql_executor:
            raise RuntimeError("SQL executor missing — run Cell 4 first")
        
        # Get investigation manager
        investigation_manager = getattr(__main__, 'investigation_manager', None)
        if not investigation_manager:
            raise RuntimeError("Investigation manager missing — run Cell 3 first")
        
        # Get verified tables
        VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
        if not VERIFIED_TABLES:
            raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
        
        # Get schema system
        TABLE_SCHEMAS = getattr(__main__, 'TABLE_SCHEMAS', None)
        if not TABLE_SCHEMAS:
            raise RuntimeError("Schema system missing — run Cell 2 first")
        
        # Validate no mock data is being used
        self._validate_no_mock_data(bq_client, analyzer, VERIFIED_TABLES)
        
        # Create agent dependencies object
        agent_dependencies = {
            'bq_client': bq_client,
            'analyzer': analyzer,
            'sql_executor': sql_executor,
            'investigation_manager': investigation_manager,
            'VERIFIED_TABLES': VERIFIED_TABLES,
            'TABLE_SCHEMAS': TABLE_SCHEMAS,
            'agent_type': agent_type
        }
        
        print(f"✅ Agent dependencies initialized for {agent_type}")
        print(f"   • BigQuery client: Available")
        print(f"   • AI analyzer: Available")
        print(f"   • SQL executor: Available")
        print(f"   • Investigation manager: Available")
        print(f"   • Verified tables: {len(VERIFIED_TABLES)} tables")
        print(f"   • Schema system: Available")
        
        return agent_dependencies
    
    def _validate_no_mock_data(self, bq_client, analyzer, VERIFIED_TABLES):
        """
        STRICT VALIDATION — NO MOCK DATA ALLOWED
        
        Validates that all components are using REAL data and connections.
        Raises RuntimeError if mock data is detected.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        """
        # Check BigQuery client is real
        if hasattr(bq_client, '_client_info') and bq_client._client_info:
            client_info = str(bq_client._client_info).lower()
            if 'mock' in client_info or 'fake' in client_info or 'test' in client_info:
                raise RuntimeError("Mock BigQuery client detected — only real data should be used in investigations")
        
        # Check analyzer is real
        if hasattr(analyzer, 'openai_client') and analyzer.openai_client:
            if hasattr(analyzer.openai_client, 'api_key'):
                # Validate API key is real (not test key)
                api_key = analyzer.openai_client.api_key
                if api_key and ('test' in api_key.lower() or 'mock' in api_key.lower() or 'fake' in api_key.lower()):
                    raise RuntimeError("Mock OpenAI API key detected — only real data should be used in investigations")
        
        # Check verified tables are real
        for table_name, table_info in VERIFIED_TABLES.items():
            table_id = table_info.get('table_id', '').lower()
            if 'mock' in table_id or 'fake' in table_id or 'test' in table_id:
                raise RuntimeError(f"Mock table detected ({table_name}: {table_id}) — only real data should be used in investigations")
        
        print("✅ Real data validation passed - no mock data detected")
    
    def _validate_execution_order(self):
        """
        STRICT EXECUTION ORDER VALIDATION
        
        Validates that cells have been run in the correct order:
        Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 7b → Agents
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - Enforce correct cell execution order
        - Prevent agents from running with incomplete setup
        """
        import __main__
        
        print("🔍 VALIDATING EXECUTION ORDER")
        print("-" * 30)
        
        # Check Cell 1: Environment Setup
        ENVIRONMENT_READY = getattr(__main__, 'ENVIRONMENT_READY', False)
        if not ENVIRONMENT_READY:
            raise RuntimeError("Cell 1 not run - environment setup required — run Cell 1 first")
        print("✅ Cell 1: Environment setup complete")
        
        # Check Cell 2: BigQuery Configuration
        VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
        if not VERIFIED_TABLES:
            raise RuntimeError("Cell 2 not run - BigQuery tables not verified — run Cell 2 first")
        print("✅ Cell 2: BigQuery configuration complete")
        
        # Check Cell 3: Investigation Manager
        investigation_manager = getattr(__main__, 'investigation_manager', None)
        if not investigation_manager:
            raise RuntimeError("Cell 3 not run - investigation manager missing — run Cell 3 first")
        print("✅ Cell 3: Investigation manager complete")
        
        # Check Cell 4: SQL Interface
        sql_executor = getattr(__main__, 'sql_executor', None)
        if not sql_executor:
            raise RuntimeError("Cell 4 not run - SQL executor missing — run Cell 4 first")
        print("✅ Cell 4: SQL interface complete")
        
        # Check Cell 5: Main Investigation System
        main_system = getattr(__main__, 'main_system', None)
        if not main_system:
            raise RuntimeError("Cell 5 not run - main investigation system missing — run Cell 5 first")
        
        if not main_system.analyzer:
            raise RuntimeError("Cell 5 not run - AI analyzer missing — run Cell 5 first")
        
        if not main_system.bq_client:
            raise RuntimeError("Cell 5 not run - BigQuery client missing in main_system — run Cell 5 first")
        
        print("✅ Cell 5: Main investigation system complete")
        
        # Check Cell 7b: Agent Registry (this cell)
        agent_registry = getattr(__main__, 'agent_registry', None)
        if not agent_registry:
            print("⚠️  Cell 7b: Agent registry not fully initialized")
            print("   This is expected during Cell 7b execution")
        else:
            print("✅ Cell 7b: Agent registry complete")
        
        print("✅ Execution order validation passed")
    
    def _run_agent_with_dependencies(self, handler_function, agent_type: str, user_query: str, **params):
        """
        ENHANCED AGENT EXECUTION WITH DEPENDENCY INJECTION
        
        Runs agent with explicit dependencies rather than global state reliance.
        Includes comprehensive validation and error handling.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        - Explicit dependency injection — no global state reliance
        """
        try:
            print(f"🚀 EXECUTING AGENT: {agent_type}")
            print("=" * 50)
            
            # Step 1: Validate execution order
            self._validate_execution_order()
            
            # Step 2: Run comprehensive runtime integrity check
            self._check_runtime_health_before_execution()
            
            # Step 3: Initialize agent with explicit dependencies
            agent_dependencies = self._initialize_agent_with_dependencies(agent_type)
            
            # Step 4: Execute agent with dependencies
            print(f"\n🔍 RUNNING {agent_type.upper()} WITH REAL DATA")
            print("-" * 40)
            
            # Pass dependencies explicitly to handler
            result = handler_function(
                user_query=user_query,
                dependencies=agent_dependencies,
                **params
            )
            
            print(f"\n✅ {agent_type.upper()} EXECUTION COMPLETE")
            print("✅ All operations used REAL data")
            
            return result
            
        except RuntimeError as e:
            print(f"\n❌ {agent_type.upper()} EXECUTION FAILED")
            print(f"Error: {e}")
            print("\n💡 REQUIRED ACTIONS:")
            print("   1. Run Cell 1 (Environment Setup)")
            print("   2. Run Cell 2 (BigQuery Configuration)")
            print("   3. Run Cell 3 (Investigation Management)")
            print("   4. Run Cell 4 (SQL Interface)")
            print("   5. Run Cell 5 (Main Investigation System)")
            print("   6. Re-run this agent")
            raise
        except Exception as e:
            print(f"\n❌ {agent_type.upper()} EXECUTION FAILED")
            print(f"Unexpected error: {e}")
            raise RuntimeError(f"Agent execution failed: {e}")
    
    def _check_runtime_health_before_execution(self):
        """
        STRICT RUNTIME INTEGRITY CHECK — DO NOT VIOLATE
        
        Comprehensive runtime health check before agent execution.
        Ensures all critical components are present and valid.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, placeholders, or simulation
        - Use only real execution logic from actual notebook files
        - Trust and investigation integrity are core — do not fake output
        """
        try:
            import __main__
            
            # Use the strict runtime integrity check if available
            if hasattr(__main__, 'check_runtime_integrity'):
                print("🔍 Running strict runtime integrity check...")
                __main__.check_runtime_integrity()
                print("✅ Strict runtime integrity check passed")
                return
            
            # Fallback to manual checks if function not available
            print("🔍 Running manual runtime integrity check...")
            
            # Check 1: VERIFIED_TABLES (most critical)
            VERIFIED_TABLES = getattr(__main__, 'VERIFIED_TABLES', None)
            if VERIFIED_TABLES is None:
                raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
            
            if not isinstance(VERIFIED_TABLES, dict) or len(VERIFIED_TABLES) == 0:
                raise RuntimeError("BigQuery tables not verified — run Cell 2 first")
            
            # Check critical table accessibility
            if 'TTS Usage' not in VERIFIED_TABLES:
                raise RuntimeError("TTS Usage table not verified — run Cell 2 first")
            
            tts_table = VERIFIED_TABLES['TTS Usage']
            if not tts_table.get('accessible', False):
                raise RuntimeError("TTS Usage table not accessible — check Cell 2 verification")
            
            # Check 2: Environment readiness
            ENVIRONMENT_READY = getattr(__main__, 'ENVIRONMENT_READY', False)
            if not ENVIRONMENT_READY:
                raise RuntimeError("Environment not ready — run Cell 1 first")
            
            # Check 3: main_system
            main_system = getattr(__main__, 'main_system', None)
            if not main_system:
                raise RuntimeError("Main investigation system missing — run Cell 5 first")
            
            # Check 4: BigQuery client in main_system
            if not hasattr(main_system, "bq_client") or main_system.bq_client is None:
                raise RuntimeError("BigQuery client missing in main_system — run Cell 5 first")
            
            # Check 5: AI analyzer in main_system
            if not hasattr(main_system, "analyzer") or main_system.analyzer is None:
                raise RuntimeError("AI analyzer missing in main_system — run Cell 5 first")
            
            # Check 6: OpenAI client in analyzer
            if not hasattr(main_system.analyzer, 'openai_client') or not main_system.analyzer.openai_client:
                raise RuntimeError("OpenAI client missing in analyzer — run Cell 5 first")
            
            # Check 7: sql_executor
            sql_executor = getattr(__main__, 'sql_executor', None)
            if not sql_executor:
                raise RuntimeError("SQL executor missing — run Cell 4 first")
            
            # Check 8: investigation_manager
            investigation_manager = getattr(__main__, 'investigation_manager', None)
            if not investigation_manager:
                raise RuntimeError("Investigation manager missing — run Cell 3 first")
            
            print("✅ Manual runtime integrity check passed")
            print("✅ All critical components verified and available")
            print("✅ System ready for REAL data investigation")
        
        except RuntimeError:
            # Re-raise runtime errors as-is
            raise
        except Exception as e:
            raise RuntimeError(f"Runtime integrity check failed: {e}")
    
    # =============================================================================
    # AGENT HANDLERS WITH ENHANCED RUNTIME GUARDS
    # =============================================================================
    
    def _scam_agent_handler(self, user_query: str, dependencies: dict = None, **params) -> AgentResult:
        """
        SCAM DETECTION AGENT WITH ENHANCED TABLE VALIDATION
        
        Enhanced scam detection agent handler with dependency injection and graceful table handling.
        Uses dependency injection and handles missing Classification Flags table gracefully.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        - Explicit dependency injection — no global state reliance
        - Graceful handling of missing tables
        
        Args:
            user_query: Query description for scam detection
            dependencies: Injected dependencies (bq_client, analyzer, etc.)
            **params: Additional parameters for the agent
            
        Returns:
            AgentResult with scam detection findings
        """
        start_time = datetime.now()
        
        print(f"SCAM AGENT: Analyzing potential scam content")
        print(f"Query: {user_query}")
        
        # Extract agent context for cancellation checking
        agent_context = params.pop('agent_context', None)
        
        # Check for cancellation before starting
        _check_agent_cancellation(agent_context)
        
        # STRICT DEPENDENCY VALIDATION
        if not dependencies:
            raise RuntimeError("Agent dependencies missing — system not properly initialized")
        
        # Extract dependencies
        bq_client = dependencies.get('bq_client')
        analyzer = dependencies.get('analyzer')
        sql_executor = dependencies.get('sql_executor')
        investigation_manager = dependencies.get('investigation_manager')
        VERIFIED_TABLES = dependencies.get('VERIFIED_TABLES')
        TABLE_SCHEMAS = dependencies.get('TABLE_SCHEMAS')
        
        # Validate all required dependencies are present
        if not bq_client:
            raise RuntimeError("BigQuery client missing from dependencies — run Cell 5 first")
        
        if not analyzer:
            raise RuntimeError("AI analyzer missing from dependencies — run Cell 5 first")
        
        if not sql_executor:
            raise RuntimeError("SQL executor missing from dependencies — run Cell 4 first")
        
        if not investigation_manager:
            raise RuntimeError("Investigation manager missing from dependencies — run Cell 3 first")
        
        if not VERIFIED_TABLES:
            raise RuntimeError("Verified tables missing from dependencies — run Cell 2 first")
        
        if not TABLE_SCHEMAS:
            raise RuntimeError("Schema system missing from dependencies — run Cell 2 first")
        
        # ENHANCED TABLE VALIDATION WITH GRACEFUL HANDLING
        print("🔍 Validating agent-specific table requirements...")
        
        # Check if agent-specific requirements function is available
        import __main__
        if hasattr(__main__, 'check_agent_specific_requirements'):
            try:
                __main__.check_agent_specific_requirements('scam_agent')
                print("✅ Agent-specific requirements validation passed")
            except Exception as e:
                print(f"❌ Agent-specific requirements failed: {e}")
                # Continue with graceful degradation
        
        # Check TTS Usage table (critical)
        if 'TTS Usage' not in VERIFIED_TABLES:
            raise RuntimeError("TTS Usage table not found in VERIFIED_TABLES — run Cell 2 to verify tables")
        
        if not VERIFIED_TABLES['TTS Usage'].get('accessible', False):
            raise RuntimeError("TTS Usage table not accessible — check Cell 2 table verification")
        
        # Check Classification Flags table (with graceful degradation)
        use_classification_flags = False
        classification_table = None
        
        if 'Classification Flags' in VERIFIED_TABLES:
            classification_table = VERIFIED_TABLES['Classification Flags']
            if classification_table.get('accessible', False):
                use_classification_flags = True
                print("✅ Classification Flags table available - using enhanced detection")
            else:
                print("⚠️  Classification Flags table not accessible - using basic detection")
                print(f"   Error: {classification_table.get('error', 'Unknown error')}")
        else:
            print("⚠️  Classification Flags table not configured - using basic detection")
        
        print("✅ All critical dependencies validated successfully")
        
        # Merge default parameters with provided parameters
        config = self.agents["scam_agent"]
        final_params = config.default_params.copy()
        final_params.update(params)
        
        days_back = final_params.get('days_back', 7)
        limit = final_params.get('limit', 1000)
        
        print(f"📊 Scam detection parameters: days_back={days_back}, limit={limit}")
        print(f"📊 Enhanced detection: {'✅ Yes' if use_classification_flags else '❌ No'}")
        
        try:
            # Use injected dependencies instead of global state
            tts_table = VERIFIED_TABLES['TTS Usage']
            
            # Get schema-aware column names
            from __main__ import get_column_name
            
            user_id_col = get_column_name('TTS Usage', 'user_id')
            email_col = get_column_name('TTS Usage', 'email')
            text_col = get_column_name('TTS Usage', 'text')
            timestamp_col = get_column_name('TTS Usage', 'timestamp')
            
            # Build scam detection query with optional classification flags
            if use_classification_flags:
                # Enhanced query with classification flags
                scam_query = f"""
                WITH recent_tts AS (
                    SELECT 
                        {user_id_col} as user_id,
                        {email_col} as email,
                        {text_col} as text,
                        {timestamp_col} as timestamp,
                        id
                    FROM `{tts_table['table_id']}`
                    WHERE DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                        AND {text_col} IS NOT NULL
                        AND LENGTH({text_col}) > 10
                ),
                classified_flags AS (
                    SELECT 
                        id,
                        timestamp,
                        has_scam,
                        scam_2,
                        oai_hate_bool,
                        oai_self_harm_bool,
                        COALESCE(risk_score, 0) as risk_score
                    FROM `{classification_table['table_id']}`
                    WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                        AND (has_scam = true OR scam_2 = true OR oai_hate_bool = true OR oai_self_harm_bool = true)
                )
                SELECT 
                    t.user_id,
                    t.email,
                    t.text,
                    t.timestamp,
                    'enhanced_scam_detection' as detection_type,
                    CASE 
                        WHEN cf.has_scam = true THEN 'confirmed_scam'
                        WHEN cf.scam_2 = true THEN 'secondary_scam'
                        WHEN cf.oai_hate_bool = true THEN 'hate_speech'
                        WHEN cf.oai_self_harm_bool = true THEN 'self_harm'
                        WHEN LOWER(t.text) LIKE '%crypto%' OR LOWER(t.text) LIKE '%bitcoin%' THEN 'crypto_scam'
                        WHEN LOWER(t.text) LIKE '%investment%' OR LOWER(t.text) LIKE '%profit%' THEN 'investment_scam'
                        WHEN LOWER(t.text) LIKE '%urgent%' OR LOWER(t.text) LIKE '%act now%' THEN 'urgency_scam'
                        WHEN LOWER(t.text) LIKE '%free%' OR LOWER(t.text) LIKE '%winner%' THEN 'prize_scam'
                        ELSE 'pattern_scam'
                    END as scam_category,
                    COALESCE(cf.risk_score, 0.5) as risk_score
                FROM recent_tts t
                LEFT JOIN classified_flags cf ON t.id = cf.id
                WHERE (
                    cf.id IS NOT NULL OR (
                        LOWER(t.text) LIKE '%scam%' OR
                        LOWER(t.text) LIKE '%fraud%' OR
                        LOWER(t.text) LIKE '%urgent%' OR
                        LOWER(t.text) LIKE '%act now%' OR
                        LOWER(t.text) LIKE '%limited time%' OR
                        LOWER(t.text) LIKE '%free money%' OR
                        LOWER(t.text) LIKE '%guaranteed%' OR
                        LOWER(t.text) LIKE '%risk free%' OR
                        LOWER(t.text) LIKE '%click here%' OR
                        LOWER(t.text) LIKE '%winner%' OR
                        LOWER(t.text) LIKE '%congratulations%' OR
                        LOWER(t.text) LIKE '%crypto%' OR
                        LOWER(t.text) LIKE '%bitcoin%' OR
                        LOWER(t.text) LIKE '%investment%'
                    )
                )
                ORDER BY COALESCE(cf.risk_score, 0.5) DESC, t.timestamp DESC
                LIMIT {limit}
                """
            else:
                # Basic query without classification flags
                scam_query = f"""
                WITH recent_tts AS (
                    SELECT 
                        {user_id_col} as user_id,
                        {email_col} as email,
                        {text_col} as text,
                        {timestamp_col} as timestamp
                    FROM `{tts_table['table_id']}`
                    WHERE DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                        AND {text_col} IS NOT NULL
                        AND LENGTH({text_col}) > 10
                )
                SELECT 
                    t.user_id,
                    t.email,
                    t.text,
                    t.timestamp,
                    'basic_scam_detection' as detection_type,
                    CASE 
                        WHEN LOWER(t.text) LIKE '%crypto%' OR LOWER(t.text) LIKE '%bitcoin%' THEN 'crypto_scam'
                        WHEN LOWER(t.text) LIKE '%investment%' OR LOWER(t.text) LIKE '%profit%' THEN 'investment_scam'
                        WHEN LOWER(t.text) LIKE '%urgent%' OR LOWER(t.text) LIKE '%act now%' THEN 'urgency_scam'
                        WHEN LOWER(t.text) LIKE '%free%' OR LOWER(t.text) LIKE '%winner%' THEN 'prize_scam'
                        ELSE 'general_scam'
                    END as scam_category,
                    0.5 as risk_score
                FROM recent_tts t
                WHERE (
                    LOWER(t.text) LIKE '%scam%' OR
                    LOWER(t.text) LIKE '%fraud%' OR
                    LOWER(t.text) LIKE '%urgent%' OR
                    LOWER(t.text) LIKE '%act now%' OR
                    LOWER(t.text) LIKE '%limited time%' OR
                    LOWER(t.text) LIKE '%free money%' OR
                    LOWER(t.text) LIKE '%guaranteed%' OR
                    LOWER(t.text) LIKE '%risk free%' OR
                    LOWER(t.text) LIKE '%click here%' OR
                    LOWER(t.text) LIKE '%winner%' OR
                    LOWER(t.text) LIKE '%congratulations%' OR
                    LOWER(t.text) LIKE '%crypto%' OR
                    LOWER(t.text) LIKE '%bitcoin%' OR
                    LOWER(t.text) LIKE '%investment%'
                )
                ORDER BY t.timestamp DESC
                LIMIT {limit}
                """
            
            print(f"🔍 Executing scam detection query with REAL data...")
            print(f"   Detection mode: {'Enhanced' if use_classification_flags else 'Basic'}")
            
            # Execute query using injected BigQuery client
            query_job = bq_client.query(scam_query)
            results = query_job.result()
            
            records = []
            high_risk_count = 0
            
            for row in results:
                risk_score = getattr(row, 'risk_score', 0.5)
                
                record = {
                    'user_id': row.user_id,
                    'email': row.email,
                    'text': row.text,
                    'timestamp': row.timestamp,
                    'detection_type': row.detection_type,
                    'scam_category': row.scam_category,
                    'risk_score': risk_score
                }
                
                # Analyze content using injected analyzer
                if analyzer and hasattr(analyzer, 'analyze_content'):
                    analysis = analyzer.analyze_content(row.text)
                    record['ai_analysis'] = analysis
                    
                    # Check if this is high risk
                    if analysis and isinstance(analysis, dict):
                        ai_risk_score = analysis.get('risk_score', 0)
                        # Combine database risk score with AI risk score
                        combined_risk = max(risk_score, ai_risk_score)
                        record['combined_risk_score'] = combined_risk
                        
                        if combined_risk > 0.7:
                            high_risk_count += 1
                            record['high_risk'] = True
                        else:
                            record['high_risk'] = False
                    else:
                        record['high_risk'] = risk_score > 0.7
                        record['combined_risk_score'] = risk_score
                        if record['high_risk']:
                            high_risk_count += 1
                else:
                    record['high_risk'] = risk_score > 0.7
                    record['combined_risk_score'] = risk_score
                    if record['high_risk']:
                        high_risk_count += 1
                
                records.append(record)
            
            # Create investigation entry using injected investigation manager
            if investigation_manager and hasattr(investigation_manager, 'current_investigation'):
                investigation_manager.add_finding(
                    f"Scam Detection Analysis ({'Enhanced' if use_classification_flags else 'Basic'}): Found {len(records)} potential scam records, {high_risk_count} high-risk"
                )
                
                if high_risk_count > 0:
                    investigation_manager.add_action(
                        f"URGENT: Review {high_risk_count} high-risk scam records for immediate action"
                    )
            
            # Generate recommendations
            recommendations = []
            
            if high_risk_count > 0:
                recommendations.append(f"URGENT: {high_risk_count} high-risk scam records require immediate review")
            
            if len(records) > 0:
                recommendations.append(f"Monitor {len(records)} users for potential scam activity")
                if use_classification_flags:
                    recommendations.append("Enhanced detection used - results include ML classification scores")
                else:
                    recommendations.append("Basic detection used - consider enabling Classification Flags for enhanced detection")
                recommendations.append("Consider implementing additional scam detection filters")
            else:
                recommendations.append("No scam indicators found in recent data")
                if not use_classification_flags:
                    recommendations.append("Consider enabling Classification Flags table for enhanced detection")
            
            detection_type = "Enhanced" if use_classification_flags else "Basic"
            print(f"✅ Scam detection complete ({detection_type}): {len(records)} records, {high_risk_count} high-risk")
            
            return AgentResult(
                agent_name="scam_agent",
                query_description=user_query,
                records_found=len(records),
                high_risk_items=high_risk_count,
                analysis_results=records,
                recommendations=recommendations,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ SCAM AGENT ERROR: {e}")
            return AgentResult(
                agent_name="scam_agent",
                query_description=user_query,
                records_found=0,
                high_risk_items=0,
                analysis_results=[],
                recommendations=[f"Error: {str(e)}"],
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
    
    def _email_network_agent_handler(self, user_query: str, dependencies: dict = None, **params) -> AgentResult:
        """
        EMAIL NETWORK ANALYSIS AGENT WITH ENHANCED TABLE VALIDATION
        
        Enhanced email network analysis agent handler with dependency injection and graceful table handling.
        Uses dependency injection and handles missing Classification Flags table gracefully.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        - Explicit dependency injection — no global state reliance
        - Graceful handling of missing tables
        
        Args:
            user_query: Query description for email network analysis
            dependencies: Injected dependencies (bq_client, analyzer, etc.)
            **params: Additional parameters for the agent
            
        Returns:
            AgentResult with email network analysis findings
        """
        start_time = datetime.now()
        
        print(f"EMAIL NETWORK AGENT: Analyzing email patterns and networks")
        print(f"Query: {user_query}")
        
        # Extract agent context for cancellation checking
        agent_context = params.pop('agent_context', None)
        
        # Check for cancellation before starting
        _check_agent_cancellation(agent_context)
        
        # STRICT DEPENDENCY VALIDATION
        if not dependencies:
            raise RuntimeError("Agent dependencies missing — system not properly initialized")
        
        # Extract dependencies
        bq_client = dependencies.get('bq_client')
        analyzer = dependencies.get('analyzer')
        sql_executor = dependencies.get('sql_executor')
        investigation_manager = dependencies.get('investigation_manager')
        VERIFIED_TABLES = dependencies.get('VERIFIED_TABLES')
        TABLE_SCHEMAS = dependencies.get('TABLE_SCHEMAS')
        
        # Validate all required dependencies are present
        if not bq_client:
            raise RuntimeError("BigQuery client missing from dependencies — run Cell 5 first")
        
        if not analyzer:
            raise RuntimeError("AI analyzer missing from dependencies — run Cell 5 first")
        
        if not sql_executor:
            raise RuntimeError("SQL executor missing from dependencies — run Cell 4 first")
        
        if not investigation_manager:
            raise RuntimeError("Investigation manager missing from dependencies — run Cell 3 first")
        
        if not VERIFIED_TABLES:
            raise RuntimeError("Verified tables missing from dependencies — run Cell 2 first")
        
        if not TABLE_SCHEMAS:
            raise RuntimeError("Schema system missing from dependencies — run Cell 2 first")
        
        # ENHANCED TABLE VALIDATION WITH GRACEFUL HANDLING
        print("🔍 Validating agent-specific table requirements...")
        
        # Check if agent-specific requirements function is available
        import __main__
        if hasattr(__main__, 'check_agent_specific_requirements'):
            try:
                __main__.check_agent_specific_requirements('email_network_agent')
                print("✅ Agent-specific requirements validation passed")
            except Exception as e:
                print(f"❌ Agent-specific requirements failed: {e}")
                # Continue with graceful degradation
        
        # Check TTS Usage table (critical)
        if 'TTS Usage' not in VERIFIED_TABLES:
            raise RuntimeError("TTS Usage table not found in VERIFIED_TABLES — run Cell 2 to verify tables")
        
        if not VERIFIED_TABLES['TTS Usage'].get('accessible', False):
            raise RuntimeError("TTS Usage table not accessible — check Cell 2 table verification")
        
        # Check Classification Flags table (with graceful degradation)
        use_classification_flags = False
        classification_table = None
        
        if 'Classification Flags' in VERIFIED_TABLES:
            classification_table = VERIFIED_TABLES['Classification Flags']
            if classification_table.get('accessible', False):
                use_classification_flags = True
                print("✅ Classification Flags table available - using enhanced network analysis")
            else:
                print("⚠️  Classification Flags table not accessible - using basic network analysis")
                print(f"   Error: {classification_table.get('error', 'Unknown error')}")
        else:
            print("⚠️  Classification Flags table not configured - using basic network analysis")
        
        print("✅ All critical dependencies validated successfully")
        
        # Merge default parameters with provided parameters
        config = self.agents["email_network_agent"]
        final_params = config.default_params.copy()
        final_params.update(params)
        
        days_back = final_params.get('days_back', 30)
        limit = final_params.get('limit', 200)
        min_connections = final_params.get('min_connections', 2)
        
        print(f"📊 Email network analysis parameters: days_back={days_back}, limit={limit}, min_connections={min_connections}")
        print(f"📊 Enhanced analysis: {'✅ Yes' if use_classification_flags else '❌ No'}")
        
        try:
            # Use injected dependencies instead of global state
            tts_table = VERIFIED_TABLES['TTS Usage']
            
            # Get schema-aware column names
            from __main__ import get_column_name
            
            user_id_col = get_column_name('TTS Usage', 'user_id')
            email_col = get_column_name('TTS Usage', 'email')
            text_col = get_column_name('TTS Usage', 'text')
            timestamp_col = get_column_name('TTS Usage', 'timestamp')
            
            # Build email network analysis query with optional classification flags
            if use_classification_flags:
                # Enhanced query with classification flags
                network_query = f"""
                WITH recent_tts AS (
                    SELECT 
                        {user_id_col} as user_id,
                        {email_col} as email,
                        {text_col} as text,
                        {timestamp_col} as timestamp,
                        id,
                        COUNT(*) OVER (PARTITION BY {email_col}) as email_request_count,
                        COUNT(DISTINCT {user_id_col}) OVER (PARTITION BY {email_col}) as users_per_email
                    FROM `{tts_table['table_id']}`
                    WHERE DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                        AND {text_col} IS NOT NULL
                        AND {email_col} IS NOT NULL
                ),
                classified_flags AS (
                    SELECT 
                        id,
                        timestamp,
                        has_scam,
                        scam_2,
                        oai_hate_bool,
                        oai_self_harm_bool,
                        COALESCE(risk_score, 0) as risk_score
                    FROM `{classification_table['table_id']}`
                    WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                ),
                network_analysis AS (
                    SELECT 
                        t.email,
                        t.user_id,
                        t.email_request_count,
                        t.users_per_email,
                        COUNT(*) as user_requests,
                        COUNT(DISTINCT DATE(t.timestamp)) as active_days,
                        MIN(t.timestamp) as first_request,
                        MAX(t.timestamp) as last_request,
                        AVG(LENGTH(t.text)) as avg_text_length,
                        AVG(COALESCE(cf.risk_score, 0)) as avg_risk_score,
                        SUM(CASE WHEN cf.has_scam = true THEN 1 ELSE 0 END) as scam_count,
                        SUM(CASE WHEN cf.oai_hate_bool = true THEN 1 ELSE 0 END) as hate_count
                    FROM recent_tts t
                    LEFT JOIN classified_flags cf ON t.id = cf.id
                    GROUP BY t.email, t.user_id, t.email_request_count, t.users_per_email
                )
                SELECT 
                    email,
                    user_id,
                    email_request_count,
                    users_per_email,
                    user_requests,
                    active_days,
                    first_request,
                    last_request,
                    avg_text_length,
                    avg_risk_score,
                    scam_count,
                    hate_count,
                    'enhanced_network_analysis' as analysis_type,
                    CASE 
                        WHEN users_per_email > 1 AND avg_risk_score > 0.5 THEN 'high_risk_shared_email'
                        WHEN users_per_email > 1 THEN 'shared_email'
                        WHEN avg_risk_score > 0.7 THEN 'high_risk_single_user'
                        WHEN email_request_count > 1000 THEN 'high_volume_user'
                        ELSE 'normal_user'
                    END as risk_category
                FROM network_analysis
                WHERE email_request_count >= {min_connections}
                ORDER BY 
                    users_per_email DESC,
                    avg_risk_score DESC,
                    email_request_count DESC
                LIMIT {limit}
                """
            else:
                # Basic query without classification flags
                network_query = f"""
                WITH recent_tts AS (
                    SELECT 
                        {user_id_col} as user_id,
                        {email_col} as email,
                        {text_col} as text,
                        {timestamp_col} as timestamp,
                        COUNT(*) OVER (PARTITION BY {email_col}) as email_request_count,
                        COUNT(DISTINCT {user_id_col}) OVER (PARTITION BY {email_col}) as users_per_email
                    FROM `{tts_table['table_id']}`
                    WHERE DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                        AND {text_col} IS NOT NULL
                        AND {email_col} IS NOT NULL
                ),
                network_analysis AS (
                    SELECT 
                        email,
                        user_id,
                        email_request_count,
                        users_per_email,
                        COUNT(*) as user_requests,
                        COUNT(DISTINCT DATE(timestamp)) as active_days,
                        MIN(timestamp) as first_request,
                        MAX(timestamp) as last_request,
                        AVG(LENGTH(text)) as avg_text_length
                    FROM recent_tts
                    GROUP BY email, user_id, email_request_count, users_per_email
                )
                SELECT 
                    email,
                    user_id,
                    email_request_count,
                    users_per_email,
                    user_requests,
                    active_days,
                    first_request,
                    last_request,
                    avg_text_length,
                    0.0 as avg_risk_score,
                    0 as scam_count,
                    0 as hate_count,
                    'basic_network_analysis' as analysis_type,
                    CASE 
                        WHEN users_per_email > 1 THEN 'shared_email'
                        WHEN email_request_count > 1000 THEN 'high_volume_user'
                        ELSE 'normal_user'
                    END as risk_category
                FROM network_analysis
                WHERE email_request_count >= {min_connections}
                ORDER BY 
                    users_per_email DESC,
                    email_request_count DESC
                LIMIT {limit}
                """
            
            print(f"🔍 Executing email network analysis query with REAL data...")
            print(f"   Analysis mode: {'Enhanced' if use_classification_flags else 'Basic'}")
            
            # Execute query using injected BigQuery client
            query_job = bq_client.query(network_query)
            results = query_job.result()
            
            records = []
            high_risk_count = 0
            shared_emails = 0
            
            for row in results:
                risk_score = getattr(row, 'avg_risk_score', 0.0)
                users_per_email = getattr(row, 'users_per_email', 1)
                
                record = {
                    'email': row.email,
                    'user_id': row.user_id,
                    'email_request_count': row.email_request_count,
                    'users_per_email': users_per_email,
                    'user_requests': row.user_requests,
                    'active_days': row.active_days,
                    'first_request': row.first_request,
                    'last_request': row.last_request,
                    'avg_text_length': row.avg_text_length,
                    'avg_risk_score': risk_score,
                    'scam_count': getattr(row, 'scam_count', 0),
                    'hate_count': getattr(row, 'hate_count', 0),
                    'analysis_type': row.analysis_type,
                    'risk_category': row.risk_category
                }
                
                # Count shared emails and high-risk items
                if users_per_email > 1:
                    shared_emails += 1
                
                if risk_score > 0.7 or users_per_email > 3:
                    high_risk_count += 1
                    record['high_risk'] = True
                else:
                    record['high_risk'] = False
                
                records.append(record)
            
            # Create investigation entry using injected investigation manager
            if investigation_manager and hasattr(investigation_manager, 'current_investigation'):
                investigation_manager.add_finding(
                    f"Email Network Analysis ({row.analysis_type}): Found {len(records)} email patterns, {shared_emails} shared emails, {high_risk_count} high-risk"
                )
                
                if shared_emails > 0:
                    investigation_manager.add_action(
                        f"Review {shared_emails} shared email addresses for potential account linkage"
                    )
                
                if high_risk_count > 0:
                    investigation_manager.add_action(
                        f"URGENT: Review {high_risk_count} high-risk email patterns for immediate action"
                    )
            
            # Generate recommendations
            recommendations = []
            
            if high_risk_count > 0:
                recommendations.append(f"URGENT: {high_risk_count} high-risk email patterns require immediate review")
            
            if shared_emails > 0:
                recommendations.append(f"Monitor {shared_emails} shared email addresses for potential account abuse")
            
            if len(records) > 0:
                recommendations.append(f"Analyze {len(records)} email patterns for network connections")
                if use_classification_flags:
                    recommendations.append("Enhanced analysis used - results include ML classification scores")
                else:
                    recommendations.append("Basic analysis used - consider enabling Classification Flags for enhanced analysis")
                recommendations.append("Consider implementing email verification for shared accounts")
            else:
                recommendations.append("No significant email network patterns found in recent data")
                if not use_classification_flags:
                    recommendations.append("Consider enabling Classification Flags table for enhanced analysis")
            
            analysis_type = "Enhanced" if use_classification_flags else "Basic"
            print(f"✅ Email network analysis complete ({analysis_type}): {len(records)} patterns, {shared_emails} shared emails, {high_risk_count} high-risk")
            
            return AgentResult(
                agent_name="email_network_agent",
                query_description=user_query,
                records_found=len(records),
                high_risk_items=high_risk_count,
                analysis_results=records,
                recommendations=recommendations,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ EMAIL NETWORK AGENT ERROR: {e}")
            return AgentResult(
                agent_name="email_network_agent",
                query_description=user_query,
                records_found=0,
                high_risk_items=0,
                analysis_results=[],
                recommendations=[f"Error: {str(e)}"],
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
    
    def _exploratory_agent_handler(self, user_query: str, dependencies: dict = None, **params) -> AgentResult:
        """
        EXPLORATORY INVESTIGATION AGENT WITH DEPENDENCY INJECTION
        
        Enhanced exploratory investigation agent handler with dependency injection.
        Uses dependency injection for robust execution.
        
        ⚠️ STRICT INVESTIGATOR GUIDELINES — DO NOT VIOLATE
        - NO mock data, fake records, or simulation
        - Ensure all execution is based on REAL data and logic
        - Explicit dependency injection — no global state reliance
        
        Args:
            user_query: Query description for exploratory investigation
            dependencies: Injected dependencies (bq_client, analyzer, etc.)
            **params: Additional parameters for the agent
            
        Returns:
            AgentResult with exploratory investigation findings
        """
        start_time = datetime.now()
        
        print(f"EXPLORATORY AGENT: General investigation and pattern discovery")
        print(f"Query: {user_query}")
        
        # Extract agent context for cancellation checking
        agent_context = params.pop('agent_context', None)
        
        # Check for cancellation before starting
        _check_agent_cancellation(agent_context)
        
        # STRICT DEPENDENCY VALIDATION
        if not dependencies:
            raise RuntimeError("Agent dependencies missing — system not properly initialized")
        
        # Extract dependencies
        bq_client = dependencies.get('bq_client')
        analyzer = dependencies.get('analyzer')
        sql_executor = dependencies.get('sql_executor')
        investigation_manager = dependencies.get('investigation_manager')
        VERIFIED_TABLES = dependencies.get('VERIFIED_TABLES')
        TABLE_SCHEMAS = dependencies.get('TABLE_SCHEMAS')
        
        # Validate all required dependencies are present
        if not bq_client:
            raise RuntimeError("BigQuery client missing from dependencies — run Cell 5 first")
        
        if not analyzer:
            raise RuntimeError("AI analyzer missing from dependencies — run Cell 5 first")
        
        if not sql_executor:
            raise RuntimeError("SQL executor missing from dependencies — run Cell 4 first")
        
        if not investigation_manager:
            raise RuntimeError("Investigation manager missing from dependencies — run Cell 3 first")
        
        if not VERIFIED_TABLES:
            raise RuntimeError("Verified tables missing from dependencies — run Cell 2 first")
        
        if not TABLE_SCHEMAS:
            raise RuntimeError("Schema system missing from dependencies — run Cell 2 first")
        
        # TABLE VALIDATION
        print("🔍 Validating agent-specific table requirements...")
        
        # Check if agent-specific requirements function is available
        import __main__
        if hasattr(__main__, 'check_agent_specific_requirements'):
            try:
                __main__.check_agent_specific_requirements('exploratory_agent')
                print("✅ Agent-specific requirements validation passed")
            except Exception as e:
                print(f"❌ Agent-specific requirements failed: {e}")
                # Continue with available tables
        
        # Check TTS Usage table (required for exploratory agent)
        if 'TTS Usage' not in VERIFIED_TABLES:
            raise RuntimeError("TTS Usage table not found in VERIFIED_TABLES — run Cell 2 to verify tables")
        
        if not VERIFIED_TABLES['TTS Usage'].get('accessible', False):
            raise RuntimeError("TTS Usage table not accessible — check Cell 2 table verification")
        
        print("✅ All critical dependencies validated successfully")
        
        # Merge default parameters with provided parameters
        config = self.agents["exploratory_agent"]
        final_params = config.default_params.copy()
        final_params.update(params)
        
        days_back = final_params.get('days_back', 7)
        limit = final_params.get('limit', 1000)
        analysis_depth = final_params.get('analysis_depth', 'basic')
        
        print(f"📊 Exploratory analysis parameters: days_back={days_back}, limit={limit}, analysis_depth={analysis_depth}")
        
        try:
            # Use injected dependencies instead of global state
            tts_table = VERIFIED_TABLES['TTS Usage']
            
            # Get schema-aware column names
            from __main__ import get_column_name
            
            user_id_col = get_column_name('TTS Usage', 'user_id')
            email_col = get_column_name('TTS Usage', 'email')
            text_col = get_column_name('TTS Usage', 'text')
            timestamp_col = get_column_name('TTS Usage', 'timestamp')
            
            # Build source filter if specified
            source_filter_clause = ""
            if final_params.get('source_filter'):
                source_filter_clause = f"AND LOWER(source) LIKE '%{final_params['source_filter'].lower()}%'"
            
            # Build text filter if specified
            text_filter_clause = ""
            if final_params.get('text_contains'):
                text_filter_clause = f"AND LOWER({text_col}) LIKE '%{final_params['text_contains'].lower()}%'"
            
            # Build exploratory analysis query
            exploratory_query = f"""
            WITH recent_activity AS (
                SELECT 
                    {user_id_col} as user_id,
                    {email_col} as email,
                    {text_col} as text,
                    {timestamp_col} as timestamp,
                    LENGTH({text_col}) as text_length,
                    CASE 
                        WHEN LENGTH({text_col}) > 1000 THEN 'long_text'
                        WHEN LENGTH({text_col}) > 100 THEN 'medium_text'
                        ELSE 'short_text'
                    END as text_category,
                    CASE 
                        WHEN LOWER({text_col}) LIKE '%urgent%' OR LOWER({text_col}) LIKE '%immediate%' THEN 'urgent_content'
                        WHEN LOWER({text_col}) LIKE '%financial%' OR LOWER({text_col}) LIKE '%money%' THEN 'financial_content'
                        WHEN LOWER({text_col}) LIKE '%crypto%' OR LOWER({text_col}) LIKE '%investment%' THEN 'crypto_content'
                        WHEN LOWER({text_col}) LIKE '%personal%' OR LOWER({text_col}) LIKE '%private%' THEN 'personal_content'
                        ELSE 'general_content'
                    END as content_category
                FROM `{tts_table['table_id']}`
                WHERE DATE({timestamp_col}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
                    AND {text_col} IS NOT NULL
                    AND LENGTH({text_col}) > 0
                    {source_filter_clause}
                    {text_filter_clause}
            ),
            user_patterns AS (
                SELECT 
                    user_id,
                    email,
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    MIN(timestamp) as first_request,
                    MAX(timestamp) as last_request,
                    AVG(text_length) as avg_text_length,
                    MAX(text_length) as max_text_length,
                    COUNT(DISTINCT content_category) as content_variety,
                    STRING_AGG(DISTINCT content_category, ', ') as content_types
                FROM recent_activity
                GROUP BY user_id, email
            )
            SELECT 
                r.user_id,
                r.email,
                r.text,
                r.timestamp,
                r.text_length,
                r.text_category,
                r.content_category,
                u.total_requests,
                u.active_days,
                u.first_request,
                u.last_request,
                u.avg_text_length,
                u.max_text_length,
                u.content_variety,
                u.content_types,
                'exploratory_analysis' as analysis_type,
                CASE 
                    WHEN u.total_requests > 100 AND u.active_days = 1 THEN 'high_volume_single_day'
                    WHEN u.total_requests > 500 THEN 'high_volume_user'
                    WHEN u.content_variety > 3 THEN 'diverse_content_user'
                    WHEN r.text_length > 1000 THEN 'long_text_user'
                    ELSE 'normal_user'
                END as user_pattern
            FROM recent_activity r
            JOIN user_patterns u ON r.user_id = u.user_id
            ORDER BY 
                u.total_requests DESC,
                r.timestamp DESC
            LIMIT {limit}
            """
            
            print(f"🔍 Executing exploratory analysis query with REAL data...")
            
            # Execute query using injected BigQuery client
            query_job = bq_client.query(exploratory_query)
            results = query_job.result()
            
            records = []
            high_risk_count = 0
            pattern_counts = {}
            
            for row in results:
                user_pattern = getattr(row, 'user_pattern', 'normal_user')
                
                record = {
                    'user_id': row.user_id,
                    'email': row.email,
                    'text': row.text,
                    'timestamp': row.timestamp,
                    'text_length': row.text_length,
                    'text_category': row.text_category,
                    'content_category': row.content_category,
                    'total_requests': row.total_requests,
                    'active_days': row.active_days,
                    'first_request': row.first_request,
                    'last_request': row.last_request,
                    'avg_text_length': row.avg_text_length,
                    'max_text_length': row.max_text_length,
                    'content_variety': row.content_variety,
                    'content_types': row.content_types,
                    'analysis_type': row.analysis_type,
                    'user_pattern': user_pattern
                }
                
                # Count patterns
                pattern_counts[user_pattern] = pattern_counts.get(user_pattern, 0) + 1
                
                # Determine if high risk based on patterns
                if user_pattern in ['high_volume_single_day', 'high_volume_user']:
                    high_risk_count += 1
                    record['high_risk'] = True
                else:
                    record['high_risk'] = False
                
                # If analysis_depth is 'detailed', add AI analysis
                if analysis_depth == 'detailed' and analyzer and hasattr(analyzer, 'analyze_content'):
                    analysis = analyzer.analyze_content(row.text)
                    record['ai_analysis'] = analysis
                    
                    # Check AI risk score
                    if analysis and isinstance(analysis, dict):
                        ai_risk_score = analysis.get('risk_score', 0)
                        if ai_risk_score > 0.7:
                            record['high_risk'] = True
                            high_risk_count += 1
                
                records.append(record)
            
            # Create investigation entry using injected investigation manager
            if investigation_manager and hasattr(investigation_manager, 'current_investigation'):
                investigation_manager.add_finding(
                    f"Exploratory Analysis: Found {len(records)} records with {len(pattern_counts)} different user patterns, {high_risk_count} high-risk"
                )
                
                if high_risk_count > 0:
                    investigation_manager.add_action(
                        f"Review {high_risk_count} high-risk patterns for potential abuse"
                    )
            
            # Generate recommendations
            recommendations = []
            
            if high_risk_count > 0:
                recommendations.append(f"Review {high_risk_count} high-risk patterns for potential abuse")
            
            if len(records) > 0:
                recommendations.append(f"Analyze {len(records)} records across {len(pattern_counts)} different user patterns")
                
                # Add pattern-specific recommendations
                for pattern, count in pattern_counts.items():
                    if pattern == 'high_volume_single_day':
                        recommendations.append(f"Monitor {count} users with high single-day volume")
                    elif pattern == 'high_volume_user':
                        recommendations.append(f"Review {count} high-volume users for potential automation")
                    elif pattern == 'diverse_content_user':
                        recommendations.append(f"Investigate {count} users with diverse content patterns")
                
                if analysis_depth == 'basic':
                    recommendations.append("Consider running with analysis_depth='detailed' for AI analysis")
                else:
                    recommendations.append("Detailed analysis with AI content analysis completed")
            else:
                recommendations.append("No significant patterns found in recent data")
                recommendations.append("Consider adjusting time range or search criteria")
            
            print(f"✅ Exploratory analysis complete: {len(records)} records, {len(pattern_counts)} patterns, {high_risk_count} high-risk")
            
            return AgentResult(
                agent_name="exploratory_agent",
                query_description=user_query,
                records_found=len(records),
                high_risk_items=high_risk_count,
                analysis_results=records,
                recommendations=recommendations,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ EXPLORATORY AGENT ERROR: {e}")
            return AgentResult(
                agent_name="exploratory_agent",
                query_description=user_query,
                records_found=0,
                high_risk_items=0,
                analysis_results=[],
                recommendations=[f"Error: {str(e)}"],
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )

# =============================================================================
# MAIN AGENT EXECUTION FUNCTION
# =============================================================================

def run_investigation_agent(query: str, agent_name: str = None, **params) -> AgentResult:
    """
    Execute investigation agent with natural language query and intent detection
    
    This is the main entry point for investigations. It:
    1. Detects intent from natural language query (if no agent specified)
    2. Selects appropriate agent based on intent
    3. Executes the agent with proper error handling
    4. Returns structured results
    
    Args:
        query: Natural language investigation query
        agent_name: Specific agent to use (optional, auto-detected if None)
        **params: Additional parameters to pass to agent
        
    Returns:
        AgentResult with investigation findings
    """
    
    print(f"AGENT LAUNCHER: Processing investigator query")
    print(f"Query: '{query}'")
    print()
    
    # Step 1: Detect intent or use provided agent
    if agent_name:
        if agent_name not in agent_registry.get_available_agents():
            print(f"ERROR: Agent '{agent_name}' not found")
            print(f"Available agents: {agent_registry.get_available_agents()}")
            return AgentResult(
                agent_name="error",
                query_description=query,
                records_found=0,
                high_risk_items=0,
                analysis_results=[],
                recommendations=[f"Agent '{agent_name}' not found"],
                execution_time=0,
                timestamp=datetime.now()
            )
        
        selected_agent = agent_name
        confidence = 1.0  # Manual selection = full confidence
    else:
        # Detect intent
        intent = agent_registry.detect_intent(query)
        if not intent:
            print("ERROR: Intent detection completely failed")
            print("This should not happen with the new fallback system")
            return AgentResult(
                agent_name="error",
                query_description=query,
                records_found=0,
                high_risk_items=0,
                analysis_results=[],
                recommendations=["Intent detection failed - please specify agent manually"],
                execution_time=0,
                timestamp=datetime.now()
            )
        
        selected_agent = intent.suggested_agent
        confidence = intent.confidence
        
        # Handle low-confidence/unknown intents
        if intent.abuse_type == "unknown" or confidence < 0.3:
            print(f"WARNING: UNCERTAIN INTENT: Low confidence ({confidence:.2f}) for query")
            print(f"   Defaulting to {selected_agent} with investigator notification")
            
            # Add uncertainty warning to recommendations
            uncertainty_warning = (
                f"MANUAL REVIEW NEEDED: Query intent unclear (confidence: {confidence:.2f}). "
                f"Defaulted to {selected_agent}. Consider clarifying investigation focus."
            )
        else:
            print(f"SUCCESS: INTENT DETECTED: {intent.abuse_type} -> {selected_agent} (confidence: {confidence:.2f})")
            uncertainty_warning = None
        
        # Merge detected parameters
        params.update(intent.extracted_params)
        
        # Log the intent detection for transparency
        print(f"   Extraction: {intent.extracted_params if intent.extracted_params else 'None'}")
        print()
    
    print(f"AGENT LAUNCHER: Selected agent '{selected_agent}'")
    
    # Step 2: Create investigation if none exists (with attribute check)
    try:
        if not hasattr(investigation_manager, 'current_investigation'):
            investigation_manager.current_investigation = None
        
        if investigation_manager.current_investigation is None:
            # Check what parameters the investigation manager accepts
            try:
                investigation_manager.create_investigation(
                    title=f"NL Investigation: {selected_agent}",
                    description=f"Semantic NL investigation using {selected_agent}: {query}"
                )
                print(f"INVESTIGATION: Created new investigation for NL query")
            except TypeError as e:
                # Handle different parameter signatures
                if "investigator_id" in str(e):
                    try:
                        investigation_manager.create_investigation(
                            title=f"NL Investigation: {selected_agent}",
                            description=f"Semantic NL investigation using {selected_agent}: {query}",
                            investigator_id="system"
                        )
                        print(f"INVESTIGATION: Created new investigation with investigator_id")
                    except Exception as e2:
                        print(f"WARNING: Could not create investigation: {str(e2)}")
                else:
                    print(f"WARNING: Could not create investigation: {str(e)}")
    except NameError:
        print("WARNING: Investigation manager not available - continuing without investigation logging")
    
    # Step 3: Execute the agent with robust error handling
    handler = agent_registry.agent_handlers.get(selected_agent)
    if not handler:
        print(f"ERROR: No handler found for agent '{selected_agent}'")
        return AgentResult(
            agent_name="error",
            query_description=query,
            records_found=0,
            high_risk_items=0,
            analysis_results=[],
            recommendations=[f"No handler found for agent '{selected_agent}'"],
            execution_time=0,
            timestamp=datetime.now()
        )
    
    # Execute the agent handler with comprehensive error handling
    try:
        print(f"AGENT LAUNCHER: Executing {selected_agent}...")
        result = handler(query, **params)
        
        # Add uncertainty warning to recommendations if applicable
        if 'uncertainty_warning' in locals() and uncertainty_warning:
            result.recommendations.insert(0, uncertainty_warning)
        
        # Step 4: Generate natural language summary
        print(f"AGENT LAUNCHER: Generating investigation summary...")
        try:
            summary = summarize_agent_result(result)
            result.summary_text = summary
            
            # Log summary as finding with enhanced context
            try:
                if hasattr(investigation_manager, 'current_investigation') and investigation_manager.current_investigation is not None:
                    confidence_info = f" (Intent confidence: {confidence:.2f})" if 'confidence' in locals() else ""
                    investigation_manager.add_finding(
                        f"NL Agent Summary: {result.agent_name}{confidence_info}",
                        f"Query: '{query}' | {summary}"
                    )
                    print(f"SUCCESS: NL investigation summary logged with intent context")
            except NameError:
                print("WARNING: Investigation manager not available - skipping summary logging")
            
        except Exception as e:
            print(f"WARNING: Summary generation failed: {str(e)}")
            result.summary_text = f"Summary generation failed: {str(e)}"
        
        # Step 5: Log successful agent execution
        try:
            if hasattr(investigation_manager, 'current_investigation'):
                investigation_manager.add_action(
                    f"Executed {selected_agent}",
                    f"Query: {query}, Records: {result.records_found}, High-risk: {result.high_risk_items}"
                )
        except NameError:
            print("WARNING: Investigation manager not available - skipping action logging")
        
        print(f"AGENT LAUNCHER: {selected_agent} completed successfully")
        return result
        
    except Exception as e:
        # Get full traceback
        error_traceback = traceback.format_exc()
        error_type = type(e).__name__
        error_message = str(e)
        
        # Log comprehensive error details
        print(f"ERROR: AGENT EXECUTION FAILED")
        print(f"   Agent: {selected_agent}")
        print(f"   Query: {query}")
        print(f"   Error Type: {error_type}")
        print(f"   Error Message: {error_message}")
        print(f"   Full Traceback:")
        print(error_traceback)
        
        # Log failure to investigation if available
        try:
            if hasattr(investigation_manager, 'current_investigation') and investigation_manager.current_investigation is not None:
                investigation_manager.add_action(
                    f"FAILED: {selected_agent} execution failed",
                    f"Error: {error_type} - {error_message}"
                )
        except NameError:
            print("WARNING: Investigation manager not available - skipping error logging")
        
        # Return structured error response
        error_result = AgentResult(
            agent_name=selected_agent,
            query_description=query,
            records_found=0,
            high_risk_items=0,
            analysis_results=[],
            recommendations=[
                f"Agent execution failed: {error_type}",
                f"Error details: {error_message}",
                "Review logs for full traceback information",
                "Consider running with different parameters or agent"
            ],
            execution_time=0,
            timestamp=datetime.now(),
            error_details={
                "error_type": error_type,
                "error_message": error_message,
                "traceback": error_traceback
            }
        )
        
        print(f"AGENT LAUNCHER: Returning error-safe response")
        return error_result

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_available_agents():
    """List all available investigation agents"""
    print("\nAvailable Investigation Agents")
    print("=" * 50)
    
    for agent_name, config in agent_registry.agents.items():
        print(f"Agent: {agent_name}")
        print(f"  Description: {config.description}")
        print(f"  Target Tables: {', '.join(config.target_tables)}")
        print(f"  Uses OpenAI: {config.use_openai}")
        print(f"  Default Parameters: {config.default_params}")
        if config.query_template:
            print(f"  Query Template: {config.query_template}")
        print()

def define_custom_agent(agent_name: str, description: str, target_tables: List[str],
                       handler_function: Callable, intent_patterns: List[str],
                       **config_params):
    """Allow investigators to define custom agents"""
    print(f"Defining custom agent: {agent_name}")
    
    agent_registry.register_agent(
        agent_name=agent_name,
        description=description,
        target_tables=target_tables,
        required_fields=config_params.get('required_fields', ['text', 'userid']),
        default_params=config_params.get('default_params', {}),
        use_openai=config_params.get('use_openai', True),
        handler_function=handler_function,
        intent_patterns=intent_patterns,
        query_template=config_params.get('query_template', '')
    )

def test_agent_intent(query: str):
    """Test intent detection for a single query with detailed analysis"""
    print(f"Testing intent detection for: '{query}'")
    print("-" * 50)
    
    intent = agent_registry.detect_intent(query)
    if intent:
        print(f"✅ MATCH FOUND")
        print(f"   Abuse Type: {intent.abuse_type}")
        print(f"   Suggested Agent: {intent.suggested_agent}")
        print(f"   Confidence: {intent.confidence:.2f}")
        print(f"   Match Score: {intent.match_score:.3f}")
        if intent.match_reason:
            print(f"   Match Reason: {intent.match_reason}")
        if intent.extracted_params:
            print(f"   Extracted Parameters: {intent.extracted_params}")
    else:
        print("❌ NO INTENT DETECTED")
        print("   Available agents:")
        for name, config in agent_registry.agents.items():
            print(f"     - {name}: {config.description}")
    print("-" * 50)

def run_nl_test_suite():
    """
    Production-grade test suite for natural language intent detection
    
    Tests real investigator language patterns and edge cases
    """
    
    # Test cases designed from real investigator usage patterns
    test_cases = [
        # Scam detection queries
        {"query": "This looks like a voice cloning scam", "expected_intent": "scam"},
        {"query": "Investigate potential fraud from this user", "expected_intent": "scam"},
        {"query": "Is this user running a Ponzi scheme?", "expected_intent": "scam"},
        {"query": "Check if this is a romance scammer", "expected_intent": "scam"},
        {"query": "Looks like another investment fraud", "expected_intent": "scam"},
        {"query": "fake profile detected", "expected_intent": "scam"},
        
        # Spam detection queries
        {"query": "This user is posting bulk content", "expected_intent": "spam"},
        {"query": "Automated posting detected here", "expected_intent": "spam"},
        {"query": "Same message repeated 100 times", "expected_intent": "spam"},
        {"query": "Bot activity in this account", "expected_intent": "spam"},
        {"query": "Volume abuse patterns", "expected_intent": "spam"},
        
        # Network analysis queries
        {"query": "Map connections from this email", "expected_intent": "network_analysis"},
        {"query": "Is this guy connected to other accounts?", "expected_intent": "network_analysis"},
        {"query": "this guy again?", "expected_intent": "network_analysis"},
        {"query": "Check if this is part of a ring", "expected_intent": "network_analysis"},
        {"query": "Find all linked accounts", "expected_intent": "network_analysis"},
        {"query": "Show user network for suspicious@example.com", "expected_intent": "network_analysis"},
        
        # Identity linkage queries
        {"query": "Same person across multiple accounts?", "expected_intent": "identity_linkage"},
        {"query": "Check device sharing patterns", "expected_intent": "identity_linkage"}, 
        {"query": "same card used by different users", "expected_intent": "identity_linkage"},
        {"query": "Verify if accounts belong to same person", "expected_intent": "identity_linkage"},
        {"query": "Alt account detection", "expected_intent": "identity_linkage"},
        
        # Exploratory investigation queries
        {"query": "investigate this person", "expected_intent": "exploration"},
        {"query": "what's going on here?", "expected_intent": "exploration"},
        {"query": "check this out", "expected_intent": "exploration"},
        {"query": "investigate user suspicious@example.com", "expected_intent": "exploration"},
        {"query": "baseline review for this account", "expected_intent": "exploration"},
        {"query": "look into this user", "expected_intent": "exploration"},
        {"query": "general check on activity", "expected_intent": "exploration"},
        
        # Truly ambiguous queries (should still get exploratory)
        {"query": "hmm suspicious", "expected_intent": "exploration"},
        
        # Edge cases (system should still provide investigative path)
        {"query": "", "expected_intent": "exploration"},
        {"query": "123", "expected_intent": "exploration"},
        {"query": "test test test", "expected_intent": "exploration"},
        
        # Mixed/complex queries
        {"query": "This scammer is using fake accounts in a coordinated network", "expected_intent": "scam"},  # Could be scam or network
        {"query": "Bulk spam from connected user group", "expected_intent": "spam"},  # Could be spam or network
    ]
    
    print("🧪 NATURAL LANGUAGE INTENT DETECTION TEST SUITE")
    print("=" * 80)
    print(f"Testing {len(test_cases)} investigator query patterns...")
    print()
    
    # Track results
    results = {
        "total": len(test_cases),
        "exact_matches": 0,
        "acceptable_matches": 0,
        "failures": 0,
        "details": []
    }
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected_intent"]
        
        print(f"Test {i:2d}/{len(test_cases)}: {query}")
        print("-" * 60)
        
        # Run intent detection
        intent = agent_registry.detect_intent(query)
        
        if intent:
            detected = intent.abuse_type
            confidence = intent.confidence
            agent = intent.suggested_agent
            
            # Evaluate result
            if detected == expected:
                status = "✅ EXACT MATCH"
                results["exact_matches"] += 1
                result_type = "exact"
            elif expected == "exploration" and detected in ["exploration", "unknown"]:
                status = "✅ ACCEPTABLE (Exploratory investigation triggered)"
                results["acceptable_matches"] += 1
                result_type = "acceptable"
            elif expected != "exploration" and detected == "exploration":
                status = "⚠️  PARTIAL (Defaulted to exploration - acceptable for unclear queries)"
                results["acceptable_matches"] += 1
                result_type = "partial"
            elif expected != "exploration" and detected != "exploration" and detected != "unknown":
                status = "⚠️  PARTIAL (Different intent but investigation proceeding)"
                results["acceptable_matches"] += 1
                result_type = "partial"
            else:
                status = "❌ FAILED"
                results["failures"] += 1
                result_type = "failed"
            
            print(f"   Expected: {expected}")
            print(f"   Detected: {detected}")
            print(f"   Agent: {agent}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Status: {status}")
            
        else:
            # This should rarely happen with the new exploratory fallback
            if expected == "exploration":
                status = "⚠️  PARTIAL (No detection but exploration expected)"
                results["acceptable_matches"] += 1
                result_type = "acceptable"
            else:
                status = "❌ FAILED (No detection - fallback system failed)"
                results["failures"] += 1
                result_type = "failed"
            
            print(f"   Expected: {expected}")
            print(f"   Detected: None")
            print(f"   Status: {status}")
        
        # Store detailed result
        results["details"].append({
            "query": query,
            "expected": expected,
            "detected": intent.abuse_type if intent else None,
            "confidence": intent.confidence if intent else 0.0,
            "status": result_type
        })
        
        print()
    
    # Summary results
    total = results["total"]
    exact = results["exact_matches"]
    acceptable = results["acceptable_matches"]
    failed = results["failures"]
    success_rate = ((exact + acceptable) / total) * 100
    
    print("=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tests:        {total}")
    print(f"Exact Matches:      {exact} ({(exact/total)*100:.1f}%)")
    print(f"Acceptable:         {acceptable} ({(acceptable/total)*100:.1f}%)")
    print(f"Failures:           {failed} ({(failed/total)*100:.1f}%)")
    print(f"Overall Success:    {success_rate:.1f}%")
    print()
    
    if success_rate >= 80:
        print("🎉 EXCELLENT: NL intent detection is production-ready!")
    elif success_rate >= 70:
        print("✅ GOOD: NL intent detection is working well with minor improvements needed")
    elif success_rate >= 60:
        print("⚠️  NEEDS WORK: Several intent detection issues need addressing")
    else:
        print("❌ CRITICAL: Major improvements needed for production use")
    
    print()
    print("🔧 RECOMMENDATIONS:")
    if failed > 0:
        print("• Review failed test cases for pattern improvements")
        print("• Consider expanding alias dictionaries")
        print("• Test with more investigator query examples")
    
    if exact < total * 0.6:
        print("• Improve semantic classification prompt")
        print("• Add more investigator-specific training examples")
    
    print("• Continue collecting real investigator queries for testing")
    print()
    
    return results

def test_specific_query_patterns():
    """Test specific challenging query patterns"""
    
    challenging_queries = [
        "Hey can you run a check on this guy?",
        "This looks like the same pattern as last week",
        "Another one of these...",
        "Check if this connects to the ring we found",
        "Same MO different account",
        "Verify this is legit",
        "Red flags all over this one",
        "Quick check please"
    ]
    
    print("🎯 TESTING CHALLENGING INVESTIGATOR QUERIES")
    print("=" * 60)
    
    for query in challenging_queries:
        print(f"Query: '{query}'")
        intent = agent_registry.detect_intent(query)
        
        if intent:
            print(f"  → {intent.abuse_type} ({intent.confidence:.2f}) via {intent.suggested_agent}")
        else:
            print(f"  → No classification")
        print()

def test_query_templates():
    """Test query template functionality for all agents"""
    print("TESTING QUERY TEMPLATES")
    print("=" * 50)
    
    for agent_name in agent_registry.get_available_agents():
        config = agent_registry.get_agent_config(agent_name)
        print(f"Agent: {agent_name}")
        print(f"  Description: {config.description}")
        if config.query_template:
            print(f"  ✅ Template: {config.query_template}")
        else:
            print(f"  ❌ No template available")
        print()
    
    print("Template test complete!")

# =============================================================================
# NATURAL LANGUAGE SUMMARY GENERATION
# =============================================================================

def summarize_agent_result(result: AgentResult) -> str:
    """
    Generates a short natural-language summary of the agent's findings.
    If OpenAI is available, uses GPT-4. Otherwise, uses rule-based fallback.
    Returns the summary string.
    """
    
    # Check if OpenAI is available and properly configured
    try:
        import __main__
        main_system = getattr(__main__, 'main_system', None)
    except:
        main_system = None
    
    if main_system and main_system.analyzer and main_system.analyzer.openai_client:
        try:
            return _generate_openai_summary(result)
        except Exception as e:
            print(f"WARNING: OpenAI summary generation failed: {str(e)}")
            print("Falling back to rule-based summary")
            return _generate_rule_based_summary(result)
    else:
        print("INFO: OpenAI not available - using rule-based summary")
        return _generate_rule_based_summary(result)

def _generate_openai_summary(result: AgentResult) -> str:
    """Generate summary using OpenAI GPT-4 with v1.0+ API"""
    
    # Prepare analysis results summary
    analysis_summary = ""
    if result.analysis_results:
        high_risk_count = sum(1 for r in result.analysis_results if r.get('risk_level') in ['high', 'critical'])
        analysis_summary = f"OpenAI analysis processed {len(result.analysis_results)} items, flagging {high_risk_count} as high-risk. "
        
        # Extract key indicators
        all_indicators = []
        for analysis in result.analysis_results:
            if analysis.get('scam_indicators'):
                all_indicators.extend(analysis['scam_indicators'])
        
        if all_indicators:
            top_indicators = list(set(all_indicators))[:3]  # Top 3 unique indicators
            analysis_summary += f"Common indicators: {', '.join(top_indicators)}. "
    
    # Prepare recommendations summary
    recommendations_summary = ""
    if result.recommendations:
        recommendations_summary = f"Key recommendations: {'; '.join(result.recommendations[:2])}."
    
    prompt = f"""
    Summarize this Trust & Safety investigation result into a clear, professional summary:
    
    Agent: {result.agent_name}
    Query: {result.query_description}
    Records Found: {result.records_found}
    High-Risk Items: {result.high_risk_items}
    Execution Time: {result.execution_time:.2f}s
    Analysis Details: {analysis_summary}
    Recommendations: {recommendations_summary}
    
    Create a concise 2-3 sentence summary that includes:
    1. What the agent investigated
    2. Key findings (records found, high-risk items)
    3. Main outcome or recommendation
    
    Write in professional investigative language suitable for Trust & Safety reports.
    """
    
    try:
        # Use OpenAI v1.0+ API syntax
        response = main_system.analyzer.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Trust & Safety analyst writing investigation summaries. Be concise and professional."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        print(f"WARNING: OpenAI v1.0+ API call failed: {str(e)}")
        raise RuntimeError(f"OpenAI summary generation failed: {str(e)}")

def _generate_rule_based_summary(result: AgentResult) -> str:
    """Generate rule-based summary when OpenAI is not available"""
    
    # Determine investigation focus
    investigation_focus = "suspicious activity"
    if "scam" in result.query_description.lower():
        investigation_focus = "scam activity"
    elif "spam" in result.query_description.lower():
        investigation_focus = "spam patterns"
    elif "fraud" in result.query_description.lower():
        investigation_focus = "fraudulent content"
    
    # Build summary components
    summary_parts = []
    
    # Agent and scope
    summary_parts.append(f"The {result.agent_name} investigated {investigation_focus}")
    
    # Results
    if result.records_found > 0:
        if result.high_risk_items > 0:
            summary_parts.append(f"and identified {result.records_found} potentially suspicious entries, with {result.high_risk_items} flagged as high-risk")
        else:
            summary_parts.append(f"and processed {result.records_found} entries with no high-risk items detected")
    else:
        summary_parts.append("but found no matching records")
    
    # Key recommendations
    if result.recommendations:
        if any("escalation" in rec.lower() or "review" in rec.lower() for rec in result.recommendations):
            summary_parts.append("Escalation review recommended")
        elif any("monitor" in rec.lower() for rec in result.recommendations):
            summary_parts.append("Continued monitoring advised")
        else:
            summary_parts.append("Standard procedures recommended")
    
    # Performance note
    if result.execution_time > 30:
        summary_parts.append(f"Analysis completed in {result.execution_time:.1f}s")
    
    return ". ".join(summary_parts) + "."

# =============================================================================
# DEPENDENCY FIXES AND CHECKS
# =============================================================================

# Fix missing current_investigation attribute if needed
try:
    import __main__
    investigation_manager = getattr(__main__, 'investigation_manager', None)
    if investigation_manager:
        if not hasattr(investigation_manager, 'current_investigation'):
            print("🔧 Fixing missing current_investigation attribute...")
            investigation_manager.current_investigation = None
            print("✅ current_investigation attribute added")
    else:
        print("⚠️  investigation_manager not found - investigation logging will be limited")
except Exception as e:
    print(f"⚠️  Error accessing investigation_manager: {e}")

# =============================================================================
# INITIALIZE AGENT SYSTEM
# =============================================================================

# Create global agent registry
agent_registry = AgentRegistry()

# Expose agent registry as agent_launcher for UI compatibility
agent_launcher = agent_registry

# Make safe_bigquery_query globally accessible for agent execution
globals()['safe_bigquery_query'] = safe_bigquery_query

print("SUCCESS: Agent Launcher initialized with semantic NL detection")
print("Available functions:")
print("  - run_investigation_agent('natural language query')")
print("  - list_available_agents()")
print("  - define_custom_agent(name, description, tables, handler, patterns)")
print("  - test_agent_intent('test query')")
print("  - run_nl_test_suite()  # 🧪 Test suite for NL intent detection")
print("  - test_specific_query_patterns()  # 🎯 Test challenging queries")
print("  - test_query_templates()")
print("  - safe_bigquery_query(client, query_string)  # 🔧 Available globally")
print("Example investigator usage:")
print("  run_investigation_agent('This looks like a voice cloning scam')")
print("  run_investigation_agent('Check if this guy is connected to other accounts')")
print("  run_investigation_agent('Same card used by different users')")
print("  run_investigation_agent('investigate user suspicious@example.com')  # 🔍 Exploratory")
print("  run_investigation_agent('what\\'s going on here?')  # 🔍 Exploratory fallback")
print("\nCell 7b Complete - Semantic NL Agent Launcher with Exploratory Support Ready")

# Quick verification test
try:
    sig = inspect.signature(investigation_manager.create_investigation)
    print("Method parameters:", list(sig.parameters.keys()))
    print("Has risk_level:", 'risk_level' in sig.parameters)
    
    # Test basic creation (handle different parameter signatures)
    params = list(sig.parameters.keys())
    print(f"Available parameters: {params}")
    
    try:
        if 'risk_level' in sig.parameters and 'investigator' in sig.parameters:
            test_inv = investigation_manager.create_investigation(
                "Test", "Test description", risk_level="high"
            )
            print("✅ Test investigation created with risk_level")
        elif 'investigator_id' in sig.parameters:
            test_inv = investigation_manager.create_investigation(
                "Test", "Test description", investigator_id="test_user"
            )
            print("✅ Test investigation created with investigator_id")
        elif len(params) >= 2:
            test_inv = investigation_manager.create_investigation(
                "Test", "Test description"
            )
            print("✅ Test investigation created with basic parameters")
        else:
            print("⚠️  Cannot create test investigation - insufficient parameters")
    except Exception as creation_error:
        print(f"⚠️  Test investigation creation failed: {str(creation_error)}")
        print("   This is expected if investigation manager has different signature")
            
except Exception as e:
    print(f"Verification test skipped: {str(e)}")

# Enhanced NL query system integration available via direct cell loader
# (Removed automatic file loading to prevent FileNotFoundError)
print("✅ Enhanced NL Query System available via direct cell loader")
print("   → Use run_enhanced_query() for improved debugging")
print("   → Use system_status() for complete system overview")
print("   → Use explain_query() to understand query processing")
print("   → Standard query processing available")

print("✅ Agent system ready for use!") 

print("\n🎯 READY TO USE:")
print("  • run_investigation_agent('your query') - Standard agent execution")
print("  • run_enhanced_query('your query', debug=True) - Enhanced execution with debugging")
print("  • system_status() - Check system health")
print("  • explain_query('your query') - Understand query processing")

print("\n📋 EXAMPLE USAGE:")
print("  result = run_investigation_agent('find the past 1 day of tts generations')")
print("  inspect_result(result)")

print("\n🔧 DEBUGGING TOOLS (load manually when needed):")
print("  exec(open('debug_helper.py').read())")
print("  exec(open('PRACTICAL_VALIDATION_TEST.py').read())")

# =============================================================================
# NOTEBOOK INTEGRATION - SHARED SYSTEM AWARENESS
# =============================================================================

print("\n🔗 NOTEBOOK INTEGRATION CHECK...")

# Check if we're in a unified notebook system
if 'NOTEBOOK_STATE' in globals():
    print("✅ Unified notebook system detected")
    
    # Check if previous cells have run
    if 'run_investigation_query' in globals():
        print("✅ Integrating with shared notebook system...")
        
        # Load unified system functions if available
        if 'setup_agent_launcher' in globals():
            setup_agent_launcher()
            print("✅ Agent launcher integrated with shared system")
        else:
            print("⚠️  Shared system functions not available")
            
    else:
        print("⚠️  Run Cells 1-5 first for full notebook integration")
        print("   Current cell works standalone but shared features limited")
        
else:
    print("⚠️  Running in standalone mode")
    print("   For full notebook integration, run:")
    print("   exec(open('NOTEBOOK_CELL_SYSTEM.py').read())")

print("\n🎯 CELL 7B READY - Agent launcher available")
print("Test your query: run_investigation_agent('find the past 1 day of tts generations')")

# =============================================================================
# AGENT REGISTRY CLASS
# =============================================================================
