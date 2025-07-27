# ==============================================================
# CELL 07c: Runtime Manager
# Purpose: Agent runtime executor and thread-safe status tracking
# Dependencies: Agent registry, system core, investigation manager
# ==============================================================

# @title Cell 7c: Agent Runtime Manager — Parallel Execution & Status Tracking
# Thread-based agent execution manager for Trust & Safety investigations

import threading
import time
import uuid
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import pandas as pd
import json
import sys
import os

# Ensure current directory is in path
if '.' not in sys.path:
    sys.path.insert(0, '.')
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# =============================================================================
# AGENT CONTEXT AND STATUS TRACKING
# =============================================================================

@dataclass
class AgentContext:
    agent_id: str
    agent_name: str
    query_description: str
    investigation_id: Optional[str]
    target_email: Optional[str]
    intent_detected: str
    start_time: datetime
    thread_id: Optional[int] = None
    should_stop: threading.Event = None
    last_log_message: str = ""
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None
    status: str = "running"

@dataclass
class AgentStatus:
    agent_id: str
    status: str  # running, completed, canceled, failed
    agent_name: str
    investigation_id: Optional[str]
    intent_detected: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    last_log_message: str
    error_message: Optional[str]
    records_found: int = 0
    high_risk_items: int = 0

# =============================================================================
# AGENT RUNTIME MANAGER
# =============================================================================

class AgentRuntimeManager:
    def __init__(self, max_concurrent_agents: int = 5):
        self.max_concurrent_agents = max_concurrent_agents
        self.active_agents: Dict[str, AgentContext] = {}
        self.agent_history: List[AgentStatus] = []
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_agents)
        self.status_lock = threading.Lock()
        self._agent_futures: Dict[str, Future] = {}
        print(f"[SYSTEM] Agent Runtime Manager initialized (max concurrent: {max_concurrent_agents})")

    def launch_agent_sync(self, query: str, agent_name: Optional[str] = None, **params) -> str:
        """
        Launch an agent synchronously and return its ID
        """
        try:
            from cell_7b_agent_launcher import agent_registry, run_investigation_agent
            if not agent_name:
                intent = agent_registry.detect_intent(query)
                if intent:
                    agent_name = intent.suggested_agent
                    intent_detected = intent.abuse_type
                    params.update(intent.extracted_params)
                else:
                    agent_name = "exploratory_agent"
                    intent_detected = "exploration"
            
            # Generate unique agent ID
            agent_id = str(uuid.uuid4())[:8]
            
            # Create agent context
            agent_context = AgentContext(
                agent_id=agent_id,
                agent_name=agent_name,
                query=query,
                params=params,
                start_time=datetime.now(),
                intent_detected=intent_detected if 'intent_detected' in locals() else "direct",
                status="running"
            )
            
            # Add to active agents
            self.active_agents[agent_id] = agent_context
            
            # Start agent execution
            print(f"[AGENT] Starting {agent_name} (ID: {agent_id})")
            
            # Run the agent
            result = run_investigation_agent(query, agent_name, agent_context=agent_context, **params)
            
            # Update context with result
            agent_context.result = result
            agent_context.status = "completed"
            agent_context.end_time = datetime.now()
            
            print(f"[AGENT] Completed {agent_name} (ID: {agent_id})")
            
            return agent_id
            
        except Exception as e:
            # Update context with error
            if 'agent_context' in locals():
                agent_context.status = "failed"
                agent_context.error = str(e)
                agent_context.end_time = datetime.now()
            
            print(f"[ERROR] Agent launch failed: {e}")
            return None

    def launch_agent_async(self, query: str, agent_name: Optional[str] = None, **params) -> str:
        """
        Launch an agent asynchronously and return its ID
        
        DEPRECATED: Use launch_agent_sync() instead for proper synchronous execution
        """
        print("WARNING: launch_agent_async() is deprecated. Use launch_agent_sync() instead.")
        return self.launch_agent_sync(query, agent_name, **params)

    def launch_agent(self,
                     agent_handler: Callable,
                     query: str,
                     agent_name: str = "unknown",
                     investigation_id: Optional[str] = None,
                     intent_detected: str = "unknown",
                     **params) -> str:
        """
        Core agent launcher: runs in thread pool
        """
        agent_id = f"agent_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        ctx = AgentContext(
            agent_id=agent_id,
            agent_name=agent_name,
            query_description=query,
            investigation_id=investigation_id,
            target_email=None,
            intent_detected=intent_detected,
            start_time=datetime.now(),
            should_stop=threading.Event(),
            last_log_message="launched"
        )
        with self.status_lock:
            self.active_agents[agent_id] = ctx
        future = self.executor.submit(self._execute_agent_wrapper, ctx, agent_handler, query, **params)
        self._agent_futures[agent_id] = future
        print(f"[LAUNCH] Agent launched: {agent_name} (ID: {agent_id})")
        return agent_id

    def _execute_agent_wrapper(self, ctx: AgentContext, handler: Callable, query: str, **params):
        ctx.thread_id = threading.get_ident()
        ctx.last_log_message = "started"
        try:
            if ctx.should_stop.is_set():
                return None
            result = handler(query, ctx, **params)
            ctx.result = result
            ctx.end_time = datetime.now()
            duration = (ctx.end_time - ctx.start_time).total_seconds()
            status = AgentStatus(
                agent_id=ctx.agent_id,
                status="completed",
                agent_name=ctx.agent_name,
                investigation_id=ctx.investigation_id,
                intent_detected=ctx.intent_detected,
                start_time=ctx.start_time,
                end_time=ctx.end_time,
                duration_seconds=duration,
                last_log_message="completed",
                error_message=None,
                records_found=getattr(result, 'records_found', 0),
                high_risk_items=getattr(result, 'high_risk_items', 0)
            )
            with self.status_lock:
                self.agent_history.append(status)
                del self.active_agents[ctx.agent_id]
            print(f"[SUCCESS] AGENT {ctx.agent_id}: completed")
            return result
        except Exception as e:
            ctx.end_time = datetime.now()
            err = str(e)
            ctx.error_message = err
            status = AgentStatus(
                agent_id=ctx.agent_id,
                status="failed",
                agent_name=ctx.agent_name,
                investigation_id=ctx.investigation_id,
                intent_detected=ctx.intent_detected,
                start_time=ctx.start_time,
                end_time=ctx.end_time,
                duration_seconds=(ctx.end_time - ctx.start_time).total_seconds(),
                last_log_message="failed",
                error_message=err
            )
            with self.status_lock:
                self.agent_history.append(status)
                if ctx.agent_id in self.active_agents:
                    del self.active_agents[ctx.agent_id]
            print(f"[ERROR] AGENT {ctx.agent_id}: {err}")
            return None

    def stop_agent(self, agent_id: str) -> bool:
        """
        Stop a running agent
        """
        with self.status_lock:
            if agent_id not in self.active_agents:
                print(f"[ERROR] Agent {agent_id} not found or already stopped")
                return False
            ctx = self.active_agents[agent_id]
            ctx.should_stop.set()
            print(f"[STOP] Stop signal sent to agent {agent_id}")
            future = self._agent_futures.get(agent_id)
            if future and future.cancel():
                print(f"[STOP] Agent {agent_id} canceled before execution")
            return True

    def get_agent_result(self, agent_id: str) -> Optional[Any]:
        if agent_id not in self._agent_futures:
            print(f"[ERROR] Agent {agent_id} not found")
            return None
        future = self._agent_futures[agent_id]
        if future.done():
            try:
                return future.result()
            except Exception as e:
                print(f"[ERROR] Agent {agent_id} failed: {str(e)}")
        else:
            print(f"[INFO] Agent {agent_id} still running")
        return None

    def show_agent_status(self):
        df = pd.DataFrame([vars(status) for status in self.agent_history[-10:]])
        print("\n[STATUS] Agent Status Dashboard")
        print(df.to_string(index=False))

# create global instance
agent_runtime_manager = AgentRuntimeManager()
runtime_manager = agent_runtime_manager
print(f"SUCCESS: Runtime Manager globally available as 'runtime_manager' with {len(agent_runtime_manager.active_agents)} active agents")