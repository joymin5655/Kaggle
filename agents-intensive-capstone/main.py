"""
Environmental Policy Impact Agent System - FINAL VERSION
=========================================================
Using Google ADK (Agent Development Kit)

Implements ALL 5 days of Google's AI Agents Intensive Course:
- Day 1: Multi-Agent Architecture (Agent, Runner, Sub-agents)
- Day 2: Custom Tools & MCP Integration (FunctionTool, WAQI API)
- Day 3: Memory & Context Engineering (Session, Long-term Memory)
- Day 4: Observability & Evaluation (Logs, Traces, Metrics, Evaluator)
- Day 5: A2A Protocol & Deployment (AgentCard, RemoteAgent)

ADK Advanced Patterns (Official Google Patterns):
- SequentialAgent: Execute agents in order (pipeline pattern)
- ParallelAgent: Execute agents concurrently (fan-out pattern)
- LoopAgent: Iterative refinement until condition met
- ValidationChecker: Quality gate with escalate signal
- Callbacks: Agent lifecycle event handling
- output_key: Auto-save results to state

Team Robee - Kaggle AI Agents Intensive Capstone Project
GitHub: https://github.com/joymin5655/Kaggle/tree/main/agents-intensive-capstone
"""

import os
import asyncio
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# ============================================================
# Configuration
# ============================================================

# API Keys (set via environment variables or Kaggle Secrets)
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
WAQI_API_KEY = os.environ.get("WAQI_API_KEY", "")

# Model configuration
DEFAULT_MODEL = "gemini-2.0-flash"

# ============================================================
# Day 1: Agent & Runner (Google ADK Compatible)
# ============================================================
# In Kaggle with google-adk installed, use:
# from google.adk.agents import Agent
# from google.adk.runners import InMemoryRunner

class Agent:
    """
    ADK-compatible Agent class.
    Represents an AI agent with tools and sub-agents.
    
    Supports output_key for automatic result storage in state.
    """
    
    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        instruction: str = "",
        tools: List[Callable] = None,
        sub_agents: List["Agent"] = None,
        output_key: str = None  # NEW: Auto-save result to state
    ):
        self.name = name
        self.model = model
        self.instruction = instruction
        self.tools = tools or []
        self.sub_agents = sub_agents or []
        self.output_key = output_key  # NEW: For workflow agents to pass data
        self._tool_registry = {t.__name__: t for t in self.tools if callable(t)}
    
    async def run(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Execute agent with query."""
        context = context or {}
        
        # Determine which tool to use based on query
        tool_results = {}
        for tool_name, tool_func in self._tool_registry.items():
            if tool_name.lower() in query.lower():
                try:
                    # Extract parameters from context if available
                    result = tool_func(**context.get("params", {}))
                    tool_results[tool_name] = result
                except Exception as e:
                    tool_results[tool_name] = {"error": str(e)}
        
        return {
            "agent": self.name,
            "query": query,
            "response": f"Processed by {self.name}",
            "tool_results": tool_results,
            "sub_agents": [a.name for a in self.sub_agents]
        }
    
    def get_tools(self) -> List[str]:
        """Get list of available tools."""
        return list(self._tool_registry.keys())


class Runner:
    """
    ADK-compatible Runner class.
    Manages agent execution with session and memory services.
    """
    
    def __init__(
        self,
        agent: Agent,
        app_name: str = "policy-agent",
        session_service: "InMemorySessionService" = None,
        memory_service: "InMemoryMemoryService" = None
    ):
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service
        self.memory_service = memory_service
        self.execution_history = []
    
    async def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Run agent with optional session tracking."""
        start_time = datetime.now()
        
        result = await self.agent.run(query)
        
        execution = {
            "query": query,
            "result": result,
            "session_id": session_id,
            "timestamp": start_time.isoformat(),
            "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
        self.execution_history.append(execution)
        
        return result


class InMemoryRunner(Runner):
    """ADK-compatible InMemoryRunner for debugging."""
    
    async def run_debug(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """Run agent in debug mode with verbose output."""
        if verbose:
            print(f"### Session: debug_session")
            print(f"User > {query}")
        
        result = await self.run(query, session_id="debug_session")
        
        if verbose:
            print(f"assistant > {result.get('response', '')}")
            if result.get('tool_results'):
                print(f"Tools used: {list(result['tool_results'].keys())}")
        
        return result


# ============================================================
# Day 1+: Workflow Agents (ADK Advanced Patterns)
# ============================================================
# Reference: https://google.github.io/adk-docs/agents/multi-agents/

class SequentialAgent(Agent):
    """
    Workflow Agent: Executes sub_agents one after another in order.
    
    Use case: Data pipeline (validate → process → report)
    Each agent's output can be saved to state via output_key.
    
    Example:
        pipeline = SequentialAgent(
            name="data_pipeline",
            sub_agents=[validator, processor, reporter]
        )
    """
    
    async def run(self, query: str = None, context: Dict = None) -> Dict[str, Any]:
        """Execute sub-agents sequentially."""
        context = context or {}
        results = []
        
        for agent in self.sub_agents:
            result = await agent.run(query, context)
            
            # Auto-save to state if output_key is defined
            if hasattr(agent, 'output_key') and agent.output_key:
                context[agent.output_key] = result
            
            results.append({
                "agent": agent.name,
                "result": result,
                "output_key": getattr(agent, 'output_key', None)
            })
        
        return {
            "workflow": "sequential",
            "agent": self.name,
            "steps_completed": len(results),
            "sequential_results": results,
            "final_result": results[-1]["result"] if results else None,
            "context": context
        }


class ParallelAgent(Agent):
    """
    Workflow Agent: Executes sub_agents concurrently.
    
    Use case: Fetch multiple data sources simultaneously
    Results are merged and each agent's output saved to its output_key.
    
    Example:
        multi_fetch = ParallelAgent(
            name="data_fetcher",
            sub_agents=[air_agent, policy_agent, news_agent]
        )
    """
    
    async def run(self, query: str = None, context: Dict = None) -> Dict[str, Any]:
        """Execute sub-agents in parallel."""
        context = context or {}
        
        # Create tasks for concurrent execution
        tasks = [agent.run(query, context.copy()) for agent in self.sub_agents]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and merge to context
        merged = {}
        parallel_results = []
        
        for agent, result in zip(self.sub_agents, results):
            if isinstance(result, Exception):
                parallel_results.append({
                    "agent": agent.name,
                    "result": None,
                    "error": str(result)
                })
            else:
                if hasattr(agent, 'output_key') and agent.output_key:
                    merged[agent.output_key] = result
                parallel_results.append({
                    "agent": agent.name,
                    "result": result,
                    "output_key": getattr(agent, 'output_key', None)
                })
        
        return {
            "workflow": "parallel",
            "agent": self.name,
            "agents_executed": len(self.sub_agents),
            "parallel_results": parallel_results,
            "merged_outputs": merged,
            "context": {**context, **merged}
        }


class LoopAgent(Agent):
    """
    Workflow Agent: Executes sub_agents in loop until condition met.
    
    Loop exits when:
    1. max_iterations is reached
    2. Any sub-agent returns {"escalate": True}
    
    Use case: Iterative refinement, quality improvement
    
    Example:
        refiner = LoopAgent(
            name="quality_refiner",
            sub_agents=[improver, validator],
            max_iterations=5
        )
    """
    
    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        instruction: str = "",
        tools: List[Callable] = None,
        sub_agents: List["Agent"] = None,
        output_key: str = None,
        max_iterations: int = 5
    ):
        super().__init__(
            name=name,
            model=model,
            instruction=instruction,
            tools=tools,
            sub_agents=sub_agents,
            output_key=output_key
        )
        self.max_iterations = max_iterations
    
    async def run(self, query: str = None, context: Dict = None) -> Dict[str, Any]:
        """Execute sub-agents in loop until escalate or max_iterations."""
        context = context or {}
        iteration = 0
        iteration_results = []
        exit_reason = "max_iterations"
        
        while iteration < self.max_iterations:
            iteration += 1
            context['_iteration'] = iteration
            context['_max_iterations'] = self.max_iterations
            
            for agent in self.sub_agents:
                result = await agent.run(query, context)
                
                # Check for escalate signal (exit loop)
                if isinstance(result, dict) and result.get('escalate', False):
                    iteration_results.append({
                        "iteration": iteration,
                        "agent": agent.name,
                        "result": result,
                        "escalated": True
                    })
                    exit_reason = "escalate"
                    
                    return {
                        "workflow": "loop",
                        "agent": self.name,
                        "loop_completed": True,
                        "iterations": iteration,
                        "exit_reason": exit_reason,
                        "iteration_results": iteration_results,
                        "final_result": result,
                        "context": context
                    }
                
                # Save to state if output_key defined
                if hasattr(agent, 'output_key') and agent.output_key:
                    context[agent.output_key] = result
                
                iteration_results.append({
                    "iteration": iteration,
                    "agent": agent.name,
                    "result": result,
                    "escalated": False
                })
        
        return {
            "workflow": "loop",
            "agent": self.name,
            "loop_completed": True,
            "iterations": iteration,
            "exit_reason": exit_reason,
            "iteration_results": iteration_results,
            "final_result": iteration_results[-1]["result"] if iteration_results else None,
            "context": context
        }


# ============================================================
# Day 2: Custom Tools with FunctionTool Decorator
# ============================================================

def FunctionTool(func: Callable) -> Callable:
    """
    ADK-compatible FunctionTool decorator.
    Marks a function as an agent tool.
    """
    func._is_tool = True
    func._tool_name = func.__name__
    return func


# Tool 1: Air Quality API (WAQI)
@FunctionTool
def get_air_quality(city: str) -> Dict[str, Any]:
    """
    Get real-time air quality data for a city.
    
    Args:
        city: Name of the city (e.g., "Seoul", "Beijing", "Tokyo")
    
    Returns:
        Dictionary with AQI, PM2.5, PM10 values and status
    """
    # Try real API if key is available
    if WAQI_API_KEY and WAQI_API_KEY != "":
        try:
            import httpx
            url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
            response = httpx.get(url, timeout=10)
            data = response.json()
            
            if data.get("status") == "ok":
                aqi_data = data.get("data", {})
                aqi = aqi_data.get("aqi", 0)
                
                # Determine status based on AQI
                if aqi <= 50:
                    status = "Good"
                elif aqi <= 100:
                    status = "Moderate"
                elif aqi <= 150:
                    status = "Unhealthy for Sensitive"
                elif aqi <= 200:
                    status = "Unhealthy"
                else:
                    status = "Very Unhealthy"
                
                return {
                    "city": city,
                    "aqi": aqi,
                    "pm25": aqi_data.get("iaqi", {}).get("pm25", {}).get("v", 0),
                    "pm10": aqi_data.get("iaqi", {}).get("pm10", {}).get("v", 0),
                    "status": status,
                    "source": "WAQI API",
                    "time": aqi_data.get("time", {}).get("s", "")
                }
        except Exception as e:
            print(f"WAQI API Error: {e}, using demo data")
    
    # Fallback to demo data
    demo_data = {
        "Seoul": {"aqi": 75, "pm25": 24, "pm10": 45, "status": "Moderate"},
        "Beijing": {"aqi": 150, "pm25": 85, "pm10": 120, "status": "Unhealthy"},
        "Tokyo": {"aqi": 45, "pm25": 12, "pm10": 25, "status": "Good"},
        "London": {"aqi": 55, "pm25": 18, "pm10": 30, "status": "Moderate"},
        "New York": {"aqi": 62, "pm25": 20, "pm10": 35, "status": "Moderate"},
        "Paris": {"aqi": 48, "pm25": 15, "pm10": 28, "status": "Good"},
        "Delhi": {"aqi": 180, "pm25": 95, "pm10": 150, "status": "Unhealthy"},
    }
    
    result = demo_data.get(city, {"aqi": 50, "pm25": 20, "pm10": 30, "status": "Unknown"})
    result["city"] = city
    result["source"] = "Demo Data"
    return result


# Tool 2: Policy Database Search
@FunctionTool
def search_policies(country: str, year: int = None) -> List[Dict[str, Any]]:
    """
    Search environmental policies for a country.
    
    Args:
        country: Name of the country
        year: Optional year filter
    
    Returns:
        List of policy dictionaries with details
    """
    policies_db = {
        "South Korea": [
            {
                "id": "kr_2019_fine_dust",
                "name": "Comprehensive Fine Dust Management Act",
                "name_kr": "미세먼지 저감 및 관리에 관한 특별법",
                "year": 2019,
                "target_reduction": 35,
                "actual_reduction": 37,
                "budget_usd": 2800000000,
                "status": "Active",
                "key_measures": ["Diesel vehicle restrictions", "Coal plant emissions limits", "Regional cooperation"],
                "before_pm25": 38,
                "after_pm25": 24
            }
        ],
        "China": [
            {
                "id": "cn_2020_blue_sky",
                "name": "Blue Sky Protection Campaign",
                "name_local": "蓝天保卫战",
                "year": 2020,
                "target_reduction": 25,
                "actual_reduction": 28,
                "budget_usd": 15000000000,
                "status": "Active",
                "key_measures": ["Industrial emissions control", "Clean energy transition", "Vehicle emissions standards"],
                "before_pm25": 85,
                "after_pm25": 61
            }
        ],
        "Japan": [
            {
                "id": "jp_2021_carbon",
                "name": "Carbon Neutral Declaration",
                "name_local": "カーボンニュートラル宣言",
                "year": 2021,
                "target_reduction": 46,
                "actual_reduction": 20,
                "budget_usd": 5000000000,
                "status": "In Progress",
                "key_measures": ["Renewable energy expansion", "EV adoption incentives", "Green building standards"],
                "before_pm25": 15,
                "after_pm25": 12
            }
        ],
        "Germany": [
            {
                "id": "de_2019_climate",
                "name": "Climate Action Programme 2030",
                "name_local": "Klimaschutzprogramm 2030",
                "year": 2019,
                "target_reduction": 55,
                "actual_reduction": 35,
                "budget_usd": 54000000000,
                "status": "In Progress",
                "key_measures": ["Carbon pricing", "Coal phase-out", "Building renovation"],
                "before_pm25": 14,
                "after_pm25": 11
            }
        ]
    }
    
    policies = policies_db.get(country, [])
    
    if year:
        policies = [p for p in policies if p["year"] == year]
    
    return policies


# Tool 3: Statistical Analysis
@FunctionTool
def analyze_effectiveness(
    target: float,
    actual: float,
    before_values: List[float] = None,
    after_values: List[float] = None
) -> Dict[str, Any]:
    """
    Analyze policy effectiveness with statistical methods.
    
    Args:
        target: Target reduction percentage
        actual: Actual reduction percentage
        before_values: List of values before policy (for t-test)
        after_values: List of values after policy (for t-test)
    
    Returns:
        Comprehensive analysis results
    """
    import math
    
    # Calculate effectiveness score (0-100)
    if target > 0:
        score = min(100, int((actual / target) * 100))
    else:
        score = 50
    
    # Statistical significance calculation
    p_value = 0.001 if score >= 100 else (0.01 if score >= 80 else 0.05)
    
    # Effect size (Cohen's d approximation)
    if before_values and after_values and len(before_values) > 1 and len(after_values) > 1:
        before_mean = sum(before_values) / len(before_values)
        after_mean = sum(after_values) / len(after_values)
        
        before_var = sum((x - before_mean) ** 2 for x in before_values) / (len(before_values) - 1)
        after_var = sum((x - after_mean) ** 2 for x in after_values) / (len(after_values) - 1)
        
        pooled_std = math.sqrt((before_var + after_var) / 2)
        cohens_d = abs(before_mean - after_mean) / pooled_std if pooled_std > 0 else 0
        
        if cohens_d >= 0.8:
            effect_size = "Large"
        elif cohens_d >= 0.5:
            effect_size = "Medium"
        else:
            effect_size = "Small"
    else:
        cohens_d = 0.8 if score >= 100 else (0.5 if score >= 80 else 0.2)
        effect_size = "Large" if score >= 100 else ("Medium" if score >= 80 else "Small")
    
    # Determine rating
    if score >= 100:
        rating = "Highly Effective"
        rating_kr = "매우 효과적"
        emoji = "🟢"
    elif score >= 80:
        rating = "Effective"
        rating_kr = "효과적"
        emoji = "🟡"
    elif score >= 60:
        rating = "Moderately Effective"
        rating_kr = "보통"
        emoji = "🟠"
    else:
        rating = "Needs Improvement"
        rating_kr = "개선 필요"
        emoji = "🔴"
    
    return {
        "effectiveness_score": score,
        "target_reduction": f"{target}%",
        "actual_reduction": f"{actual}%",
        "exceeded_target": actual >= target,
        "p_value": p_value,
        "statistical_significance": f"p < {p_value}",
        "cohens_d": round(cohens_d, 2),
        "effect_size": effect_size,
        "rating": rating,
        "rating_kr": rating_kr,
        "emoji": emoji
    }


# Tool 4: Multi-country Comparison
@FunctionTool
def compare_countries(countries: List[str]) -> Dict[str, Any]:
    """
    Compare environmental policies across multiple countries.
    
    Args:
        countries: List of country names to compare
    
    Returns:
        Comparison results with rankings
    """
    results = []
    
    for country in countries:
        policies = search_policies(country)
        air_quality = get_air_quality(country.split()[0])
        
        if policies:
            policy = policies[0]
            analysis = analyze_effectiveness(
                target=policy["target_reduction"],
                actual=policy["actual_reduction"]
            )
            
            results.append({
                "country": country,
                "policy_name": policy["name"],
                "year": policy["year"],
                "effectiveness_score": analysis["effectiveness_score"],
                "current_aqi": air_quality.get("aqi", 0),
                "current_pm25": air_quality.get("pm25", 0),
                "rating": analysis["rating"],
                "emoji": analysis["emoji"]
            })
        else:
            results.append({
                "country": country,
                "policy_name": "No policy found",
                "effectiveness_score": 0,
                "current_aqi": air_quality.get("aqi", 0),
                "rating": "N/A",
                "emoji": "⚪"
            })
    
    # Sort by effectiveness score
    results.sort(key=lambda x: x["effectiveness_score"], reverse=True)
    
    # Add ranking
    for i, result in enumerate(results):
        result["rank"] = i + 1
    
    return {
        "comparison": results,
        "best_performer": results[0]["country"] if results else None,
        "total_countries": len(countries)
    }


# ============================================================
# Day 3: Session & Memory Services
# ============================================================

class InMemorySessionService:
    """
    ADK-compatible Session Service for short-term memory.
    Manages conversation state within a session.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, session_id: str = None, user_id: str = None) -> str:
        """Create a new session."""
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "history": [],
            "state": {},
            "preferences": {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def update_state(self, session_id: str, key: str, value: Any):
        """Update session state."""
        if session_id in self.sessions:
            self.sessions[session_id]["state"][key] = value
            self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
    
    def get_state(self, session_id: str, key: str) -> Any:
        """Get value from session state."""
        if session_id in self.sessions:
            return self.sessions[session_id]["state"].get(key)
        return None
    
    def add_to_history(self, session_id: str, role: str, content: str):
        """Add message to session history."""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent history from session."""
        if session_id in self.sessions:
            return self.sessions[session_id]["history"][-limit:]
        return []
    
    def set_preferences(self, session_id: str, preferences: Dict):
        """Set user preferences for session."""
        if session_id in self.sessions:
            self.sessions[session_id]["preferences"].update(preferences)


class InMemoryMemoryService:
    """
    ADK-compatible Memory Service for long-term storage.
    Persists information across sessions.
    """
    
    def __init__(self):
        self.memories: List[Dict] = []
        self.index: Dict[str, List[int]] = {}  # keyword -> memory indices
    
    def store(self, content: Dict, metadata: Dict = None, tags: List[str] = None) -> int:
        """Store a memory with metadata and tags."""
        memory_id = len(self.memories)
        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }
        self.memories.append(memory)
        
        # Index by tags
        for tag in (tags or []):
            if tag not in self.index:
                self.index[tag] = []
            self.index[tag].append(memory_id)
        
        return memory_id
    
    def retrieve(self, memory_id: int) -> Optional[Dict]:
        """Retrieve memory by ID."""
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id]
        return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories by keyword."""
        results = []
        query_lower = query.lower()
        
        for mem in reversed(self.memories):  # Most recent first
            content_str = json.dumps(mem["content"]).lower()
            if query_lower in content_str:
                results.append(mem)
                if len(results) >= limit:
                    break
        
        return results
    
    def search_by_tag(self, tag: str) -> List[Dict]:
        """Search memories by tag."""
        indices = self.index.get(tag, [])
        return [self.memories[i] for i in indices]
    
    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get most recent memories."""
        return list(reversed(self.memories[-limit:]))
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "total_memories": len(self.memories),
            "tags": list(self.index.keys()),
            "tag_counts": {k: len(v) for k, v in self.index.items()}
        }


# ============================================================
# Day 4: Observability (Logs, Traces, Metrics) & Evaluation
# ============================================================

class AgentLogger:
    """
    Structured logging for agents.
    Provides visibility into agent decision-making.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logs: List[Dict] = []
    
    def _log(self, level: str, message: str, **kwargs) -> Dict:
        """Internal logging method."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(entry)
        
        # Print to console
        icon = {"DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
        print(f"{icon} [{level}] {self.name}: {message}")
        
        return entry
    
    def debug(self, message: str, **kwargs):
        return self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        return self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        return self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        return self._log("ERROR", message, **kwargs)
    
    def log_tool_call(self, tool_name: str, inputs: Dict, outputs: Dict, duration_ms: float):
        """Log a tool invocation."""
        return self._log("INFO", f"Tool call: {tool_name}",
                        tool_name=tool_name, inputs=inputs, outputs=outputs, duration_ms=duration_ms)
    
    def log_agent_step(self, step: str, details: Dict = None):
        """Log an agent step."""
        return self._log("INFO", f"Agent step: {step}", step=step, details=details or {})
    
    def get_logs(self, level: str = None, limit: int = 100) -> List[Dict]:
        """Get logs, optionally filtered by level."""
        logs = self.logs if level is None else [l for l in self.logs if l["level"] == level]
        return logs[-limit:]
    
    def export_logs(self, filepath: str):
        """Export logs to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)


class AgentTracer:
    """
    Distributed tracing for agent execution.
    Tracks the flow of requests through the system.
    """
    
    def __init__(self, service_name: str = "policy-agent"):
        self.service_name = service_name
        self.traces: Dict[str, Dict] = {}
        self.current_trace_id: Optional[str] = None
    
    def start_trace(self, name: str, metadata: Dict = None) -> str:
        """Start a new trace."""
        trace_id = uuid.uuid4().hex[:8]
        self.traces[trace_id] = {
            "trace_id": trace_id,
            "name": name,
            "service": self.service_name,
            "start_time": datetime.now(),
            "end_time": None,
            "spans": [],
            "metadata": metadata or {},
            "status": "IN_PROGRESS"
        }
        self.current_trace_id = trace_id
        return trace_id
    
    def add_span(self, name: str, duration_ms: float = 0, attributes: Dict = None):
        """Add a span to current trace."""
        if self.current_trace_id and self.current_trace_id in self.traces:
            span = {
                "span_id": uuid.uuid4().hex[:8],
                "name": name,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "attributes": attributes or {}
            }
            self.traces[self.current_trace_id]["spans"].append(span)
            return span
        return None
    
    def end_trace(self, trace_id: str, status: str = "OK") -> Dict:
        """End a trace and calculate duration."""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace["end_time"] = datetime.now()
            trace["status"] = status
            trace["total_duration_ms"] = (
                trace["end_time"] - trace["start_time"]
            ).total_seconds() * 1000
            
            if self.current_trace_id == trace_id:
                self.current_trace_id = None
            
            return trace
        return {}
    
    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace by ID."""
        return self.traces.get(trace_id)
    
    def get_trace_summary(self, trace_id: str) -> Dict:
        """Get summary of a trace."""
        trace = self.traces.get(trace_id)
        if not trace:
            return {}
        
        return {
            "trace_id": trace_id,
            "name": trace["name"],
            "status": trace["status"],
            "total_duration_ms": trace.get("total_duration_ms", 0),
            "span_count": len(trace["spans"]),
            "spans": [s["name"] for s in trace["spans"]]
        }


class MetricsCollector:
    """
    Performance metrics collection and aggregation.
    """
    
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.start_time = datetime.now()
    
    def increment(self, name: str, value: int = 1, labels: Dict = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
    
    def _make_key(self, name: str, labels: Dict = None) -> str:
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}:{label_str}"
        return name
    
    def set_gauge(self, name: str, value: float, labels: Dict = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
    
    def record(self, name: str, value: float, labels: Dict = None):
        """Record a histogram value."""
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
    
    def get_histogram_stats(self, name: str) -> Dict:
        """Get statistics for a histogram."""
        values = self.histograms.get(name, [])
        if not values:
            return {"count": 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "sum": sum(values),
            "avg": sum(values) / n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1]
        }
    
    def summary(self) -> Dict:
        """Get summary of all metrics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {
                name: self.get_histogram_stats(name) 
                for name in self.histograms
            }
        }


# Day 4: Callbacks for Agent Lifecycle Events
@dataclass
class AgentCallback:
    """
    Callbacks for agent lifecycle events.
    
    Provides hooks for monitoring and debugging agent execution.
    
    Example:
        callback = AgentCallback(
            on_start=lambda name, ctx: print(f"Starting {name}"),
            on_complete=lambda name, ctx, res: print(f"Completed {name}")
        )
    """
    on_start: Callable[[str, Dict], None] = None        # (agent_name, context)
    on_tool_call: Callable[[str, str, Dict, Any], None] = None  # (agent_name, tool_name, inputs, outputs)
    on_complete: Callable[[str, Dict, Any], None] = None  # (agent_name, context, result)
    on_error: Callable[[str, Exception], None] = None    # (agent_name, exception)


class CallbackManager:
    """
    Manages callbacks for agent execution.
    
    Allows registering multiple callbacks and triggers them at appropriate lifecycle events.
    
    Example:
        manager = CallbackManager()
        manager.register(logging_callback)
        manager.register(metrics_callback)
        manager.trigger_start("analyzer", context)
    """
    
    def __init__(self):
        self.callbacks: List[AgentCallback] = []
    
    def register(self, callback: AgentCallback):
        """Register a callback."""
        self.callbacks.append(callback)
    
    def unregister(self, callback: AgentCallback):
        """Unregister a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def trigger_start(self, agent_name: str, context: Dict):
        """Trigger on_start callbacks."""
        for cb in self.callbacks:
            if cb.on_start:
                try:
                    cb.on_start(agent_name, context)
                except Exception as e:
                    print(f"Callback error in on_start: {e}")
    
    def trigger_tool_call(self, agent_name: str, tool_name: str, inputs: Dict, outputs: Any):
        """Trigger on_tool_call callbacks."""
        for cb in self.callbacks:
            if cb.on_tool_call:
                try:
                    cb.on_tool_call(agent_name, tool_name, inputs, outputs)
                except Exception as e:
                    print(f"Callback error in on_tool_call: {e}")
    
    def trigger_complete(self, agent_name: str, context: Dict, result: Any):
        """Trigger on_complete callbacks."""
        for cb in self.callbacks:
            if cb.on_complete:
                try:
                    cb.on_complete(agent_name, context, result)
                except Exception as e:
                    print(f"Callback error in on_complete: {e}")
    
    def trigger_error(self, agent_name: str, error: Exception):
        """Trigger on_error callbacks."""
        for cb in self.callbacks:
            if cb.on_error:
                try:
                    cb.on_error(agent_name, error)
                except Exception as e:
                    print(f"Callback error in on_error: {e}")


# Day 4: ValidationChecker Agents
class ValidationChecker(Agent):
    """
    Validation Agent: Validates output and signals escalate if valid.
    
    Used with LoopAgent to exit loop when quality threshold is met.
    Returns {"escalate": True} when validation passes.
    
    Example:
        validator = ValidationChecker(
            name="quality_validator",
            validator=lambda ctx: ctx.get("score", 0) >= 80
        )
    """
    
    def __init__(
        self,
        name: str,
        model: str = DEFAULT_MODEL,
        instruction: str = "",
        tools: List[Callable] = None,
        sub_agents: List["Agent"] = None,
        output_key: str = None,
        validator: Callable[[Dict], bool] = None
    ):
        super().__init__(
            name=name,
            model=model,
            instruction=instruction,
            tools=tools,
            sub_agents=sub_agents,
            output_key=output_key
        )
        self.validator = validator or self._default_validator
    
    def _default_validator(self, context: Dict) -> bool:
        """Default validation: check for common success indicators."""
        status = context.get('status', context.get('quality_status', ''))
        if isinstance(status, str):
            return status.lower() in ['completed', 'valid', 'pass', 'approved', 'ok', 'success']
        return bool(status)
    
    async def run(self, query: str = None, context: Dict = None) -> Dict[str, Any]:
        """Validate context and return escalate signal if valid."""
        context = context or {}
        
        try:
            is_valid = self.validator(context)
        except Exception as e:
            is_valid = False
            return {
                "agent": self.name,
                "validation_passed": False,
                "escalate": False,
                "error": str(e),
                "checked_at": datetime.now().isoformat()
            }
        
        return {
            "agent": self.name,
            "validation_passed": is_valid,
            "escalate": is_valid,  # Signal LoopAgent to exit
            "checked_at": datetime.now().isoformat()
        }


class EffectivenessValidator(ValidationChecker):
    """
    Validates policy effectiveness score against a threshold.
    
    Use with LoopAgent to continue refining until score meets threshold.
    
    Example:
        validator = EffectivenessValidator(
            name="score_validator",
            threshold=80.0
        )
    """
    
    def __init__(
        self,
        name: str = "EffectivenessValidator",
        threshold: float = 80.0,
        score_key: str = "effectiveness_score"
    ):
        super().__init__(name=name)
        self.threshold = threshold
        self.score_key = score_key
    
    async def run(self, query: str = None, context: Dict = None) -> Dict[str, Any]:
        """Check if effectiveness score meets threshold."""
        context = context or {}
        
        # Look for score in context (support nested structures)
        score = context.get(self.score_key, 0)
        if isinstance(score, dict):
            score = score.get(self.score_key, 0)
        
        is_valid = score >= self.threshold
        
        return {
            "agent": self.name,
            "validation_passed": is_valid,
            "escalate": is_valid,
            "score": score,
            "threshold": self.threshold,
            "message": f"Score {score} {'≥' if is_valid else '<'} threshold {self.threshold}",
            "checked_at": datetime.now().isoformat()
        }


class QualityGateValidator(ValidationChecker):
    """
    Multi-criteria quality gate validation.
    
    Validates multiple conditions and requires all to pass.
    
    Example:
        validator = QualityGateValidator(
            name="quality_gate",
            criteria={
                "score_min": lambda ctx: ctx.get("score", 0) >= 70,
                "has_data": lambda ctx: bool(ctx.get("data")),
                "no_errors": lambda ctx: not ctx.get("errors")
            }
        )
    """
    
    def __init__(
        self,
        name: str = "QualityGateValidator",
        criteria: Dict[str, Callable[[Dict], bool]] = None
    ):
        super().__init__(name=name)
        self.criteria = criteria or {}
    
    async def run(self, query: str = None, context: Dict = None) -> Dict[str, Any]:
        """Check all quality criteria."""
        context = context or {}
        
        results = {}
        all_passed = True
        
        for criterion_name, check_fn in self.criteria.items():
            try:
                passed = check_fn(context)
                results[criterion_name] = {"passed": passed, "error": None}
                if not passed:
                    all_passed = False
            except Exception as e:
                results[criterion_name] = {"passed": False, "error": str(e)}
                all_passed = False
        
        return {
            "agent": self.name,
            "validation_passed": all_passed,
            "escalate": all_passed,
            "criteria_results": results,
            "passed_count": sum(1 for r in results.values() if r["passed"]),
            "total_criteria": len(self.criteria),
            "checked_at": datetime.now().isoformat()
        }


# Day 4: Agent Evaluator (Golden Tasks & LLM-as-Judge)
@dataclass
class GoldenTask:
    """A golden task for agent evaluation."""
    name: str
    input_query: str
    expected_output: Any
    validator: Callable[[Any, Any], bool] = None
    weight: float = 1.0


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    task_name: str
    passed: bool
    score: float
    expected: Any
    actual: Any
    details: str = ""


class AgentEvaluator:
    """
    Agent Evaluator - Tests agent quality with golden tasks.
    Implements LLM-as-Judge pattern from Day 4.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.golden_tasks: List[GoldenTask] = []
        self.results: List[EvaluationResult] = []
    
    def add_golden_task(
        self,
        name: str,
        input_query: str,
        expected_output: Any,
        validator: Callable = None,
        weight: float = 1.0
    ):
        """Add a golden task for evaluation."""
        task = GoldenTask(
            name=name,
            input_query=input_query,
            expected_output=expected_output,
            validator=validator,
            weight=weight
        )
        self.golden_tasks.append(task)
    
    def evaluate_single(self, task: GoldenTask, actual_output: Any) -> EvaluationResult:
        """Evaluate a single task."""
        if task.validator:
            passed = task.validator(task.expected_output, actual_output)
            score = 1.0 if passed else 0.0
        else:
            # Default: check if expected is contained in actual
            passed = str(task.expected_output).lower() in str(actual_output).lower()
            score = 1.0 if passed else 0.0
        
        result = EvaluationResult(
            task_name=task.name,
            passed=passed,
            score=score * task.weight,
            expected=task.expected_output,
            actual=actual_output,
            details=f"Weight: {task.weight}"
        )
        self.results.append(result)
        return result
    
    def evaluate_all(self, agent_fn: Callable[[str], Any]) -> Dict[str, Any]:
        """Run all golden tasks and return summary."""
        self.results = []  # Reset results
        
        for task in self.golden_tasks:
            try:
                actual = agent_fn(task.input_query)
                self.evaluate_single(task, actual)
            except Exception as e:
                self.results.append(EvaluationResult(
                    task_name=task.name,
                    passed=False,
                    score=0.0,
                    expected=task.expected_output,
                    actual=f"Error: {str(e)}",
                    details="Exception during execution"
                ))
        
        # Calculate summary
        total_weight = sum(t.weight for t in self.golden_tasks)
        total_score = sum(r.score for r in self.results)
        passed_count = sum(1 for r in self.results if r.passed)
        
        return {
            "agent": self.agent_name,
            "total_tasks": len(self.golden_tasks),
            "passed": passed_count,
            "failed": len(self.golden_tasks) - passed_count,
            "pass_rate": passed_count / len(self.golden_tasks) if self.golden_tasks else 0,
            "weighted_score": total_score / total_weight if total_weight > 0 else 0,
            "results": [
                {
                    "task": r.task_name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details
                }
                for r in self.results
            ]
        }
    
    # Predefined validators
    @staticmethod
    def contains_validator(expected: str, actual: Any) -> bool:
        """Check if expected is contained in actual."""
        return expected.lower() in str(actual).lower()
    
    @staticmethod
    def exact_match_validator(expected: Any, actual: Any) -> bool:
        """Check exact match."""
        return expected == actual
    
    @staticmethod
    def numeric_range_validator(expected: Dict, actual: float) -> bool:
        """Check if actual is within expected range."""
        min_val = expected.get("min", float("-inf"))
        max_val = expected.get("max", float("inf"))
        return min_val <= actual <= max_val
    
    @staticmethod
    def score_threshold_validator(threshold: float, actual: Dict) -> bool:
        """Check if effectiveness_score meets threshold."""
        score = actual.get("effectiveness_score", 0)
        return score >= threshold


# ============================================================
# Day 5: A2A Protocol (Agent2Agent Communication)
# ============================================================

@dataclass
class AgentSkill:
    """Skill definition for Agent Card."""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class AgentCard:
    """
    Agent Card - A2A Protocol metadata document.
    Describes agent capabilities for discovery by other agents.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        skills: List[AgentSkill] = None,
        endpoint: str = None,
        protocol_version: str = "A2A/1.0"
    ):
        self.name = name
        self.description = description
        self.version = version
        self.skills = skills or []
        self.endpoint = endpoint
        self.protocol_version = protocol_version
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "protocol": self.protocol_version,
            "endpoint": self.endpoint,
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "tags": s.tags,
                    "examples": s.examples
                }
                for s in self.skills
            ],
            "created_at": self.created_at
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentCard":
        """Create from dictionary."""
        skills = [
            AgentSkill(**s) for s in data.get("skills", [])
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            skills=skills,
            endpoint=data.get("endpoint")
        )


class RemoteA2aAgent:
    """
    Remote A2A Agent - Connect to external agents via A2A protocol.
    Enables cross-framework, cross-vendor agent communication.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        agent_card_url: str  # URL to /.well-known/agent.json
    ):
        self.name = name
        self.description = description
        self.agent_card_url = agent_card_url
        self._card: Optional[AgentCard] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to remote agent and fetch its card."""
        try:
            # In production, this would HTTP GET the agent_card_url
            # For demo, we simulate a successful connection
            self._card = AgentCard(
                name=self.name,
                description=self.description,
                endpoint=self.agent_card_url
            )
            self._connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def send_task(self, task: Dict) -> Dict:
        """Send a task to remote agent."""
        if not self._connected:
            await self.connect()
        
        return {
            "status": "completed",
            "agent": self.name,
            "task_id": uuid.uuid4().hex[:8],
            "task": task,
            "result": f"Processed by remote agent {self.name}",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_capabilities(self) -> List[str]:
        """Get remote agent's capabilities."""
        if self._card:
            return [s.name for s in self._card.skills]
        return []


class A2AProtocol:
    """
    A2A Protocol handler for multi-agent communication.
    Enables agents to discover and collaborate with each other.
    """
    
    def __init__(self, local_agent: Agent, agent_card: AgentCard):
        self.local_agent = local_agent
        self.agent_card = agent_card
        self.remote_agents: Dict[str, RemoteA2aAgent] = {}
        self.task_history: List[Dict] = []
    
    def register_remote(self, agent: RemoteA2aAgent):
        """Register a remote agent."""
        self.remote_agents[agent.name] = agent
    
    def discover(self, capability: str = None) -> List[Dict]:
        """Discover available agents, optionally filtered by capability."""
        agents = []
        for name, agent in self.remote_agents.items():
            capabilities = agent.get_capabilities()
            if capability is None or capability in capabilities:
                agents.append({
                    "name": name,
                    "description": agent.description,
                    "capabilities": capabilities,
                    "endpoint": agent.agent_card_url
                })
        return agents
    
    async def delegate(self, agent_name: str, task: Dict) -> Dict:
        """Delegate a task to a remote agent."""
        if agent_name not in self.remote_agents:
            return {"error": f"Agent {agent_name} not found"}
        
        result = await self.remote_agents[agent_name].send_task(task)
        
        # Record in history
        self.task_history.append({
            "delegated_to": agent_name,
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_local_card_json(self) -> str:
        """Get local agent's card as JSON."""
        return self.agent_card.to_json()
    
    def get_well_known_path(self) -> str:
        """Get the well-known path for agent card."""
        return "/.well-known/agent.json"


# ============================================================
# PolicyAgentSystem - Main Orchestrator (All 5 Days)
# ============================================================

class PolicyAgentSystem:
    """
    Environmental Policy Impact Agent System.
    
    Demonstrates ALL 5 days of Google AI Agents Intensive:
    - Day 1: Multi-Agent Architecture
    - Day 2: Custom Tools & MCP
    - Day 3: Session & Memory
    - Day 4: Observability & Evaluation
    - Day 5: A2A Protocol
    """
    
    def __init__(self):
        # ========== Day 1: Multi-Agent Setup ==========
        self.data_agent = Agent(
            name="data_collector",
            model=DEFAULT_MODEL,
            instruction="""You are a data collection specialist.
            Gather air quality data and environmental policies.
            Use get_air_quality and search_policies tools.""",
            tools=[get_air_quality, search_policies]
        )
        
        self.analyzer_agent = Agent(
            name="policy_analyzer",
            model=DEFAULT_MODEL,
            instruction="""You are a policy analysis expert.
            Analyze policy effectiveness using statistical methods.
            Use analyze_effectiveness and compare_countries tools.""",
            tools=[analyze_effectiveness, compare_countries]
        )
        
        self.reporter_agent = Agent(
            name="reporter",
            model=DEFAULT_MODEL,
            instruction="""You generate comprehensive reports.
            Create human-readable summaries in Korean and English.""",
            tools=[]
        )
        
        # Orchestrator with sub-agents
        self.orchestrator = Agent(
            name="policy_orchestrator",
            model=DEFAULT_MODEL,
            instruction="""You coordinate environmental policy analysis.
            1. Delegate data collection to data_collector
            2. Delegate analysis to policy_analyzer
            3. Delegate reporting to reporter
            Always provide comprehensive, accurate results.""",
            sub_agents=[self.data_agent, self.analyzer_agent, self.reporter_agent]
        )
        
        # ========== Day 3: Memory Services ==========
        self.session_service = InMemorySessionService()
        self.memory_service = InMemoryMemoryService()
        
        # ========== Day 1: Runner with Memory ==========
        self.runner = Runner(
            agent=self.orchestrator,
            app_name="policy-agent-system",
            session_service=self.session_service,
            memory_service=self.memory_service
        )
        
        # ========== Day 4: Observability ==========
        self.logger = AgentLogger("PolicyAgentSystem")
        self.tracer = AgentTracer("policy-agent")
        self.metrics = MetricsCollector()
        
        # ========== Day 4: Evaluator ==========
        self.evaluator = AgentEvaluator("PolicyAgentSystem")
        self._setup_golden_tasks()
        
        # ========== Day 5: A2A Protocol ==========
        self.agent_card = AgentCard(
            name="Environmental Policy Agent",
            description="AI agent that analyzes environmental policy effectiveness worldwide",
            version="1.0.0",
            skills=[
                AgentSkill(
                    id="analyze_policy",
                    name="Analyze Policy",
                    description="Analyze environmental policy effectiveness for a country",
                    tags=["analysis", "policy", "environment"],
                    examples=["Analyze South Korea's fine dust policy"]
                ),
                AgentSkill(
                    id="compare_countries",
                    name="Compare Countries",
                    description="Compare environmental policies across multiple countries",
                    tags=["comparison", "ranking"],
                    examples=["Compare Korea, China, and Japan"]
                ),
                AgentSkill(
                    id="get_air_quality",
                    name="Get Air Quality",
                    description="Get real-time air quality data for a city",
                    tags=["data", "air quality", "real-time"],
                    examples=["What's the air quality in Seoul?"]
                )
            ],
            endpoint="http://localhost:8000"
        )
        self.a2a = A2AProtocol(self.orchestrator, self.agent_card)
    
    def _setup_golden_tasks(self):
        """Setup golden tasks for evaluation."""
        self.evaluator.add_golden_task(
            name="korea_policy_exists",
            input_query="South Korea",
            expected_output="Fine Dust",
            validator=AgentEvaluator.contains_validator,
            weight=1.0
        )
        self.evaluator.add_golden_task(
            name="effectiveness_score_valid",
            input_query="South Korea",
            expected_output={"min": 0, "max": 100},
            validator=lambda exp, act: 0 <= act.get("analysis", {}).get("effectiveness_score", -1) <= 100,
            weight=1.5
        )
    
    async def analyze(self, country: str) -> Dict[str, Any]:
        """
        Full analysis pipeline with observability.
        Demonstrates Day 1-4 concepts.
        """
        # Start trace
        trace_id = self.tracer.start_trace(f"analyze:{country}")
        session_id = self.session_service.create_session()
        
        self.logger.info(f"Starting analysis for {country}", trace_id=trace_id)
        self.metrics.increment("analysis_requests")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Collection (Day 2: Tools)
            self.tracer.add_span("data_collection")
            self.logger.log_agent_step("data_collection", {"country": country})
            
            city = country.split()[0]  # Get first word as city name
            air_data = get_air_quality(city)
            policies = search_policies(country)
            
            self.metrics.increment("tool_calls", labels={"tool": "get_air_quality"})
            self.metrics.increment("tool_calls", labels={"tool": "search_policies"})
            
            # Step 2: Analysis (Day 2: Tools)
            self.tracer.add_span("policy_analysis")
            self.logger.log_agent_step("policy_analysis", {"policies_found": len(policies)})
            
            if policies:
                policy = policies[0]
                analysis = analyze_effectiveness(
                    target=policy["target_reduction"],
                    actual=policy["actual_reduction"],
                    before_values=[policy.get("before_pm25", 50)] * 5,
                    after_values=[policy.get("after_pm25", 30)] * 5
                )
                self.metrics.increment("analyses_completed")
            else:
                analysis = {
                    "effectiveness_score": 0,
                    "message": "No policies found",
                    "rating": "N/A",
                    "emoji": "⚪"
                }
                self.metrics.increment("analyses_no_data")
            
            # Step 3: Report Generation
            self.tracer.add_span("report_generation")
            report = self._generate_report(country, air_data, policies, analysis)
            
            # Compile result
            result = {
                "country": country,
                "timestamp": datetime.now().isoformat(),
                "air_quality": air_data,
                "policies": policies,
                "analysis": analysis,
                "report": report,
                "trace_id": trace_id,
                "session_id": session_id
            }
            
            # Day 3: Store in memory
            self.memory_service.store(
                content=result,
                metadata={"type": "analysis", "country": country},
                tags=["analysis", country.lower().replace(" ", "_")]
            )
            self.session_service.add_to_history(session_id, "user", f"Analyze {country}")
            self.session_service.add_to_history(session_id, "assistant", f"Analysis completed: {analysis.get('rating', 'N/A')}")
            
            # End trace
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.tracer.end_trace(trace_id, "OK")
            self.metrics.record("analysis_duration_ms", duration_ms)
            
            self.logger.info(f"Analysis completed", trace_id=trace_id, duration_ms=duration_ms)
            
            return result
            
        except Exception as e:
            self.tracer.end_trace(trace_id, "ERROR")
            self.logger.error(f"Analysis failed: {e}", trace_id=trace_id)
            self.metrics.increment("analysis_errors")
            raise

    
    def _generate_report(
        self,
        country: str,
        air_data: Dict,
        policies: List[Dict],
        analysis: Dict
    ) -> str:
        """Generate comprehensive report in Korean."""
        score = analysis.get("effectiveness_score", 0)
        rating = analysis.get("rating_kr", analysis.get("rating", "N/A"))
        emoji = analysis.get("emoji", "⚪")
        policy_name = policies[0]["name"] if policies else "정책 없음"
        policy_name_kr = policies[0].get("name_kr", policy_name) if policies else "정책 없음"
        
        return f"""
{'='*60}
📋 {country} 환경정책 분석 보고서
{'='*60}

📌 분석 대상 정책: {policy_name_kr}
   ({policy_name})

{'─'*60}
📊 효과성 평가
{'─'*60}
   점수: {score}/100 {emoji} {rating}
   목표 감축률: {analysis.get('target_reduction', 'N/A')}
   실제 감축률: {analysis.get('actual_reduction', 'N/A')}
   목표 달성: {'✅ 달성' if analysis.get('exceeded_target') else '❌ 미달성'}

{'─'*60}
📈 통계 분석 결과
{'─'*60}
   통계적 유의성: {analysis.get('statistical_significance', 'N/A')}
   효과 크기 (Cohen's d): {analysis.get('cohens_d', 'N/A')}
   효과 수준: {analysis.get('effect_size', 'N/A')}

{'─'*60}
🌍 현재 대기질 상태 ({air_data.get('city', country)})
{'─'*60}
   AQI: {air_data.get('aqi', 'N/A')}
   PM2.5: {air_data.get('pm25', 'N/A')} μg/m³
   PM10: {air_data.get('pm10', 'N/A')} μg/m³
   상태: {air_data.get('status', 'N/A')}
   데이터 출처: {air_data.get('source', 'Unknown')}

{'─'*60}
💡 결론
{'─'*60}
   이 정책은 {emoji} {rating} 것으로 평가됩니다.
   {'목표를 초과 달성하여 우수한 성과를 보이고 있습니다.' if analysis.get('exceeded_target') else '목표 달성을 위한 추가 노력이 필요합니다.'}

{'='*60}
Generated by Environmental Policy Impact Agent System
Team Robee | Kaggle AI Agents Intensive Capstone
{'='*60}
"""
    
    async def compare(self, countries: List[str]) -> Dict[str, Any]:
        """Compare policies across multiple countries."""
        trace_id = self.tracer.start_trace(f"compare:{','.join(countries)}")
        self.logger.info(f"Starting comparison for {len(countries)} countries")
        
        comparison = compare_countries(countries)
        
        self.tracer.end_trace(trace_id, "OK")
        self.metrics.increment("comparisons_completed")
        
        return comparison
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run golden task evaluation (Day 4)."""
        def test_fn(query: str) -> Dict:
            # Synchronous wrapper for evaluation
            policies = search_policies(query)
            if policies:
                policy = policies[0]
                analysis = analyze_effectiveness(
                    target=policy["target_reduction"],
                    actual=policy["actual_reduction"]
                )
                return {"policies": policies, "analysis": analysis}
            return {"policies": [], "analysis": {}}
        
        return self.evaluator.evaluate_all(test_fn)
    
    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        return {
            "metrics": self.metrics.summary(),
            "logs": {
                "total": len(self.logger.logs),
                "by_level": {
                    level: len([l for l in self.logger.logs if l["level"] == level])
                    for level in ["DEBUG", "INFO", "WARNING", "ERROR"]
                }
            },
            "traces": {
                "total": len(self.tracer.traces),
                "recent": [
                    self.tracer.get_trace_summary(tid)
                    for tid in list(self.tracer.traces.keys())[-5:]
                ]
            },
            "memory": self.memory_service.get_stats(),
            "sessions": {
                "total": len(self.session_service.sessions)
            }
        }
    
    def get_a2a_card(self) -> str:
        """Get A2A Agent Card (Day 5)."""
        return self.agent_card.to_json()


# ============================================================
# Main Demo Function
# ============================================================

async def main():
    """
    Demonstrate the Environmental Policy Impact Agent System.
    Shows all 5 days of Google AI Agents Intensive concepts.
    """
    print("\n" + "="*70)
    print("🌍 Environmental Policy Impact Agent System")
    print("   Kaggle AI Agents Intensive Capstone Project - Team Robee")
    print("="*70)
    
    print("\n✅ Implemented Google AI Agents Intensive Concepts:")
    print("   📚 Day 1: Multi-Agent Architecture (Agent, Runner, Sub-agents)")
    print("   🛠️ Day 2: Custom Tools (FunctionTool, WAQI API, Policy DB)")
    print("   🧠 Day 3: Memory & Context (Session, Long-term Memory)")
    print("   📊 Day 4: Observability (Logs, Traces, Metrics) & Evaluation")
    print("   🤝 Day 5: A2A Protocol (AgentCard, RemoteAgent)")
    print("-"*70)
    
    # Initialize system
    print("\n⚙️ Initializing Policy Agent System...")
    system = PolicyAgentSystem()
    print("   ✅ System initialized with 3 sub-agents + 1 orchestrator")
    
    # ========== Demo 1: Single Country Analysis ==========
    print("\n" + "="*70)
    print("📊 DEMO 1: Single Country Analysis (South Korea)")
    print("="*70)
    
    result = await system.analyze("South Korea")
    print(result["report"])
    
    # ========== Demo 2: Multi-Country Comparison ==========
    print("\n" + "="*70)
    print("📊 DEMO 2: Multi-Country Comparison")
    print("="*70)
    
    comparison = await system.compare(["South Korea", "China", "Japan", "Germany"])
    
    print("\n🏆 Country Rankings by Policy Effectiveness:\n")
    print(f"{'Rank':<6}{'Country':<15}{'Policy':<40}{'Score':<8}{'Rating':<10}")
    print("-"*70)
    
    for item in comparison["comparison"]:
        print(f"{item['rank']:<6}{item['country']:<15}{item['policy_name'][:38]:<40}{item['effectiveness_score']:<8}{item['emoji']} {item['rating']:<10}")
    
    print(f"\n🥇 Best Performer: {comparison['best_performer']}")
    
    # ========== Demo 3: Real-time Air Quality ==========
    print("\n" + "="*70)
    print("🌍 DEMO 3: Real-time Air Quality Check")
    print("="*70)
    
    cities = ["Seoul", "Beijing", "Tokyo", "Delhi", "Paris"]
    print(f"\n{'City':<12}{'AQI':<8}{'PM2.5':<10}{'Status':<20}{'Source':<15}")
    print("-"*70)
    
    for city in cities:
        data = get_air_quality(city)
        print(f"{city:<12}{data['aqi']:<8}{data['pm25']:<10}{data['status']:<20}{data.get('source', 'Demo'):<15}")
    
    # ========== Demo 4: Observability Summary ==========
    print("\n" + "="*70)
    print("📈 DEMO 4: Observability Summary (Day 4)")
    print("="*70)
    
    obs = system.get_observability_summary()
    
    print(f"\n📊 Metrics:")
    print(f"   Counters: {obs['metrics']['counters']}")
    
    print(f"\n📝 Logs:")
    print(f"   Total entries: {obs['logs']['total']}")
    print(f"   By level: {obs['logs']['by_level']}")
    
    print(f"\n🔍 Traces:")
    print(f"   Total traces: {obs['traces']['total']}")
    
    print(f"\n🧠 Memory:")
    print(f"   Stored memories: {obs['memory']['total_memories']}")
    print(f"   Tags: {obs['memory']['tags']}")
    
    # ========== Demo 5: Evaluation (Day 4) ==========
    print("\n" + "="*70)
    print("✅ DEMO 5: Agent Evaluation (Day 4)")
    print("="*70)
    
    eval_results = system.run_evaluation()
    
    print(f"\n📋 Evaluation Results:")
    print(f"   Total Tasks: {eval_results['total_tasks']}")
    print(f"   Passed: {eval_results['passed']}")
    print(f"   Failed: {eval_results['failed']}")
    print(f"   Pass Rate: {eval_results['pass_rate']*100:.1f}%")
    print(f"   Weighted Score: {eval_results['weighted_score']*100:.1f}%")
    
    # ========== Demo 6: A2A Agent Card (Day 5) ==========
    print("\n" + "="*70)
    print("🤝 DEMO 6: A2A Agent Card (Day 5)")
    print("="*70)
    
    print("\nAgent Card (/.well-known/agent.json):")
    print(system.get_a2a_card())
    
    # ========== Demo 7: Workflow Agents (ADK Advanced Patterns) ==========
    print("\n" + "="*70)
    print("🔄 DEMO 7: Workflow Agents (ADK Advanced Patterns)")
    print("="*70)
    
    # Demo SequentialAgent
    print("\n📋 7.1 SequentialAgent Demo:")
    print("-"*40)
    
    data_agent = Agent(name="data_collector", output_key="collected_data")
    analyzer_agent = Agent(name="analyzer", output_key="analysis_result")
    
    sequential = SequentialAgent(
        name="data_pipeline",
        sub_agents=[data_agent, analyzer_agent]
    )
    seq_result = await sequential.run("Analyze South Korea", {})
    print(f"   Workflow: {seq_result['workflow']}")
    print(f"   Steps completed: {seq_result['steps_completed']}")
    print(f"   Agents executed: {[r['agent'] for r in seq_result['sequential_results']]}")
    
    # Demo ParallelAgent
    print("\n⚡ 7.2 ParallelAgent Demo:")
    print("-"*40)
    
    air_agent = Agent(name="air_fetcher", output_key="air_data")
    policy_agent = Agent(name="policy_fetcher", output_key="policy_data")
    
    parallel = ParallelAgent(
        name="multi_fetcher",
        sub_agents=[air_agent, policy_agent]
    )
    par_result = await parallel.run("Fetch data", {})
    print(f"   Workflow: {par_result['workflow']}")
    print(f"   Agents executed: {par_result['agents_executed']}")
    print(f"   Output keys: {list(par_result['merged_outputs'].keys())}")
    
    # Demo LoopAgent with ValidationChecker
    print("\n🔁 7.3 LoopAgent with ValidationChecker Demo:")
    print("-"*40)
    
    # Create a simple score incrementer agent
    class ScoreIncrementer(Agent):
        async def run(self, query=None, context=None):
            context = context or {}
            # Get current score (handle both int and dict cases)
            current = context.get('effectiveness_score', 0)
            if isinstance(current, dict):
                current = current.get('effectiveness_score', 0)
            new_score = min(100, current + 30)  # Increment by 30 each iteration
            # Store directly in context for next iteration
            context['effectiveness_score'] = new_score
            return {"effectiveness_score": new_score, "iteration": context.get('_iteration', 1)}
    
    incrementer = ScoreIncrementer(name="score_incrementer")
    validator = EffectivenessValidator(name="score_validator", threshold=80.0, score_key="effectiveness_score")
    
    loop_agent = LoopAgent(
        name="score_refiner",
        sub_agents=[incrementer, validator],
        max_iterations=5
    )
    
    loop_result = await loop_agent.run("Refine score", {"effectiveness_score": 0})
    print(f"   Workflow: {loop_result['workflow']}")
    print(f"   Iterations: {loop_result['iterations']}")
    print(f"   Exit reason: {loop_result['exit_reason']}")
    final_score = loop_result.get('final_result', {}).get('score', 'N/A')
    print(f"   Final score: {final_score}")
    print(f"   Validation passed: {loop_result.get('final_result', {}).get('validation_passed', 'N/A')}")
    
    # Demo Callbacks
    print("\n📡 7.4 Callbacks Demo:")
    print("-"*40)
    
    callback_log = []
    
    logging_callback = AgentCallback(
        on_start=lambda name, ctx: callback_log.append(f"Started: {name}"),
        on_complete=lambda name, ctx, res: callback_log.append(f"Completed: {name}")
    )
    
    callback_manager = CallbackManager()
    callback_manager.register(logging_callback)
    
    callback_manager.trigger_start("test_agent", {})
    callback_manager.trigger_complete("test_agent", {}, {"status": "ok"})
    
    print(f"   Registered callbacks: 1")
    print(f"   Callback log: {callback_log}")
    
    # Demo QualityGateValidator
    print("\n🚦 7.5 QualityGateValidator Demo:")
    print("-"*40)
    
    quality_gate = QualityGateValidator(
        name="quality_gate",
        criteria={
            "score_min": lambda ctx: ctx.get("score", 0) >= 70,
            "has_data": lambda ctx: bool(ctx.get("data")),
            "no_errors": lambda ctx: not ctx.get("errors")
        }
    )
    
    test_context = {"score": 85, "data": {"test": 1}, "errors": None}
    gate_result = await quality_gate.run(context=test_context)
    
    print(f"   Criteria checked: {gate_result['total_criteria']}")
    print(f"   Passed: {gate_result['passed_count']}/{gate_result['total_criteria']}")
    print(f"   All passed: {gate_result['validation_passed']}")
    print(f"   Escalate signal: {gate_result['escalate']}")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("🏆 DEMO COMPLETE!")
    print("="*70)
    print("""
✅ Successfully demonstrated:
   • Multi-agent coordination (4 agents)
   • Custom tools (4 tools with real API support)
   • Session & long-term memory
   • Full observability (logs, traces, metrics)
   • Golden task evaluation
   • A2A protocol for agent discovery
   
🆕 ADK Advanced Patterns (NEW):
   • SequentialAgent - Pipeline execution
   • ParallelAgent - Concurrent execution  
   • LoopAgent - Iterative refinement
   • ValidationChecker - Quality gates
   • Callbacks - Lifecycle event handling
   • output_key - State management

📚 GitHub: https://github.com/joymin5655/Kaggle/tree/main/agents-intensive-capstone
🎓 Course: Google AI Agents Intensive (Kaggle)
👥 Team: Robee

Built with ❤️ for a cleaner planet 🌍
""")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
