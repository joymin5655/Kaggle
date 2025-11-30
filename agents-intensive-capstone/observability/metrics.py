"""
Agent Metrics - Performance metrics collection
Day 4: Observability, Logging, Tracing, Evaluation
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

@dataclass
class Metric:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

class MetricsCollector:
    """Collects and aggregates agent performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
    
    # Counter methods
    def increment(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self.counters[key] += value
        self._record(name, self.counters[key], labels, "count")
    
    def _make_key(self, name: str, labels: Optional[Dict]) -> str:
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}:{label_str}"
        return name
    
    def _record(self, name: str, value: float, labels: Optional[Dict], unit: str):
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        self.metrics[name].append(metric)

    # Gauge methods
    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        self._record(name, value, labels, "gauge")
    
    # Histogram/timing methods
    def record_duration(self, name: str, duration_ms: float, labels: Optional[Dict] = None):
        """Record a duration measurement."""
        self._record(name, duration_ms, labels, "ms")
    
    def record_value(self, name: str, value: float, unit: str = "", labels: Optional[Dict] = None):
        """Record a generic value."""
        self._record(name, value, labels, unit)
    
    # Agent-specific metrics
    def record_tool_call(self, tool_name: str, duration_ms: float, success: bool):
        """Record a tool call metric."""
        self.increment("tool_calls_total", labels={"tool": tool_name, "success": str(success)})
        self.record_duration("tool_duration_ms", duration_ms, labels={"tool": tool_name})
    
    def record_agent_step(self, agent_name: str, step_name: str, duration_ms: float):
        """Record an agent step metric."""
        self.increment("agent_steps_total", labels={"agent": agent_name})
        self.record_duration("agent_step_duration_ms", duration_ms, labels={"agent": agent_name, "step": step_name})
    
    def record_llm_call(self, model: str, tokens_in: int, tokens_out: int, duration_ms: float):
        """Record LLM API call metrics."""
        self.increment("llm_calls_total", labels={"model": model})
        self.record_value("llm_tokens_in", tokens_in, "tokens", {"model": model})
        self.record_value("llm_tokens_out", tokens_out, "tokens", {"model": model})
        self.record_duration("llm_duration_ms", duration_ms, {"model": model})
    
    # Aggregation methods
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistical summary for a metric."""
        values = [m.value for m in self.metrics.get(name, [])]
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {name: self.get_stats(name) for name in self.metrics}
        }
