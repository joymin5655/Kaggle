"""
Agent Tracer - Distributed tracing for multi-agent systems
Day 4: Observability, Logging, Tracing, Evaluation
"""
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class Span:
    """Represents a single span in a trace."""
    span_id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "IN_PROGRESS"
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def add_event(self, name: str, attributes: Optional[Dict] = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status
        }


class AgentTracer:
    """Distributed tracer for agent operations."""
    
    def __init__(self, service_name: str = "policy-agent-system"):
        self.service_name = service_name
        self.traces: Dict[str, List[Span]] = {}
        self.active_spans: Dict[str, Span] = {}
    
    def start_trace(self, name: str) -> str:
        """Start a new trace and return trace_id."""
        trace_id = str(uuid.uuid4())[:8]
        span = self._create_span(name, trace_id, None)
        self.traces[trace_id] = [span]
        self.active_spans[trace_id] = span
        return trace_id
    
    def _create_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str]
    ) -> Span:
        return Span(
            span_id=str(uuid.uuid4())[:8],
            trace_id=trace_id,
            name=name,
            start_time=datetime.now(),
            parent_span_id=parent_span_id
        )
    
    @contextmanager
    def span(self, trace_id: str, name: str, attributes: Optional[Dict] = None):
        """Context manager for creating child spans."""
        parent_span = self.active_spans.get(trace_id)
        parent_span_id = parent_span.span_id if parent_span else None
        
        new_span = self._create_span(name, trace_id, parent_span_id)
        if attributes:
            new_span.attributes = attributes
        
        self.traces[trace_id].append(new_span)
        old_active = self.active_spans.get(trace_id)
        self.active_spans[trace_id] = new_span
        
        try:
            yield new_span
            new_span.status = "OK"
        except Exception as e:
            new_span.status = "ERROR"
            new_span.attributes["error"] = str(e)
            raise
        finally:
            new_span.end_time = datetime.now()
            if old_active:
                self.active_spans[trace_id] = old_active
    
    def end_trace(self, trace_id: str):
        """End a trace."""
        if trace_id in self.traces:
            root_span = self.traces[trace_id][0]
            root_span.end_time = datetime.now()
            root_span.status = "OK"
    
    def get_trace(self, trace_id: str) -> List[Dict]:
        """Get all spans for a trace."""
        if trace_id in self.traces:
            return [span.to_dict() for span in self.traces[trace_id]]
        return []
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a trace."""
        spans = self.traces.get(trace_id, [])
        if not spans:
            return {}
        
        root = spans[0]
        return {
            "trace_id": trace_id,
            "name": root.name,
            "total_duration_ms": root.duration_ms,
            "span_count": len(spans),
            "status": root.status,
            "spans": [s.name for s in spans]
        }
