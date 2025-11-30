"""
Agent Logger - Structured logging for AI agents
Day 4: Observability, Logging, Tracing, Evaluation
"""
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AgentLogger:
    """Structured logger for agent observability."""
    
    def __init__(self, agent_name: str, log_file: Optional[str] = None):
        self.agent_name = agent_name
        self.logs: list[Dict[str, Any]] = []
        
        # Setup Python logger
        self.logger = logging.getLogger(f"agent.{agent_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create structured log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "level": level.value,
            "message": message,
            "metadata": metadata or {}
        }
        self.logs.append(entry)
        return entry

    def log_tool_call(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True
    ):
        """Log a tool invocation."""
        metadata = {
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs,
            "duration_ms": duration_ms,
            "success": success
        }
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Tool call: {tool_name} - {'Success' if success else 'Failed'}"
        entry = self._create_log_entry(level, message, metadata)
        
        if success:
            self.logger.info(f"{tool_name} completed in {duration_ms}ms")
        else:
            self.logger.error(f"{tool_name} failed: {outputs}")
        
        return entry
    
    def log_agent_step(
        self,
        step_name: str,
        input_data: Any,
        output_data: Any,
        reasoning: Optional[str] = None
    ):
        """Log an agent reasoning step."""
        metadata = {
            "step": step_name,
            "input": str(input_data)[:500],
            "output": str(output_data)[:500],
            "reasoning": reasoning
        }
        entry = self._create_log_entry(LogLevel.INFO, f"Agent step: {step_name}", metadata)
        self.logger.info(f"Step '{step_name}' completed")
        return entry
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log an error with context."""
        metadata = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        entry = self._create_log_entry(LogLevel.ERROR, f"Error: {error}", metadata)
        self.logger.error(f"{type(error).__name__}: {error}")
        return entry
    
    def get_logs(self, level: Optional[LogLevel] = None) -> list[Dict]:
        """Get all logs, optionally filtered by level."""
        if level is None:
            return self.logs
        return [log for log in self.logs if log["level"] == level.value]
    
    def export_logs(self, filepath: str):
        """Export logs to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2, default=str)
        self.logger.info(f"Logs exported to {filepath}")
