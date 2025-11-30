"""
Agent Evaluator - Quality evaluation for AI agents
Day 4: Observability, Logging, Tracing, Evaluation
"""
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class EvaluationResult:
    """Result of an evaluation test."""
    test_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    expected: Any
    actual: Any
    timestamp: datetime
    details: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "score": self.score,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }

@dataclass
class GoldenTask:
    """A golden task for evaluation."""
    name: str
    input_query: str
    expected_output: Any
    validator: Optional[Callable[[Any, Any], bool]] = None
    
class AgentEvaluator:
    """Evaluates agent performance against golden tasks."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.golden_tasks: List[GoldenTask] = []
        self.results: List[EvaluationResult] = []
    
    def add_golden_task(
        self,
        name: str,
        input_query: str,
        expected_output: Any,
        validator: Optional[Callable] = None
    ):
        """Add a golden task for evaluation."""
        task = GoldenTask(
            name=name,
            input_query=input_query,
            expected_output=expected_output,
            validator=validator
        )
        self.golden_tasks.append(task)

    def evaluate_task(
        self,
        task: GoldenTask,
        actual_output: Any
    ) -> EvaluationResult:
        """Evaluate a single task."""
        if task.validator:
            passed = task.validator(task.expected_output, actual_output)
            score = 1.0 if passed else 0.0
        else:
            # Default string comparison
            passed = str(task.expected_output) == str(actual_output)
            score = 1.0 if passed else 0.0
        
        result = EvaluationResult(
            test_name=task.name,
            passed=passed,
            score=score,
            expected=task.expected_output,
            actual=actual_output,
            timestamp=datetime.now()
        )
        self.results.append(result)
        return result
    
    def run_all_evaluations(
        self,
        agent_fn: Callable[[str], Any]
    ) -> Dict[str, Any]:
        """Run all golden tasks and return summary."""
        results = []
        for task in self.golden_tasks:
            actual = agent_fn(task.input_query)
            result = self.evaluate_task(task, actual)
            results.append(result)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        return {
            "agent": self.agent_name,
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_score": sum(r.score for r in results) / total if total > 0 else 0,
            "results": [r.to_dict() for r in results]
        }
    
    # Predefined validators
    @staticmethod
    def contains_validator(expected: str, actual: str) -> bool:
        """Check if actual contains expected."""
        return expected.lower() in str(actual).lower()
    
    @staticmethod
    def numeric_range_validator(expected: Dict, actual: float) -> bool:
        """Check if actual is within expected range."""
        min_val = expected.get("min", float("-inf"))
        max_val = expected.get("max", float("inf"))
        return min_val <= actual <= max_val
    
    @staticmethod
    def key_exists_validator(expected_keys: List[str], actual: Dict) -> bool:
        """Check if all expected keys exist."""
        if not isinstance(actual, dict):
            return False
        return all(key in actual for key in expected_keys)
    
    def export_results(self, filepath: str):
        """Export evaluation results to JSON."""
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
