"""
Policy Analyzer Agent
Responsible for statistical analysis of policy effectiveness

Day 1 Concept: Specialized Agent in Multi-Agent System
"""

from typing import Dict, Any, List, Optional
from tools.analysis_tool import (
    calculate_trend,
    compare_before_after,
    calculate_statistical_significance,
    calculate_moving_average
)


class PolicyAnalyzerAgent:
    """
    Policy Analysis Agent - Part of Multi-Agent System (Day 1)
    
    Responsibilities:
    - Perform statistical analysis on environmental data
    - Compare before/after policy metrics
    - Calculate statistical significance
    - Identify trends and patterns
    
    Tools Used (Day 2):
    - calculate_trend
    - compare_before_after
    - calculate_statistical_significance
    """
    
    def __init__(self):
        self.name = "PolicyAnalyzer"
        self.description = "Analyzes policy effectiveness using statistics"
    
    def analyze_policy_impact(
        self,
        before_data: List[float],
        after_data: List[float],
        metric_name: str = "PM2.5"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of policy impact.
        
        Args:
            before_data: Values before policy
            after_data: Values after policy
            metric_name: Name of the metric
        
        Returns:
            Complete analysis results
        """
        # Basic comparison
        comparison = compare_before_after(
            before_data, 
            after_data, 
            metric_name
        )
        
        # Statistical significance
        significance = calculate_statistical_significance(
            before_data, 
            after_data
        )
        
        # Trend analysis
        all_data = before_data + after_data
        trend = calculate_trend(all_data)
        
        # Combine results
        return {
            "metric": metric_name,
            "comparison": comparison,
            "statistical_test": significance,
            "trend_analysis": trend,
            "summary": self._generate_analysis_summary(
                comparison, 
                significance
            ),
            "status": "success"
        }
    
    def _generate_analysis_summary(
        self,
        comparison: Dict,
        significance: Dict
    ) -> str:
        """Generate human-readable analysis summary."""
        parts = []
        
        # Change direction
        if comparison.get("improvement"):
            parts.append("정책 시행 후 개선 효과가 관찰되었습니다.")
        else:
            parts.append("정책 시행 후 악화 추세가 관찰되었습니다.")
        
        # Magnitude
        change = abs(comparison.get("percent_change", 0))
        parts.append(f"변화율: {change:.1f}%")
        
        # Effect size
        effect = comparison.get("effect_size", "")
        if effect in ["medium", "large"]:
            parts.append(f"효과 크기: {effect} (실질적 의미 있음)")
        
        # Statistical significance
        if significance.get("significant"):
            parts.append("통계적으로 유의미한 변화 (p < 0.05)")
        else:
            parts.append("통계적 유의성 확인 필요")
        
        return " | ".join(parts)
    
    def calculate_effectiveness_score(
        self,
        comparison: Dict,
        significance: Dict
    ) -> float:
        """
        Calculate overall effectiveness score (0-100).
        
        Args:
            comparison: Comparison analysis results
            significance: Statistical significance results
        
        Returns:
            Effectiveness score
        """
        score = 50  # Base score
        
        # Improvement bonus
        if comparison.get("improvement"):
            score += 20
        
        # Change magnitude
        change = abs(comparison.get("percent_change", 0))
        if change > 30:
            score += 15
        elif change > 15:
            score += 10
        elif change > 5:
            score += 5
        
        # Statistical significance
        if significance.get("significant"):
            score += 15
        
        # Effect size
        effect = comparison.get("effect_size", "")
        if effect == "large":
            score += 10
        elif effect == "medium":
            score += 5
        
        return min(100, max(0, score))
    
    def analyze_trend(
        self,
        data: List[float],
        timestamps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze trend in time series data.
        
        Args:
            data: List of values
            timestamps: Optional timestamps
        
        Returns:
            Trend analysis results
        """
        trend_result = calculate_trend(data, timestamps)
        
        # Add moving average for smoothing
        if len(data) >= 7:
            ma_result = calculate_moving_average(data, window=7)
            trend_result["moving_average"] = ma_result.get("moving_average")
        
        return trend_result
    
    def compare_policies(
        self,
        policy_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare effectiveness of multiple policies.
        
        Args:
            policy_results: List of analysis results for different policies
        
        Returns:
            Comparison results
        """
        scored_policies = []
        
        for result in policy_results:
            comparison = result.get("comparison", {})
            significance = result.get("statistical_test", {})
            
            score = self.calculate_effectiveness_score(
                comparison, 
                significance
            )
            
            scored_policies.append({
                "policy": result.get("policy"),
                "country": result.get("country"),
                "score": score,
                "reduction": comparison.get("percent_change", 0),
                "significant": significance.get("significant", False)
            })
        
        # Sort by score
        scored_policies.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "ranked_policies": scored_policies,
            "best_policy": scored_policies[0] if scored_policies else None,
            "average_score": (
                sum(p["score"] for p in scored_policies) / len(scored_policies)
                if scored_policies else 0
            ),
            "status": "success"
        }


# Create singleton instance
policy_analyzer_agent = PolicyAnalyzerAgent()
