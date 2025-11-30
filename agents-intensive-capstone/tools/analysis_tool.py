"""
Statistical Analysis Tool
Perform statistical analysis on environmental data

Day 2 Concept: Custom Tool Implementation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics
import math


def calculate_trend(
    values: List[float],
    timestamps: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate trend in time series data using linear regression.
    
    This is a custom tool for the Policy Analysis Agent.
    
    Args:
        values: List of numeric values
        timestamps: Optional list of ISO timestamps
    
    Returns:
        Trend analysis results
    
    Example:
        >>> data = [40, 38, 36, 34, 32, 30]
        >>> result = calculate_trend(data)
        >>> print(result["trend"])
        "decreasing"
    """
    if len(values) < 2:
        return {"error": "Need at least 2 data points", "status": "error"}
    
    n = len(values)
    x = list(range(n))
    
    # Linear regression: y = mx + b
    mean_x = sum(x) / n
    mean_y = sum(values) / n
    
    # Calculate slope
    numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    intercept = mean_y - slope * mean_x
    
    # Calculate R-squared
    ss_res = sum((values[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
    ss_tot = sum((values[i] - mean_y) ** 2 for i in range(n))
    
    if ss_tot == 0:
        r_squared = 0
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    # Determine trend direction
    if abs(slope) < 0.01 * mean_y:  # Less than 1% change
        trend = "stable"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    # Calculate predicted values
    predicted = [slope * i + intercept for i in x]
    
    return {
        "trend": trend,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "mean": round(mean_y, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "first_value": values[0],
        "last_value": values[-1],
        "predicted_next": round(slope * n + intercept, 2),
        "data_points": n,
        "status": "success"
    }


def compare_before_after(
    before_data: List[float],
    after_data: List[float],
    metric_name: str = "value"
) -> Dict[str, Any]:
    """
    Compare data before and after a policy implementation.
    
    This is a custom tool for the Policy Analysis Agent.
    
    Args:
        before_data: Values before policy
        after_data: Values after policy
        metric_name: Name of the metric being compared
    
    Returns:
        Comparison analysis results
    """
    if not before_data or not after_data:
        return {"error": "Need data for both periods", "status": "error"}
    
    # Calculate statistics
    before_mean = statistics.mean(before_data)
    after_mean = statistics.mean(after_data)
    
    before_std = statistics.stdev(before_data) if len(before_data) > 1 else 0
    after_std = statistics.stdev(after_data) if len(after_data) > 1 else 0
    
    # Calculate change
    absolute_change = after_mean - before_mean
    percent_change = (absolute_change / before_mean * 100) if before_mean != 0 else 0
    
    # Determine if improvement (for air quality, lower is better)
    improvement = after_mean < before_mean
    
    # Calculate effect size (Cohen's d)
    pooled_std = math.sqrt((before_std**2 + after_std**2) / 2)
    if pooled_std > 0:
        cohens_d = abs(before_mean - after_mean) / pooled_std
    else:
        cohens_d = 0
    
    # Interpret effect size
    if cohens_d < 0.2:
        effect_interpretation = "negligible"
    elif cohens_d < 0.5:
        effect_interpretation = "small"
    elif cohens_d < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    return {
        "metric": metric_name,
        "before_mean": round(before_mean, 2),
        "after_mean": round(after_mean, 2),
        "before_std": round(before_std, 2),
        "after_std": round(after_std, 2),
        "absolute_change": round(absolute_change, 2),
        "percent_change": round(percent_change, 2),
        "improvement": improvement,
        "cohens_d": round(cohens_d, 3),
        "effect_size": effect_interpretation,
        "sample_size_before": len(before_data),
        "sample_size_after": len(after_data),
        "status": "success"
    }


def calculate_statistical_significance(
    before_data: List[float],
    after_data: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate statistical significance using t-test.
    
    This is a custom tool for the Policy Analysis Agent.
    
    Args:
        before_data: Values before policy
        after_data: Values after policy
        alpha: Significance level (default 0.05)
    
    Returns:
        Statistical test results
    """
    if len(before_data) < 2 or len(after_data) < 2:
        return {
            "error": "Need at least 2 data points in each group",
            "status": "error"
        }
    
    # Calculate means and standard deviations
    mean1 = statistics.mean(before_data)
    mean2 = statistics.mean(after_data)
    std1 = statistics.stdev(before_data)
    std2 = statistics.stdev(after_data)
    n1 = len(before_data)
    n2 = len(after_data)
    
    # Calculate t-statistic (Welch's t-test)
    se = math.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    if se == 0:
        return {
            "error": "Cannot calculate t-statistic (zero variance)",
            "status": "error"
        }
    
    t_statistic = (mean1 - mean2) / se
    
    # Degrees of freedom (Welch-Satterthwaite)
    numerator = ((std1**2 / n1) + (std2**2 / n2)) ** 2
    denominator = (
        (std1**2 / n1)**2 / (n1 - 1) + 
        (std2**2 / n2)**2 / (n2 - 1)
    )
    
    if denominator == 0:
        df = n1 + n2 - 2
    else:
        df = numerator / denominator
    
    # Approximate p-value using normal distribution
    # (simplified - in production use scipy.stats)
    z = abs(t_statistic)
    
    # Approximation for two-tailed p-value
    if z > 3.5:
        p_value = 0.0005
    elif z > 3.0:
        p_value = 0.003
    elif z > 2.5:
        p_value = 0.012
    elif z > 2.0:
        p_value = 0.046
    elif z > 1.5:
        p_value = 0.134
    elif z > 1.0:
        p_value = 0.317
    else:
        p_value = 0.5 + z * 0.15  # Rough approximation
    
    # Determine significance
    significant = p_value < alpha
    
    # Confidence interval (95%)
    margin = 1.96 * se  # Approximate for large samples
    ci_lower = (mean1 - mean2) - margin
    ci_upper = (mean1 - mean2) + margin
    
    return {
        "t_statistic": round(t_statistic, 4),
        "degrees_of_freedom": round(df, 2),
        "p_value": round(p_value, 4),
        "alpha": alpha,
        "significant": significant,
        "confidence_interval": {
            "lower": round(ci_lower, 2),
            "upper": round(ci_upper, 2),
            "level": 0.95
        },
        "interpretation": (
            "Statistically significant difference" if significant
            else "No statistically significant difference"
        ),
        "status": "success"
    }


def calculate_moving_average(
    values: List[float],
    window: int = 7
) -> Dict[str, Any]:
    """
    Calculate moving average for smoothing data.
    
    Args:
        values: List of values
        window: Window size for averaging
    
    Returns:
        Moving average values
    """
    if len(values) < window:
        return {"error": f"Need at least {window} data points", "status": "error"}
    
    moving_avg = []
    for i in range(len(values) - window + 1):
        avg = sum(values[i:i+window]) / window
        moving_avg.append(round(avg, 2))
    
    return {
        "original_count": len(values),
        "smoothed_count": len(moving_avg),
        "window_size": window,
        "moving_average": moving_avg,
        "status": "success"
    }
