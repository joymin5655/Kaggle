"""
Custom Tools for Environmental Policy Agent System

Day 2 Concept: Custom Tools & MCP Integration
"""

from .waqi_tool import (
    fetch_waqi_realtime_data,
    get_aqi_color,
    get_aqi_category
)
from .policy_db_tool import (
    search_environmental_policies,
    get_policy_details
)
from .analysis_tool import (
    calculate_trend,
    compare_before_after,
    calculate_statistical_significance
)
from .visualization_tool import (
    generate_globe_visualization_config,
    create_timeline_chart_config,
    create_comparison_chart_config
)

__all__ = [
    # WAQI Tools
    "fetch_waqi_realtime_data",
    "get_aqi_color",
    "get_aqi_category",
    # Policy DB Tools
    "search_environmental_policies",
    "get_policy_details",
    # Analysis Tools
    "calculate_trend",
    "compare_before_after",
    "calculate_statistical_significance",
    # Visualization Tools
    "generate_globe_visualization_config",
    "create_timeline_chart_config",
    "create_comparison_chart_config",
]
