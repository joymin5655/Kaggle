"""
Visualizer Agent
Responsible for creating visualizations from analysis data

Day 1 Concept: Specialized Agent in Multi-Agent System
"""

from typing import Dict, Any, List, Optional
from tools.visualization_tool import (
    generate_globe_visualization_config,
    create_timeline_chart_config,
    create_comparison_chart_config,
    create_before_after_chart_config
)


class VisualizerAgent:
    """
    Visualization Agent - Part of Multi-Agent System (Day 1)
    
    Responsibilities:
    - Generate 3D globe visualizations
    - Create timeline charts
    - Build comparison bar charts
    - Produce before/after analysis charts
    
    Tools Used (Day 2):
    - generate_globe_config
    - create_timeline_chart
    - create_comparison_chart
    """
    
    def __init__(self):
        self.name = "Visualizer"
        self.description = "Creates visualizations from analysis data"
    
    def create_globe_visualization(
        self,
        countries_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create 3D globe visualization config.
        
        Args:
            countries_data: List of country data with coordinates
        
        Returns:
            Globe configuration for Three.js
        """
        return generate_globe_visualization_config(countries_data)
    
    def create_timeline(
        self,
        time_series: List[Dict[str, Any]],
        metric: str = "pm25",
        title: str = "Air Quality Timeline"
    ) -> Dict[str, Any]:
        """
        Create timeline chart for policy impact.
        
        Args:
            time_series: Time series data
            metric: Metric being visualized
            title: Chart title
        
        Returns:
            Chart.js configuration
        """
        return create_timeline_chart_config(time_series, metric, title)
    
    def create_comparison(
        self,
        countries_data: List[Dict[str, Any]],
        metric: str = "pm25",
        title: str = "Country Comparison"
    ) -> Dict[str, Any]:
        """
        Create comparison bar chart.
        
        Args:
            countries_data: Data for each country
            metric: Metric to compare
            title: Chart title
        
        Returns:
            Chart.js configuration
        """
        return create_comparison_chart_config(countries_data, metric, title)
    
    def create_before_after_chart(
        self,
        before_values: List[float],
        after_values: List[float],
        title: str = "Policy Impact"
    ) -> Dict[str, Any]:
        """
        Create before/after comparison chart.
        
        Args:
            before_values: Values before policy
            after_values: Values after policy
            title: Chart title
        
        Returns:
            Chart.js configuration
        """
        return create_before_after_chart_config(
            before_values, 
            after_values, 
            title=title
        )
    
    def create_dashboard(
        self,
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create complete dashboard with multiple visualizations.
        
        Args:
            analysis_result: Complete analysis result
        
        Returns:
            Dashboard configuration with all charts
        """
        dashboard = {
            "title": f"Environmental Policy Dashboard",
            "charts": []
        }
        
        # Add available visualizations
        comparison = analysis_result.get("analysis", {})
        
        if comparison.get("before_mean") and comparison.get("after_mean"):
            # Create simple before/after visualization
            dashboard["charts"].append({
                "type": "summary_card",
                "data": {
                    "before": comparison.get("before_mean"),
                    "after": comparison.get("after_mean"),
                    "change": comparison.get("percent_change"),
                    "improvement": comparison.get("improvement")
                }
            })
        
        # Add country info if available
        current_aqi = analysis_result.get("current_air_quality", {})
        if current_aqi.get("aqi"):
            dashboard["charts"].append({
                "type": "aqi_gauge",
                "data": {
                    "value": current_aqi.get("aqi"),
                    "category": current_aqi.get("category"),
                    "color": current_aqi.get("color")
                }
            })
        
        dashboard["generated"] = True
        return dashboard


# Create singleton instance
visualizer_agent = VisualizerAgent()
