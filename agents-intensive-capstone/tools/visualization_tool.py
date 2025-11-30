"""
Visualization Tool
Generate visualization configurations for charts and globe

Day 2 Concept: Custom Tool Implementation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def generate_globe_visualization_config(
    countries_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate configuration for 3D globe visualization.
    
    This is a custom tool for the Visualization Agent.
    Compatible with Three.js globe implementations.
    
    Args:
        countries_data: List of dicts with country, aqi, pm25, lat, lon
    
    Returns:
        Globe visualization configuration
    """
    markers = []
    
    for country in countries_data:
        aqi = country.get("aqi", 50)
        
        # Determine color based on AQI
        if aqi <= 50:
            color = "#00e400"
        elif aqi <= 100:
            color = "#ffff00"
        elif aqi <= 150:
            color = "#ff7e00"
        elif aqi <= 200:
            color = "#ff0000"
        elif aqi <= 300:
            color = "#8f3f97"
        else:
            color = "#7e0023"
        
        # Size based on AQI (larger = worse)
        size = min(aqi / 30, 10)
        
        markers.append({
            "id": country.get("country", "Unknown").lower().replace(" ", "_"),
            "name": country.get("country", "Unknown"),
            "lat": country.get("latitude", 0),
            "lng": country.get("longitude", 0),
            "aqi": aqi,
            "pm25": country.get("pm25"),
            "color": color,
            "size": size,
            "popup": f"{country.get('country')}: AQI {aqi}"
        })
    
    return {
        "type": "globe",
        "config": {
            "globeImageUrl": "//unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
            "backgroundColor": "#000011",
            "pointOfView": {"lat": 35, "lng": 127, "altitude": 2.5},
            "atmosphereColor": "#3a228a",
            "atmosphereAltitude": 0.25
        },
        "markers": markers,
        "generated_at": datetime.now().isoformat(),
        "status": "success"
    }


def create_timeline_chart_config(
    time_series: List[Dict[str, Any]],
    metric: str = "pm25",
    title: str = "PM2.5 Timeline"
) -> Dict[str, Any]:
    """
    Generate configuration for timeline chart.
    
    Compatible with Chart.js and Plotly.
    
    Args:
        time_series: List of dicts with date, value, and optional annotations
        metric: Metric being displayed
        title: Chart title
    
    Returns:
        Chart configuration
    """
    labels = [item.get("date", "") for item in time_series]
    values = [item.get("value", 0) for item in time_series]
    
    # Find policy events for annotations
    annotations = []
    for item in time_series:
        if item.get("event"):
            annotations.append({
                "x": item.get("date"),
                "y": item.get("value"),
                "label": item.get("event")
            })
    
    return {
        "type": "line",
        "title": title,
        "config": {
            "labels": labels,
            "datasets": [{
                "label": metric.upper(),
                "data": values,
                "borderColor": "#3b82f6",
                "backgroundColor": "rgba(59, 130, 246, 0.1)",
                "fill": True,
                "tension": 0.4
            }],
            "annotations": annotations
        },
        "options": {
            "responsive": True,
            "scales": {
                "x": {"title": {"display": True, "text": "Date"}},
                "y": {"title": {"display": True, "text": f"{metric.upper()} (μg/m³)"}}
            }
        },
        "generated_at": datetime.now().isoformat(),
        "status": "success"
    }


def create_comparison_chart_config(
    countries_data: List[Dict[str, Any]],
    metric: str = "pm25",
    title: str = "Country Comparison"
) -> Dict[str, Any]:
    """
    Generate configuration for bar comparison chart.
    
    Args:
        countries_data: List of dicts with country and metric values
        metric: Metric being compared
        title: Chart title
    
    Returns:
        Chart configuration
    """
    labels = [item.get("country", "") for item in countries_data]
    values = [item.get(metric, 0) for item in countries_data]
    
    # Color based on values (lower is better for air quality)
    colors = []
    for v in values:
        if v <= 25:
            colors.append("#00e400")
        elif v <= 50:
            colors.append("#ffff00")
        elif v <= 75:
            colors.append("#ff7e00")
        else:
            colors.append("#ff0000")
    
    return {
        "type": "bar",
        "title": title,
        "config": {
            "labels": labels,
            "datasets": [{
                "label": metric.upper(),
                "data": values,
                "backgroundColor": colors,
                "borderColor": colors,
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "indexAxis": "y",
            "scales": {
                "x": {"title": {"display": True, "text": f"{metric.upper()} (μg/m³)"}}
            }
        },
        "generated_at": datetime.now().isoformat(),
        "status": "success"
    }


def create_before_after_chart_config(
    before_values: List[float],
    after_values: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Before vs After Policy"
) -> Dict[str, Any]:
    """
    Generate configuration for before/after comparison.
    
    Args:
        before_values: Values before policy
        after_values: Values after policy
        labels: Optional labels for x-axis
        title: Chart title
    
    Returns:
        Chart configuration
    """
    import statistics
    
    if labels is None:
        labels = [f"Point {i+1}" for i in range(len(before_values))]
    
    return {
        "type": "line",
        "title": title,
        "config": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Before Policy",
                    "data": before_values,
                    "borderColor": "#ef4444",
                    "backgroundColor": "rgba(239, 68, 68, 0.1)",
                    "fill": False
                },
                {
                    "label": "After Policy",
                    "data": after_values,
                    "borderColor": "#22c55e",
                    "backgroundColor": "rgba(34, 197, 94, 0.1)",
                    "fill": False
                }
            ]
        },
        "summary": {
            "before_mean": round(statistics.mean(before_values), 2),
            "after_mean": round(statistics.mean(after_values), 2),
            "improvement": statistics.mean(after_values) < statistics.mean(before_values)
        },
        "generated_at": datetime.now().isoformat(),
        "status": "success"
    }
