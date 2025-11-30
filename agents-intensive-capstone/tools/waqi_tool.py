"""
WAQI (World Air Quality Index) Tool
Fetches real-time air quality data from WAQI API

Day 2 Concept: Custom Tool Implementation
"""

import os
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

# API Configuration
WAQI_API_KEY = os.getenv("WAQI_API_KEY")
WAQI_BASE_URL = "https://api.waqi.info"

# City mapping for common names
CITY_MAPPING = {
    "South Korea": "seoul",
    "Korea": "seoul",
    "한국": "seoul",
    "China": "beijing",
    "중국": "beijing",
    "Japan": "tokyo",
    "일본": "tokyo",
    "India": "delhi",
    "인도": "delhi",
    "USA": "new york",
    "미국": "new york",
}


def get_aqi_color(aqi: int) -> str:
    """
    Get color code based on AQI value (EPA standard).
    
    Args:
        aqi: Air Quality Index value
    
    Returns:
        Hex color code
    """
    if aqi <= 50:
        return "#00e400"  # Good - Green
    elif aqi <= 100:
        return "#ffff00"  # Moderate - Yellow
    elif aqi <= 150:
        return "#ff7e00"  # Unhealthy for Sensitive - Orange
    elif aqi <= 200:
        return "#ff0000"  # Unhealthy - Red
    elif aqi <= 300:
        return "#8f3f97"  # Very Unhealthy - Purple
    else:
        return "#7e0023"  # Hazardous - Maroon


def get_aqi_category(aqi: int) -> str:
    """
    Get AQI category label.
    
    Args:
        aqi: Air Quality Index value
    
    Returns:
        Category string
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def fetch_waqi_realtime_data(
    location: str,
    pollutant: str = "pm25"
) -> Dict[str, Any]:
    """
    Fetch real-time air quality data from WAQI API.
    
    This is a custom tool for the Data Collection Agent.
    
    Args:
        location: City or country name
        pollutant: Pollutant type (pm25, pm10, o3, no2, so2, co)
    
    Returns:
        Dictionary with air quality data
    
    Example:
        >>> data = fetch_waqi_realtime_data("Seoul")
        >>> print(data["aqi"])
        45
    """
    # Map location to city
    city = CITY_MAPPING.get(location, location.lower())
    
    # Check for API key
    if not WAQI_API_KEY:
        # Return demo data if no API key
        return _get_demo_data(location)
    
    # API call
    url = f"{WAQI_BASE_URL}/feed/{city}/?token={WAQI_API_KEY}"
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                aqi_data = data["data"]
                aqi = aqi_data.get("aqi", 0)
                
                return {
                    "location": location,
                    "city": city,
                    "aqi": aqi,
                    "pm25": aqi_data.get("iaqi", {}).get("pm25", {}).get("v"),
                    "pm10": aqi_data.get("iaqi", {}).get("pm10", {}).get("v"),
                    "o3": aqi_data.get("iaqi", {}).get("o3", {}).get("v"),
                    "no2": aqi_data.get("iaqi", {}).get("no2", {}).get("v"),
                    "so2": aqi_data.get("iaqi", {}).get("so2", {}).get("v"),
                    "co": aqi_data.get("iaqi", {}).get("co", {}).get("v"),
                    "temperature": aqi_data.get("iaqi", {}).get("t", {}).get("v"),
                    "humidity": aqi_data.get("iaqi", {}).get("h", {}).get("v"),
                    "timestamp": aqi_data.get("time", {}).get("s"),
                    "category": get_aqi_category(aqi),
                    "color": get_aqi_color(aqi),
                    "source": "WAQI API",
                    "status": "success"
                }
            else:
                return {
                    "error": f"API returned status: {data.get('status')}",
                    "location": location,
                    "status": "error"
                }
                
    except httpx.TimeoutException:
        return {"error": "Request timeout", "status": "error"}
    except httpx.HTTPError as e:
        return {"error": str(e), "status": "error"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


def _get_demo_data(location: str) -> Dict[str, Any]:
    """Return demo data when API key is not available."""
    demo_data = {
        "South Korea": {"aqi": 45, "pm25": 28, "pm10": 45},
        "China": {"aqi": 125, "pm25": 78, "pm10": 120},
        "Japan": {"aqi": 35, "pm25": 18, "pm10": 30},
        "India": {"aqi": 185, "pm25": 135, "pm10": 180},
    }
    
    data = demo_data.get(location, {"aqi": 50, "pm25": 30, "pm10": 50})
    aqi = data["aqi"]
    
    return {
        "location": location,
        "city": CITY_MAPPING.get(location, location.lower()),
        "aqi": aqi,
        "pm25": data["pm25"],
        "pm10": data["pm10"],
        "o3": None,
        "no2": None,
        "so2": None,
        "co": None,
        "temperature": 20,
        "humidity": 60,
        "timestamp": datetime.now().isoformat(),
        "category": get_aqi_category(aqi),
        "color": get_aqi_color(aqi),
        "source": "Demo Data",
        "status": "success"
    }


async def fetch_waqi_realtime_data_async(
    location: str,
    pollutant: str = "pm25"
) -> Dict[str, Any]:
    """
    Async version of fetch_waqi_realtime_data.
    For use in MCP server.
    """
    city = CITY_MAPPING.get(location, location.lower())
    
    if not WAQI_API_KEY:
        return _get_demo_data(location)
    
    url = f"{WAQI_BASE_URL}/feed/{city}/?token={WAQI_API_KEY}"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "ok":
                aqi_data = data["data"]
                aqi = aqi_data.get("aqi", 0)
                
                return {
                    "location": location,
                    "aqi": aqi,
                    "pm25": aqi_data.get("iaqi", {}).get("pm25", {}).get("v"),
                    "category": get_aqi_category(aqi),
                    "color": get_aqi_color(aqi),
                    "status": "success"
                }
            else:
                return {"error": "API error", "status": "error"}
        except Exception as e:
            return {"error": str(e), "status": "error"}
