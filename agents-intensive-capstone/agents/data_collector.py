"""
Data Collection Agent
Responsible for gathering air quality data and policy information

Day 1 Concept: Specialized Agent in Multi-Agent System
"""

from typing import Dict, Any, List, Optional
from tools.waqi_tool import fetch_waqi_realtime_data, get_aqi_category
from tools.policy_db_tool import search_environmental_policies


class DataCollectorAgent:
    """
    Data Collection Agent - Part of Multi-Agent System (Day 1)
    
    Responsibilities:
    - Fetch real-time air quality data from WAQI API
    - Search environmental policy database
    - Collect historical data for analysis
    - Integrate with Google Search for policy news
    
    Tools Used (Day 2):
    - waqi_realtime_tool
    - policy_database_tool
    - google_search_tool (optional)
    """
    
    def __init__(self):
        self.name = "DataCollector"
        self.description = "Collects air quality data and policy information"
        self._cache: Dict[str, Any] = {}
    
    def collect_air_quality(
        self, 
        location: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Collect real-time air quality data.
        
        Args:
            location: City or country name
            use_cache: Whether to use cached data
        
        Returns:
            Air quality data dictionary
        """
        cache_key = f"aqi_{location}"
        
        if use_cache and cache_key in self._cache:
            data = self._cache[cache_key]
            data["from_cache"] = True
            return data
        
        data = fetch_waqi_realtime_data(location)
        
        if data.get("status") == "success":
            self._cache[cache_key] = data
        
        return data
    
    def collect_policies(
        self,
        country: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect environmental policies for a country.
        
        Args:
            country: Country name
            year_start: Filter start year
            year_end: Filter end year
        
        Returns:
            Policies data dictionary
        """
        return search_environmental_policies(
            country, 
            year_start, 
            year_end
        )
    
    def collect_multi_country_data(
        self,
        countries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Collect data for multiple countries.
        
        Args:
            countries: List of country names
        
        Returns:
            List of data for each country
        """
        results = []
        
        for country in countries:
            air_quality = self.collect_air_quality(country)
            policies = self.collect_policies(country)
            
            results.append({
                "country": country,
                "air_quality": air_quality,
                "policies": policies.get("policies", []),
                "policy_count": policies.get("count", 0)
            })
        
        return results
    
    def collect_historical_data(
        self,
        country: str,
        years: int = 5
    ) -> Dict[str, Any]:
        """
        Collect historical data (simulated for demo).
        
        Note: In production, this would call historical API endpoints.
        This is marked as a long-running operation (MCP Day 2).
        
        Args:
            country: Country name
            years: Number of years of history
        
        Returns:
            Historical data dictionary
        """
        # Simulated historical data
        # In production, this would require user approval (MCP)
        return {
            "country": country,
            "years": years,
            "data_points": years * 365,
            "note": "Historical data collection (simulated)",
            "requires_approval": True,
            "status": "success"
        }
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_items": len(self._cache),
            "cache_keys": list(self._cache.keys())
        }


# Create singleton instance for use in main.py
data_collector_agent = DataCollectorAgent()
