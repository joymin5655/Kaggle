"""
Long-Term Memory
Persistent storage for analysis results and user data

Day 3 Concept: Long-Term Memory & Context Engineering
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib


class LongTermMemory:
    """
    Manages persistent long-term memory for the agent system.
    
    This implements Day 3 concept: Long-Term Memory
    
    Features:
    - Save analysis results across sessions
    - Store user preferences permanently
    - Track historical analyses for comparison
    - Search past results
    
    Example:
        >>> memory = LongTermMemory()
        >>> memory.save_analysis_result({"country": "Korea", "pm25": 24})
        >>> past = memory.get_past_analyses("Korea")
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Path to JSON storage file
        """
        if storage_path:
            self._storage_path = Path(storage_path)
        else:
            self._storage_path = Path(__file__).parent.parent / "data" / "memory.json"
        
        self._ensure_storage_exists()
        self._data = self._load_data()
    
    def _ensure_storage_exists(self) -> None:
        """Create storage file if it doesn't exist."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self._storage_path.exists():
            self._storage_path.write_text(json.dumps({
                "analyses": [],
                "preferences": {},
                "statistics": {
                    "total_analyses": 0,
                    "countries_analyzed": [],
                    "created_at": datetime.now().isoformat()
                }
            }, indent=2))
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from storage."""
        try:
            with open(self._storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"analyses": [], "preferences": {}, "statistics": {}}
    
    def _save_data(self) -> None:
        """Save data to storage."""
        with open(self._storage_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
    
    def _generate_id(self, data: Dict) -> str:
        """Generate unique ID for an entry."""
        content = json.dumps(data, sort_keys=True)
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:12]
    
    def save_analysis_result(self, result: Dict[str, Any]) -> str:
        """
        Save an analysis result to long-term memory.
        
        Args:
            result: Analysis result dictionary
        
        Returns:
            Generated ID for the saved result
        """
        analysis_id = self._generate_id(result)
        
        entry = {
            "id": analysis_id,
            "result": result,
            "saved_at": datetime.now().isoformat(),
            "country": result.get("country"),
            "policy": result.get("policy")
        }
        
        self._data["analyses"].append(entry)
        
        # Update statistics
        stats = self._data.get("statistics", {})
        stats["total_analyses"] = stats.get("total_analyses", 0) + 1
        
        countries = set(stats.get("countries_analyzed", []))
        if result.get("country"):
            countries.add(result["country"])
        stats["countries_analyzed"] = list(countries)
        stats["last_analysis_at"] = datetime.now().isoformat()
        
        self._data["statistics"] = stats
        self._save_data()
        
        return analysis_id
    
    def get_past_analyses(
        self, 
        country: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get past analysis results.
        
        Args:
            country: Filter by country (optional)
            limit: Maximum results to return
        
        Returns:
            List of past analyses
        """
        analyses = self._data.get("analyses", [])
        
        if country:
            analyses = [
                a for a in analyses 
                if a.get("country", "").lower() == country.lower()
            ]
        
        # Sort by date (newest first)
        analyses.sort(
            key=lambda x: x.get("saved_at", ""),
            reverse=True
        )
        
        return analyses[:limit]
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific analysis by ID."""
        for analysis in self._data.get("analyses", []):
            if analysis.get("id") == analysis_id:
                return analysis
        return None
    
    def save_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Save user preferences permanently.
        
        Args:
            preferences: Dictionary of preferences
        """
        self._data["preferences"].update(preferences)
        self._data["preferences"]["updated_at"] = datetime.now().isoformat()
        self._save_data()
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get saved user preferences."""
        return self._data.get("preferences", {}).copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self._data.get("statistics", {}).copy()
        stats["storage_path"] = str(self._storage_path)
        stats["storage_size_kb"] = round(
            self._storage_path.stat().st_size / 1024, 2
        ) if self._storage_path.exists() else 0
        
        return stats
    
    def search_analyses(
        self, 
        query: str,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search past analyses by keyword.
        
        Args:
            query: Search query
            fields: Fields to search in (default: all)
        
        Returns:
            Matching analyses
        """
        query_lower = query.lower()
        results = []
        
        for analysis in self._data.get("analyses", []):
            # Search in specified fields or all string values
            search_text = ""
            
            if fields:
                for field in fields:
                    value = analysis.get(field) or analysis.get("result", {}).get(field)
                    if value:
                        search_text += str(value) + " "
            else:
                search_text = json.dumps(analysis)
            
            if query_lower in search_text.lower():
                results.append(analysis)
        
        return results
    
    def compare_with_past(
        self, 
        current_result: Dict[str, Any],
        country: str
    ) -> Dict[str, Any]:
        """
        Compare current analysis with past results.
        
        Args:
            current_result: Current analysis result
            country: Country to compare
        
        Returns:
            Comparison data
        """
        past_analyses = self.get_past_analyses(country, limit=5)
        
        if not past_analyses:
            return {
                "comparison_available": False,
                "message": f"No past analyses found for {country}"
            }
        
        # Extract PM2.5 values
        current_pm25 = current_result.get("analysis", {}).get("after_mean")
        past_pm25_values = []
        
        for pa in past_analyses:
            pm25 = pa.get("result", {}).get("analysis", {}).get("after_mean")
            if pm25 is not None:
                past_pm25_values.append(pm25)
        
        if not past_pm25_values:
            return {
                "comparison_available": False,
                "message": "No PM2.5 data in past analyses"
            }
        
        avg_past = sum(past_pm25_values) / len(past_pm25_values)
        
        return {
            "comparison_available": True,
            "current_pm25": current_pm25,
            "average_past_pm25": round(avg_past, 2),
            "trend": "improving" if current_pm25 < avg_past else "worsening",
            "change_percent": round(
                (current_pm25 - avg_past) / avg_past * 100, 2
            ) if avg_past else 0,
            "past_analyses_count": len(past_analyses)
        }
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis by ID."""
        original_count = len(self._data.get("analyses", []))
        self._data["analyses"] = [
            a for a in self._data.get("analyses", [])
            if a.get("id") != analysis_id
        ]
        
        if len(self._data["analyses"]) < original_count:
            self._save_data()
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all stored data. Use with caution!"""
        self._data = {
            "analyses": [],
            "preferences": {},
            "statistics": {
                "total_analyses": 0,
                "countries_analyzed": [],
                "created_at": datetime.now().isoformat(),
                "cleared_at": datetime.now().isoformat()
            }
        }
        self._save_data()
