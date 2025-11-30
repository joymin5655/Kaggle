"""
Session Manager
Manages conversation context and short-term memory

Day 3 Concept: Session Memory & Context Engineering
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque


class SessionManager:
    """
    Manages session-level memory for maintaining conversation context.
    
    This implements Day 3 concept: Session Memory
    
    Features:
    - Store user preferences within a session
    - Track query history
    - Maintain conversation context
    - Remember analysis parameters
    
    Example:
        >>> session = SessionManager()
        >>> session.set_favorite_countries(["South Korea", "Japan"])
        >>> session.store_user_preferences({"language": "ko"})
        >>> favorites = session.get_favorite_countries()
        ['South Korea', 'Japan']
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize session manager.
        
        Args:
            max_history: Maximum number of queries to remember
        """
        self._preferences: Dict[str, Any] = {}
        self._query_history: deque = deque(maxlen=max_history)
        self._context: Dict[str, Any] = {}
        self._favorite_countries: List[str] = []
        self._analysis_cache: Dict[str, Any] = {}
        self._session_start = datetime.now()
    
    def store_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Store user preferences for the session.
        
        Args:
            preferences: Dictionary of preferences
        """
        self._preferences.update(preferences)
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get all stored preferences."""
        return self._preferences.copy()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference."""
        return self._preferences.get(key, default)
    
    def set_favorite_countries(self, countries: List[str]) -> None:
        """Set user's favorite countries for quick access."""
        self._favorite_countries = countries
        self._preferences["favorite_countries"] = countries
    
    def get_favorite_countries(self) -> List[str]:
        """Get user's favorite countries."""
        return self._favorite_countries.copy()
    
    def store_query_history(
        self, 
        query: str, 
        result: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Store a query and its result in history.
        
        Args:
            query: User's query
            result: System's response
            metadata: Optional additional data
        """
        entry = {
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._query_history.append(entry)
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent query history.
        
        Args:
            limit: Maximum entries to return
        
        Returns:
            List of query entries
        """
        history = list(self._query_history)
        return history[-limit:] if limit else history
    
    def get_last_query(self) -> Optional[Dict]:
        """Get the most recent query."""
        if self._query_history:
            return self._query_history[-1]
        return None
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set context information for the conversation.
        
        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
    
    def get_context(self, key: str) -> Optional[Any]:
        """Get context information."""
        ctx = self._context.get(key)
        return ctx["value"] if ctx else None
    
    def get_all_context(self) -> Dict[str, Any]:
        """Get all context information."""
        return {k: v["value"] for k, v in self._context.items()}
    
    def cache_analysis(self, key: str, result: Any) -> None:
        """Cache an analysis result for quick retrieval."""
        self._analysis_cache[key] = {
            "result": result,
            "cached_at": datetime.now().isoformat()
        }
    
    def get_cached_analysis(self, key: str) -> Optional[Any]:
        """Get cached analysis result."""
        cached = self._analysis_cache.get(key)
        return cached["result"] if cached else None
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of the current session.
        
        Returns:
            Session summary including duration, query count, etc.
        """
        now = datetime.now()
        duration = now - self._session_start
        
        return {
            "session_start": self._session_start.isoformat(),
            "duration_minutes": round(duration.total_seconds() / 60, 2),
            "query_count": len(self._query_history),
            "preferences_count": len(self._preferences),
            "favorite_countries": self._favorite_countries,
            "context_keys": list(self._context.keys()),
            "cache_size": len(self._analysis_cache)
        }
    
    def reset(self) -> None:
        """Reset the session."""
        self._preferences.clear()
        self._query_history.clear()
        self._context.clear()
        self._favorite_countries.clear()
        self._analysis_cache.clear()
        self._session_start = datetime.now()
