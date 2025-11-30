"""
Configuration file for Environmental Policy Agent System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """System configuration"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    WAQI_API_KEY = os.getenv("WAQI_API_KEY")
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    # System Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
    MAX_API_CALLS_PER_DAY = int(os.getenv("MAX_API_CALLS_PER_DAY", "1000"))
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = DATA_DIR / "cache"
    MEMORY_STORAGE_PATH = os.getenv(
        "MEMORY_STORAGE_PATH", 
        str(DATA_DIR / "memory.json")
    )
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Gemini Model Settings
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    GEMINI_TEMPERATURE = 0.7
    GEMINI_MAX_TOKENS = 8192
    
    # Agent Settings
    AGENT_TIMEOUT_SECONDS = 300  # 5 minutes
    MAX_TOOL_CALLS = 20
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        warnings = []
        
        if not cls.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY")
        
        if not cls.WAQI_API_KEY:
            warnings.append("WAQI_API_KEY")
        
        if warnings:
            print(f"⚠️  Warning: Missing config: {', '.join(warnings)}")
            print("   Please set in .env file")
            return False
        
        return True


# Validate on import
Config.validate()
