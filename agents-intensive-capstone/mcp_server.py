"""
Environmental Policy MCP Server
Claude Desktop Integration using Model Context Protocol

Day 2 Concept: MCP & Tool Interoperability
"""

import os
import json
from pathlib import Path
from typing import Any
from datetime import datetime

# Try to import MCP - may not be installed
try:
    from mcp.server.fastmcp import FastMCP
    import httpx
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Run: pip install mcp fastmcp")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MCP server if available
if MCP_AVAILABLE:
    mcp = FastMCP("environmental-policy-server")
else:
    mcp = None

# Constants
WAQI_API_KEY = os.getenv("WAQI_API_KEY")
WAQI_BASE_URL = "https://api.waqi.info"
DATA_DIR = Path(__file__).parent / "data"
POLICIES_FILE = DATA_DIR / "policies.json"

# City mapping
CITY_MAPPING = {
    "South Korea": "seoul",
    "Korea": "seoul",
    "China": "beijing",
    "Japan": "tokyo",
    "India": "delhi",
}


def get_aqi_category(aqi: int) -> str:
    """Get AQI category."""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"


def get_aqi_color(aqi: int) -> str:
    """Get AQI color."""
    if aqi <= 50: return "#00e400"
    elif aqi <= 100: return "#ffff00"
    elif aqi <= 150: return "#ff7e00"
    elif aqi <= 200: return "#ff0000"
    elif aqi <= 300: return "#8f3f97"
    else: return "#7e0023"


if MCP_AVAILABLE:
    
    @mcp.tool()
    async def get_realtime_air_quality(city: str) -> dict[str, Any]:
        """
        특정 도시의 실시간 대기질 데이터를 가져옵니다.
        
        Args:
            city: 도시 이름 (예: "Seoul", "Beijing", "Tokyo")
        
        Returns:
            대기질 정보 (AQI, PM2.5 등)
        """
        # Map city name
        mapped_city = CITY_MAPPING.get(city, city.lower())
        
        if not WAQI_API_KEY:
            # Demo data
            aqi = 45
            return {
                "city": city,
                "aqi": aqi,
                "pm25": 28,
                "category": get_aqi_category(aqi),
                "color": get_aqi_color(aqi),
                "source": "Demo",
                "status": "success"
            }
        
        url = f"{WAQI_BASE_URL}/feed/{mapped_city}/?token={WAQI_API_KEY}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "ok":
                    aqi_data = data["data"]
                    aqi = aqi_data.get("aqi", 0)
                    
                    return {
                        "city": city,
                        "aqi": aqi,
                        "pm25": aqi_data.get("iaqi", {}).get("pm25", {}).get("v"),
                        "pm10": aqi_data.get("iaqi", {}).get("pm10", {}).get("v"),
                        "timestamp": aqi_data.get("time", {}).get("s"),
                        "category": get_aqi_category(aqi),
                        "color": get_aqi_color(aqi),
                        "status": "success"
                    }
                else:
                    return {"error": f"API error: {data.get('status')}", "status": "error"}
                    
            except Exception as e:
                return {"error": str(e), "status": "error"}
    
    
    @mcp.tool()
    async def search_environmental_policies(
        country: str,
        year_start: int = None,
        year_end: int = None
    ) -> dict[str, Any]:
        """
        특정 국가의 환경 정책을 검색합니다.
        
        Args:
            country: 국가 이름 (예: "South Korea", "China")
            year_start: 시작 연도 (선택)
            year_end: 종료 연도 (선택)
        
        Returns:
            정책 목록
        """
        # Load policies
        if not POLICIES_FILE.exists():
            return {"error": "Policy database not found", "policies": []}
        
        with open(POLICIES_FILE, 'r', encoding='utf-8') as f:
            all_policies = json.load(f)
        
        # Filter
        policies = [
            p for p in all_policies
            if p.get("country", "").lower() == country.lower()
        ]
        
        if year_start:
            policies = [
                p for p in policies
                if int(p.get("enacted_date", "0000")[:4]) >= year_start
            ]
        
        if year_end:
            policies = [
                p for p in policies
                if int(p.get("enacted_date", "0000")[:4]) <= year_end
            ]
        
        return {
            "country": country,
            "policies": policies,
            "count": len(policies),
            "status": "success"
        }
    
    
    @mcp.tool()
    async def analyze_policy_effectiveness(
        before_values: list[float],
        after_values: list[float]
    ) -> dict[str, Any]:
        """
        정책 전후 데이터를 비교 분석합니다.
        
        Args:
            before_values: 정책 이전 측정값 리스트
            after_values: 정책 이후 측정값 리스트
        
        Returns:
            분석 결과 (평균, 변화율, 효과성)
        """
        import statistics
        
        if not before_values or not after_values:
            return {"error": "Need data for both periods", "status": "error"}
        
        before_mean = statistics.mean(before_values)
        after_mean = statistics.mean(after_values)
        
        reduction = before_mean - after_mean
        reduction_percent = (reduction / before_mean) * 100 if before_mean > 0 else 0
        
        # Effectiveness interpretation
        if reduction_percent > 30:
            effectiveness = "highly effective"
        elif reduction_percent > 15:
            effectiveness = "moderately effective"
        elif reduction_percent > 5:
            effectiveness = "somewhat effective"
        else:
            effectiveness = "minimal effect"
        
        return {
            "before_mean": round(before_mean, 2),
            "after_mean": round(after_mean, 2),
            "reduction": round(reduction, 2),
            "reduction_percent": round(reduction_percent, 2),
            "effectiveness": effectiveness,
            "improvement": after_mean < before_mean,
            "status": "success"
        }
    
    
    @mcp.tool()
    async def compare_countries(
        countries: list[str]
    ) -> dict[str, Any]:
        """
        여러 국가의 대기질을 비교합니다.
        
        Args:
            countries: 비교할 국가 목록
        
        Returns:
            국가별 비교 데이터
        """
        results = []
        
        for country in countries:
            data = await get_realtime_air_quality(country)
            
            if data.get("status") == "success":
                results.append({
                    "country": country,
                    "aqi": data.get("aqi"),
                    "pm25": data.get("pm25"),
                    "category": data.get("category")
                })
        
        # Sort by AQI
        results.sort(key=lambda x: x.get("aqi", 999))
        
        for i, result in enumerate(results, 1):
            result["rank"] = i
        
        return {
            "comparison": results,
            "best": results[0] if results else None,
            "worst": results[-1] if results else None,
            "status": "success"
        }
    
    
    # Prompts for guided analysis
    @mcp.prompt()
    def comprehensive_analysis_prompt(country: str) -> str:
        """환경 정책 종합 분석 프롬프트"""
        return f"""
Please conduct a comprehensive environmental policy analysis for {country}.

Steps:
1. Get current air quality using get_realtime_air_quality
2. Search for policies using search_environmental_policies
3. Analyze policy effectiveness if data available
4. Provide recommendations

Format your response as:
- Current Status
- Policy Review
- Effectiveness Assessment
- Recommendations
"""
    
    
    # Resource for policy list
    @mcp.resource("policies://list")
    def get_all_policies() -> str:
        """모든 정책 목록 리소스"""
        if not POLICIES_FILE.exists():
            return json.dumps({"error": "No policies found"})
        
        with open(POLICIES_FILE, 'r', encoding='utf-8') as f:
            policies = json.load(f)
        
        return json.dumps(policies, indent=2, ensure_ascii=False)


def main():
    """Run MCP server"""
    if not MCP_AVAILABLE:
        print("❌ MCP not available. Install with:")
        print("   pip install mcp fastmcp httpx python-dotenv")
        return
    
    print("🌍 Environmental Policy MCP Server")
    print("=" * 50)
    print()
    print("Available tools:")
    print("  • get_realtime_air_quality - 실시간 대기질 조회")
    print("  • search_environmental_policies - 정책 검색")
    print("  • analyze_policy_effectiveness - 효과성 분석")
    print("  • compare_countries - 국가 비교")
    print()
    print("Starting server...")
    
    mcp.run()


if __name__ == "__main__":
    main()
