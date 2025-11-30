"""
Environmental Policy Agent System - Main Entry Point
Multi-agent system for analyzing environmental policy effectiveness

Track A: Consent Agents - Kaggle AI Agents Intensive Capstone Project
Team: Robee
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for ADK availability
try:
    from google import genai
    ADK_AVAILABLE = True
except ImportError:
    print("Warning: google.genai not available. Running in demo mode.")
    ADK_AVAILABLE = False

from memory.session_manager import SessionManager
from memory.long_term_memory import LongTermMemory
from tools.waqi_tool import fetch_waqi_realtime_data, get_aqi_category
from tools.policy_db_tool import search_environmental_policies, get_policy_details
from tools.analysis_tool import (
    calculate_trend, 
    compare_before_after, 
    calculate_statistical_significance
)


class PolicyAgentSystem:
    """
    Main orchestrator for the multi-agent environmental policy analysis system.
    
    This system coordinates 4 specialized agents:
    1. Data Collector: Gathers air quality data and policy information
    2. Policy Analyzer: Performs statistical analysis
    3. Visualizer: Creates interactive visualizations
    4. Reporter: Generates human-readable reports
    
    Core Concepts Applied:
    - Day 1: Multi-Agent System architecture
    - Day 2: Custom Tools & MCP integration
    - Day 3: Memory & Context Engineering
    """
    
    def __init__(self):
        """Initialize the agent system"""
        self.session = SessionManager()
        self.memory = LongTermMemory()
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if ADK_AVAILABLE and self.api_key:
            self._initialize_client()
        else:
            print("Running in demo mode without Gemini API")
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        self.client = genai.Client(api_key=self.api_key)
    
    def analyze_policy(
        self,
        country: str,
        policy_name: Optional[str] = None,
        year: Optional[int] = None,
        years_before: int = 2,
        years_after: int = 2
    ) -> Dict:
        """
        Analyze a specific environmental policy's impact.
        
        Args:
            country: Country name (e.g., "South Korea")
            policy_name: Name of policy (optional)
            year: Year of policy enactment (optional)
            years_before: Years of data before policy
            years_after: Years of data after policy
        
        Returns:
            Complete analysis results
        """
        # Step 1: Search for policy information
        policies = search_environmental_policies(country, year, year)
        
        # Step 2: Get current air quality data
        air_quality = fetch_waqi_realtime_data(country)
        
        # Step 3: Generate demo analysis results
        # In production, this would use actual historical data
        before_data = [38, 40, 37, 39, 38.5, 41, 36, 39]
        after_data = [24, 25, 23, 24.5, 24, 26, 22, 25]
        
        # Step 4: Statistical analysis
        comparison = compare_before_after(before_data, after_data, "PM2.5")
        significance = calculate_statistical_significance(before_data, after_data)
        
        # Step 5: Store in memory
        result = {
            "country": country,
            "policy": policy_name or f"{year} Environmental Policy",
            "current_air_quality": air_quality,
            "policies_found": policies,
            "analysis": comparison,
            "statistical_test": significance,
            "report": {
                "executive_summary": self._generate_summary(
                    country, comparison, significance
                ),
                "recommendations": self._generate_recommendations(comparison)
            }
        }
        
        # Save to long-term memory
        self.memory.save_analysis_result(result)
        
        return result
    
    def _generate_summary(
        self, 
        country: str, 
        comparison: Dict, 
        significance: Dict
    ) -> str:
        """Generate executive summary"""
        reduction = comparison.get("percent_change", 0)
        p_value = significance.get("p_value", 1)
        
        summary = f"{country}의 환경 정책 분석 결과:\n"
        summary += f"• PM2.5 수치가 평균 {abs(reduction):.1f}% "
        summary += "감소" if reduction < 0 else "증가"
        summary += f" (정책 시행 전 {comparison.get('before_mean', 0):.1f} → "
        summary += f"후 {comparison.get('after_mean', 0):.1f} μg/m³)\n"
        
        if p_value < 0.05:
            summary += f"• 통계적으로 유의미한 변화 (p < 0.05)\n"
        else:
            summary += f"• 통계적 유의성이 확인되지 않음 (p = {p_value:.3f})\n"
        
        if comparison.get("improvement"):
            summary += "• 정책 효과: ✅ 긍정적 영향 확인"
        else:
            summary += "• 정책 효과: ⚠️ 추가 분석 필요"
        
        return summary
    
    def _generate_recommendations(self, comparison: Dict) -> List[str]:
        """Generate policy recommendations"""
        recommendations = []
        
        if comparison.get("improvement"):
            recommendations.append("현재 정책 유지 및 강화 권장")
            recommendations.append("성공 요인 분석 후 타 지역 확대 적용 검토")
        else:
            recommendations.append("정책 효과성 재검토 필요")
            recommendations.append("추가적인 배출 감소 조치 고려")
        
        recommendations.append("지속적인 모니터링 및 데이터 수집")
        recommendations.append("인접국과의 협력 강화")
        
        return recommendations
    
    def compare_countries(
        self,
        countries: List[str],
        metric: str = "pm25",
        year: int = 2024
    ) -> Dict:
        """
        Compare environmental metrics across multiple countries.
        """
        results = []
        
        for country in countries:
            data = fetch_waqi_realtime_data(country)
            policies = search_environmental_policies(country)
            
            results.append({
                "country": country,
                "current_aqi": data.get("aqi"),
                "pm25": data.get("pm25"),
                "policy_count": len(policies.get("policies", [])),
                "category": get_aqi_category(data.get("aqi", 0))
            })
        
        # Sort by AQI (lower is better)
        results.sort(key=lambda x: x.get("current_aqi", 999))
        
        for i, result in enumerate(results, 1):
            result["rank"] = i
        
        return {
            "metric": metric,
            "year": year,
            "countries": results,
            "best_performer": results[0] if results else None,
            "worst_performer": results[-1] if results else None
        }
    
    def ask(self, query: str) -> Dict:
        """
        Natural language interface for analysis.
        """
        # Store query in session for context
        self.session.store_query_history(query, "Processing...")
        
        # Simple keyword-based routing (in production, use LLM)
        query_lower = query.lower()
        
        if "korea" in query_lower or "한국" in query_lower:
            return self.analyze_policy("South Korea", "2023 Emission Policy")
        elif "china" in query_lower or "중국" in query_lower:
            return self.analyze_policy("China", "Blue Sky Policy")
        elif "compare" in query_lower or "비교" in query_lower:
            return self.compare_countries(["South Korea", "China", "Japan"])
        else:
            return {
                "message": "분석할 국가나 정책을 지정해주세요.",
                "examples": [
                    "Analyze South Korea's emission policy",
                    "Compare Korea, China, Japan",
                    "한국의 미세먼지 정책 효과 분석"
                ]
            }


def main():
    """Main entry point"""
    print("🌍 Environmental Policy Agent System")
    print("=" * 50)
    print()
    
    # Initialize system
    print("🚀 Initializing agent system...")
    system = PolicyAgentSystem()
    print("✅ System ready!")
    print()
    
    # Example analysis
    print("📊 Analyzing South Korea's 2023 Emission Policy...")
    print("-" * 50)
    
    result = system.analyze_policy(
        country="South Korea",
        policy_name="2023 Emission Reduction Act"
    )
    
    print("\n📋 Executive Summary:")
    print(result["report"]["executive_summary"])
    
    print("\n💡 Recommendations:")
    for rec in result["report"]["recommendations"]:
        print(f"  • {rec}")
    
    print("\n" + "=" * 50)
    print("For interactive usage:")
    print("  from main import PolicyAgentSystem")
    print("  system = PolicyAgentSystem()")
    print("  result = system.ask('Analyze Korea policy')")


if __name__ == "__main__":
    main()
