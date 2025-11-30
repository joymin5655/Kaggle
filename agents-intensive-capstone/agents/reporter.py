"""
Reporter Agent
Responsible for generating human-readable reports and insights

Day 1 Concept: Specialized Agent in Multi-Agent System
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class ReporterAgent:
    """
    Reporting Agent - Part of Multi-Agent System (Day 1)
    
    Responsibilities:
    - Generate executive summaries
    - Extract key insights from analysis
    - Provide policy recommendations
    - Create formatted reports
    
    Tools Used (Day 2):
    - report_generation_tool
    - insight_extraction_tool
    - recommendation_tool
    """
    
    def __init__(self):
        self.name = "Reporter"
        self.description = "Generates reports and insights from analysis"
    
    def generate_executive_summary(
        self,
        analysis_result: Dict[str, Any],
        language: str = "ko"
    ) -> str:
        """
        Generate executive summary from analysis.
        
        Args:
            analysis_result: Complete analysis result
            language: Output language (ko, en)
        
        Returns:
            Executive summary text
        """
        country = analysis_result.get("country", "Unknown")
        policy = analysis_result.get("policy", "Unknown Policy")
        comparison = analysis_result.get("analysis", {})
        significance = analysis_result.get("statistical_test", {})
        
        if language == "ko":
            return self._generate_korean_summary(
                country, policy, comparison, significance
            )
        else:
            return self._generate_english_summary(
                country, policy, comparison, significance
            )
    
    def _generate_korean_summary(
        self,
        country: str,
        policy: str,
        comparison: Dict,
        significance: Dict
    ) -> str:
        """Generate Korean executive summary."""
        lines = [
            f"📊 {country} 환경 정책 분석 결과",
            f"정책: {policy}",
            "",
        ]
        
        # Results
        before = comparison.get("before_mean", 0)
        after = comparison.get("after_mean", 0)
        change = comparison.get("percent_change", 0)
        
        lines.append("【주요 결과】")
        lines.append(f"• PM2.5 농도: {before:.1f} → {after:.1f} μg/m³")
        
        if comparison.get("improvement"):
            lines.append(f"• 변화율: ▼ {abs(change):.1f}% 감소 (개선)")
        else:
            lines.append(f"• 변화율: ▲ {abs(change):.1f}% 증가")
        
        # Statistical significance
        if significance.get("significant"):
            lines.append("• 통계적 유의성: ✅ 확인됨 (p < 0.05)")
        else:
            lines.append("• 통계적 유의성: ⚠️ 확인 필요")
        
        # Effect size
        effect = comparison.get("effect_size", "")
        effect_map = {
            "large": "매우 큼",
            "medium": "보통",
            "small": "작음",
            "negligible": "미미함"
        }
        lines.append(f"• 효과 크기: {effect_map.get(effect, effect)}")
        
        return "\n".join(lines)
    
    def _generate_english_summary(
        self,
        country: str,
        policy: str,
        comparison: Dict,
        significance: Dict
    ) -> str:
        """Generate English executive summary."""
        lines = [
            f"📊 {country} Environmental Policy Analysis",
            f"Policy: {policy}",
            "",
            "【Key Results】",
        ]
        
        before = comparison.get("before_mean", 0)
        after = comparison.get("after_mean", 0)
        change = comparison.get("percent_change", 0)
        
        lines.append(f"• PM2.5 Level: {before:.1f} → {after:.1f} μg/m³")
        
        if comparison.get("improvement"):
            lines.append(f"• Change: ▼ {abs(change):.1f}% reduction")
        else:
            lines.append(f"• Change: ▲ {abs(change):.1f}% increase")
        
        if significance.get("significant"):
            lines.append("• Statistical Significance: ✅ Confirmed (p < 0.05)")
        else:
            lines.append("• Statistical Significance: ⚠️ Not confirmed")
        
        return "\n".join(lines)
    
    def extract_insights(
        self,
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract key insights from analysis.
        
        Args:
            analysis_result: Complete analysis result
        
        Returns:
            List of insight dictionaries
        """
        insights = []
        comparison = analysis_result.get("analysis", {})
        
        # Improvement insight
        if comparison.get("improvement"):
            insights.append({
                "type": "positive",
                "title": "정책 효과 확인",
                "description": f"정책 시행 후 {abs(comparison.get('percent_change', 0)):.1f}% 개선",
                "importance": "high"
            })
        else:
            insights.append({
                "type": "warning",
                "title": "추가 조치 필요",
                "description": "정책 시행 후 개선 효과가 미미함",
                "importance": "high"
            })
        
        # Effect size insight
        effect = comparison.get("effect_size", "")
        if effect in ["medium", "large"]:
            insights.append({
                "type": "positive",
                "title": "실질적 영향 확인",
                "description": f"효과 크기 '{effect}'로 실질적 변화 있음",
                "importance": "medium"
            })
        
        return insights
    
    def generate_recommendations(
        self,
        analysis_result: Dict[str, Any]
    ) -> List[str]:
        """
        Generate policy recommendations.
        
        Args:
            analysis_result: Complete analysis result
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        comparison = analysis_result.get("analysis", {})
        
        if comparison.get("improvement"):
            recommendations.extend([
                "현재 정책 유지 및 강화 권장",
                "성공 요인 분석 후 타 지역 확대 적용 검토",
                "장기적 모니터링 체계 구축"
            ])
        else:
            recommendations.extend([
                "정책 효과성 재검토 필요",
                "추가적인 배출 감소 조치 도입 고려",
                "벤치마킹 대상국 정책 분석 권장"
            ])
        
        # Always include
        recommendations.append("인접국과의 환경 협력 강화")
        recommendations.append("데이터 기반 정책 의사결정 체계 구축")
        
        return recommendations
    
    def generate_full_report(
        self,
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete report with all sections.
        
        Args:
            analysis_result: Complete analysis result
        
        Returns:
            Full report dictionary
        """
        return {
            "title": f"{analysis_result.get('country', 'Unknown')} 환경 정책 분석 보고서",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": self.generate_executive_summary(analysis_result),
            "insights": self.extract_insights(analysis_result),
            "recommendations": self.generate_recommendations(analysis_result),
            "data_summary": {
                "country": analysis_result.get("country"),
                "policy": analysis_result.get("policy"),
                "analysis": analysis_result.get("analysis"),
                "significance": analysis_result.get("statistical_test")
            }
        }


# Create singleton instance
reporter_agent = ReporterAgent()
