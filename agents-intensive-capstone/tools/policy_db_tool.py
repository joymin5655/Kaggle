"""
Policy Database Tool
Search and retrieve environmental policy information

Day 2 Concept: Custom Tool Implementation
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Data path
DATA_DIR = Path(__file__).parent.parent / "data"
POLICIES_FILE = DATA_DIR / "policies.json"


def _load_policies() -> List[Dict]:
    """Load policies from JSON file."""
    if not POLICIES_FILE.exists():
        return []
    
    try:
        with open(POLICIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def search_environmental_policies(
    country: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    policy_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search environmental policies in the database.
    
    This is a custom tool for the Data Collection Agent.
    
    Args:
        country: Country name (e.g., "South Korea", "China")
        year_start: Filter policies enacted after this year
        year_end: Filter policies enacted before this year
        policy_type: Filter by policy type (emission_reduction, carbon_reduction)
    
    Returns:
        Dictionary with matching policies
    
    Example:
        >>> policies = search_environmental_policies("South Korea", 2019, 2024)
        >>> print(len(policies["policies"]))
        2
    """
    all_policies = _load_policies()
    
    # Filter by country
    policies = [
        p for p in all_policies
        if p.get("country", "").lower() == country.lower()
    ]
    
    # Filter by year range
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
    
    # Filter by type
    if policy_type:
        policies = [
            p for p in policies
            if p.get("type") == policy_type
        ]
    
    return {
        "country": country,
        "policies": policies,
        "count": len(policies),
        "filters": {
            "year_start": year_start,
            "year_end": year_end,
            "policy_type": policy_type
        },
        "status": "success"
    }


def get_policy_details(policy_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific policy.
    
    Args:
        policy_id: Unique policy identifier
    
    Returns:
        Policy details or error
    """
    all_policies = _load_policies()
    
    for policy in all_policies:
        if policy.get("id") == policy_id:
            return {
                "policy": policy,
                "status": "success"
            }
    
    return {
        "error": f"Policy not found: {policy_id}",
        "status": "error"
    }


def get_policies_by_effectiveness(
    min_reduction: float = 0
) -> Dict[str, Any]:
    """
    Get policies sorted by effectiveness.
    
    Args:
        min_reduction: Minimum reduction percentage to include
    
    Returns:
        List of effective policies
    """
    all_policies = _load_policies()
    
    # Filter completed policies with actual reduction data
    effective_policies = [
        p for p in all_policies
        if p.get("status") == "completed" 
        and p.get("actual_reduction")
    ]
    
    # Parse reduction percentage
    for p in effective_policies:
        reduction_str = p.get("actual_reduction", "0%")
        p["reduction_value"] = float(reduction_str.replace("%", ""))
    
    # Filter by minimum reduction
    effective_policies = [
        p for p in effective_policies
        if p["reduction_value"] >= min_reduction
    ]
    
    # Sort by effectiveness
    effective_policies.sort(
        key=lambda x: x["reduction_value"], 
        reverse=True
    )
    
    return {
        "policies": effective_policies,
        "count": len(effective_policies),
        "status": "success"
    }


def compare_country_policies(
    countries: List[str]
) -> Dict[str, Any]:
    """
    Compare environmental policies across countries.
    
    Args:
        countries: List of country names
    
    Returns:
        Comparison data
    """
    comparison = []
    
    for country in countries:
        result = search_environmental_policies(country)
        policies = result.get("policies", [])
        
        # Calculate summary stats
        total_budget = sum(
            p.get("budget_usd", 0) for p in policies
        )
        active_count = sum(
            1 for p in policies if p.get("status") == "active"
        )
        completed_count = sum(
            1 for p in policies if p.get("status") == "completed"
        )
        
        comparison.append({
            "country": country,
            "total_policies": len(policies),
            "active_policies": active_count,
            "completed_policies": completed_count,
            "total_budget_usd": total_budget,
            "policies": policies
        })
    
    return {
        "comparison": comparison,
        "countries": countries,
        "status": "success"
    }
