"""
Deployment Configuration - Production deployment settings
Day 5: Prototype to Production
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_seconds: int = 60

@dataclass
class RetryConfig:
    """Retry configuration for resilience."""
    max_retries: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    retryable_errors: List[str] = field(default_factory=lambda: [
        "RateLimitError", "TimeoutError", "ServiceUnavailable"
    ])

@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    environment: Environment
    service_name: str = "policy-agent-system"
    version: str = "1.0.0"
    
    # API Configuration
    api_rate_limit: int = 100  # requests per minute
    api_timeout_seconds: int = 30
    
    # Scaling
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    
    # Retry
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    # Feature flags
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_logging: bool = True
    enable_tracing: bool = True
    enable_metrics: bool = True
    
    # Security
    require_auth: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def for_environment(cls, env: str) -> "DeploymentConfig":
        """Create config for specific environment."""
        environment = Environment(env.lower())
        
        if environment == Environment.DEVELOPMENT:
            return cls(
                environment=environment,
                api_rate_limit=1000,
                scaling=ScalingConfig(min_instances=1, max_instances=2),
                require_auth=False,
                enable_tracing=True
            )
        elif environment == Environment.STAGING:
            return cls(
                environment=environment,
                api_rate_limit=500,
                scaling=ScalingConfig(min_instances=2, max_instances=5),
                require_auth=True
            )
        else:  # PRODUCTION
            return cls(
                environment=environment,
                api_rate_limit=100,
                scaling=ScalingConfig(min_instances=3, max_instances=10),
                require_auth=True,
                allowed_origins=["https://policy-agent.example.com"]
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "service_name": self.service_name,
            "version": self.version,
            "api_rate_limit": self.api_rate_limit,
            "api_timeout_seconds": self.api_timeout_seconds,
            "scaling": {
                "min_instances": self.scaling.min_instances,
                "max_instances": self.scaling.max_instances,
                "target_cpu_utilization": self.scaling.target_cpu_utilization
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "initial_delay_ms": self.retry.initial_delay_ms
            },
            "features": {
                "caching": self.enable_caching,
                "logging": self.enable_logging,
                "tracing": self.enable_tracing,
                "metrics": self.enable_metrics
            },
            "security": {
                "require_auth": self.require_auth,
                "allowed_origins": self.allowed_origins
            }
        }
    
    def export_yaml(self, filepath: str):
        """Export config as YAML (for Kubernetes/Cloud Run)."""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def export_json(self, filepath: str):
        """Export config as JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Pre-configured deployments
DEPLOYMENT_CONFIGS = {
    "vertex_ai": DeploymentConfig(
        environment=Environment.PRODUCTION,
        service_name="policy-agent-vertex",
        scaling=ScalingConfig(min_instances=1, max_instances=5)
    ),
    "cloud_run": DeploymentConfig(
        environment=Environment.PRODUCTION,
        service_name="policy-agent-cloudrun",
        scaling=ScalingConfig(min_instances=0, max_instances=100)
    ),
    "local": DeploymentConfig(
        environment=Environment.DEVELOPMENT,
        require_auth=False,
        enable_tracing=True
    )
}
