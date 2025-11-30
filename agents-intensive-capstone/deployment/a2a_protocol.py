"""
Agent2Agent (A2A) Protocol - Inter-agent communication
Day 5: Prototype to Production

A2A Protocol enables agents to:
- Discover other agents' capabilities
- Communicate regardless of model/framework
- Delegate tasks across organizational boundaries
"""
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class AgentCard:
    """Agent Card - describes an agent's capabilities (A2A spec)."""
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "endpoint": self.endpoint,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentCard":
        return cls(**data)


@dataclass
class A2AMessage:
    """Message format for A2A communication."""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }

class A2AProtocol:
    """A2A Protocol implementation for multi-agent communication."""
    
    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self.registered_agents: Dict[str, AgentCard] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: List[A2AMessage] = []
    
    def register_agent(self, card: AgentCard):
        """Register a remote agent."""
        self.registered_agents[card.agent_id] = card
    
    def discover_agents(self, capability: Optional[str] = None) -> List[AgentCard]:
        """Discover agents, optionally filtered by capability."""
        agents = list(self.registered_agents.values())
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        return agents
    
    def create_request(
        self,
        recipient_id: str,
        action: str,
        data: Dict[str, Any]
    ) -> A2AMessage:
        """Create a request message."""
        return A2AMessage(
            message_id=str(uuid.uuid4())[:8],
            message_type=MessageType.REQUEST,
            sender_id=self.agent_card.agent_id,
            recipient_id=recipient_id,
            payload={"action": action, "data": data}
        )

    def create_response(
        self,
        request: A2AMessage,
        result: Dict[str, Any],
        success: bool = True
    ) -> A2AMessage:
        """Create a response to a request."""
        msg_type = MessageType.RESPONSE if success else MessageType.ERROR
        return A2AMessage(
            message_id=str(uuid.uuid4())[:8],
            message_type=msg_type,
            sender_id=self.agent_card.agent_id,
            recipient_id=request.sender_id,
            payload={"result": result, "success": success},
            correlation_id=request.message_id
        )
    
    def register_handler(self, action: str, handler: Callable):
        """Register a handler for an action type."""
        self.message_handlers[action] = handler
    
    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle an incoming message."""
        if message.message_type == MessageType.REQUEST:
            action = message.payload.get("action")
            handler = self.message_handlers.get(action)
            
            if handler:
                try:
                    result = await handler(message.payload.get("data", {}))
                    return self.create_response(message, result, success=True)
                except Exception as e:
                    return self.create_response(
                        message, {"error": str(e)}, success=False
                    )
            else:
                return self.create_response(
                    message, {"error": f"Unknown action: {action}"}, success=False
                )
        return None
    
    def get_agent_card_json(self) -> str:
        """Export agent card as JSON."""
        return json.dumps(self.agent_card.to_dict(), indent=2)


# Pre-defined agent cards for our system
POLICY_AGENT_CARDS = {
    "data_collector": AgentCard(
        agent_id="data-collector-001",
        name="Data Collector Agent",
        description="Collects real-time air quality and policy data",
        version="1.0.0",
        capabilities=["fetch_air_quality", "search_policies", "bulk_data_collection"],
        input_schema={"type": "object", "properties": {"country": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"data": {"type": "array"}}}
    ),
    "policy_analyzer": AgentCard(
        agent_id="policy-analyzer-001",
        name="Policy Analyzer Agent",
        description="Analyzes environmental policy effectiveness",
        version="1.0.0",
        capabilities=["trend_analysis", "statistical_significance", "effectiveness_score"],
        input_schema={"type": "object", "properties": {"policy_id": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"analysis": {"type": "object"}}}
    ),
    "visualizer": AgentCard(
        agent_id="visualizer-001",
        name="Visualization Agent",
        description="Creates interactive visualizations",
        version="1.0.0",
        capabilities=["globe_viz", "timeline_chart", "comparison_chart"],
        input_schema={"type": "object", "properties": {"data": {"type": "array"}}},
        output_schema={"type": "object", "properties": {"config": {"type": "object"}}}
    ),
    "reporter": AgentCard(
        agent_id="reporter-001",
        name="Report Generator Agent",
        description="Generates human-readable reports",
        version="1.0.0",
        capabilities=["executive_summary", "insights", "recommendations"],
        input_schema={"type": "object", "properties": {"analysis": {"type": "object"}}},
        output_schema={"type": "object", "properties": {"report": {"type": "string"}}}
    )
}
