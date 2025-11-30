"""
Test Suite for Environmental Policy Impact Agent System
========================================================
Tests for all 5 days of Google AI Agents Intensive implementation.
"""

import pytest
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    # Day 1: Agent & Runner
    Agent,
    Runner,
    InMemoryRunner,
    # Day 2: Tools
    FunctionTool,
    get_air_quality,
    search_policies,
    analyze_effectiveness,
    compare_countries,
    # Day 3: Memory
    InMemorySessionService,
    InMemoryMemoryService,
    # Day 4: Observability
    AgentLogger,
    AgentTracer,
    MetricsCollector,
    AgentEvaluator,
    GoldenTask,
    # Day 5: A2A
    AgentCard,
    AgentSkill,
    RemoteA2aAgent,
    A2AProtocol,
    # Main System
    PolicyAgentSystem,
)


# ============================================================
# Day 1: Agent & Runner Tests
# ============================================================

class TestDay1Agent:
    """Tests for Day 1: Multi-Agent Architecture."""
    
    def test_agent_creation(self):
        """Test agent can be created with basic parameters."""
        agent = Agent(name="test_agent", instruction="Test instruction")
        assert agent.name == "test_agent"
        assert agent.instruction == "Test instruction"
        assert agent.tools == []
        assert agent.sub_agents == []
    
    def test_agent_with_tools(self):
        """Test agent can be created with tools."""
        agent = Agent(
            name="test_agent",
            tools=[get_air_quality, search_policies]
        )
        assert len(agent.tools) == 2
        assert "get_air_quality" in agent.get_tools()
        assert "search_policies" in agent.get_tools()
    
    def test_agent_with_sub_agents(self):
        """Test agent can have sub-agents."""
        sub_agent1 = Agent(name="sub1")
        sub_agent2 = Agent(name="sub2")
        orchestrator = Agent(
            name="orchestrator",
            sub_agents=[sub_agent1, sub_agent2]
        )
        assert len(orchestrator.sub_agents) == 2
    
    def test_runner_creation(self):
        """Test runner can be created with agent."""
        agent = Agent(name="test")
        runner = Runner(agent=agent, app_name="test-app")
        assert runner.agent == agent
        assert runner.app_name == "test-app"


# ============================================================
# Day 2: Custom Tools Tests
# ============================================================

class TestDay2Tools:
    """Tests for Day 2: Custom Tools & MCP."""
    
    def test_get_air_quality_seoul(self):
        """Test air quality retrieval for Seoul."""
        result = get_air_quality("Seoul")
        assert "aqi" in result
        assert "pm25" in result
        assert "pm10" in result
        assert "status" in result
        assert result["city"] == "Seoul"
    
    def test_get_air_quality_unknown_city(self):
        """Test air quality for unknown city returns default."""
        result = get_air_quality("UnknownCity")
        assert "aqi" in result
        assert result["status"] == "Unknown"
    
    def test_search_policies_south_korea(self):
        """Test policy search for South Korea."""
        policies = search_policies("South Korea")
        assert len(policies) > 0
        assert policies[0]["name"] == "Comprehensive Fine Dust Management Act"
        assert policies[0]["target_reduction"] == 35
        assert policies[0]["actual_reduction"] == 37
    
    def test_search_policies_unknown_country(self):
        """Test policy search for unknown country returns empty."""
        policies = search_policies("UnknownCountry")
        assert policies == []
    
    def test_analyze_effectiveness_exceeded_target(self):
        """Test effectiveness when target is exceeded."""
        result = analyze_effectiveness(target=35, actual=37)
        assert result["effectiveness_score"] == 100
        assert result["exceeded_target"] == True
        assert result["rating"] == "Highly Effective"
        assert result["emoji"] == "🟢"
    
    def test_analyze_effectiveness_met_target(self):
        """Test effectiveness when target is exactly met."""
        result = analyze_effectiveness(target=35, actual=35)
        assert result["effectiveness_score"] == 100
        assert result["exceeded_target"] == True
    
    def test_analyze_effectiveness_below_target(self):
        """Test effectiveness when below target."""
        result = analyze_effectiveness(target=46, actual=20)
        assert result["effectiveness_score"] == 43
        assert result["exceeded_target"] == False
        assert result["rating"] == "Needs Improvement"
        assert result["emoji"] == "🔴"
    
    def test_compare_countries(self):
        """Test multi-country comparison."""
        result = compare_countries(["South Korea", "China", "Japan"])
        assert "comparison" in result
        assert "best_performer" in result
        assert len(result["comparison"]) == 3
        # Check ranking exists
        assert all("rank" in item for item in result["comparison"])
        # Check sorted by score
        scores = [item["effectiveness_score"] for item in result["comparison"]]
        assert scores == sorted(scores, reverse=True)


# ============================================================
# Day 3: Memory Tests
# ============================================================

class TestDay3Memory:
    """Tests for Day 3: Sessions & Memory."""
    
    def test_session_creation(self):
        """Test session can be created."""
        service = InMemorySessionService()
        session_id = service.create_session()
        assert session_id is not None
        assert session_id.startswith("session_")
    
    def test_session_state(self):
        """Test session state management."""
        service = InMemorySessionService()
        session_id = service.create_session()
        
        service.update_state(session_id, "country", "South Korea")
        assert service.get_state(session_id, "country") == "South Korea"
    
    def test_session_history(self):
        """Test session history tracking."""
        service = InMemorySessionService()
        session_id = service.create_session()
        
        service.add_to_history(session_id, "user", "Analyze policy")
        service.add_to_history(session_id, "assistant", "Analysis complete")
        
        history = service.get_history(session_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_memory_store_and_retrieve(self):
        """Test memory storage and retrieval."""
        service = InMemoryMemoryService()
        
        memory_id = service.store(
            content={"analysis": "test"},
            metadata={"type": "analysis"},
            tags=["test", "analysis"]
        )
        
        retrieved = service.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved["content"]["analysis"] == "test"
    
    def test_memory_search(self):
        """Test memory search by keyword."""
        service = InMemoryMemoryService()
        
        service.store(content={"country": "South Korea"}, tags=["korea"])
        service.store(content={"country": "Japan"}, tags=["japan"])
        
        results = service.search("Korea")
        assert len(results) == 1
        assert results[0]["content"]["country"] == "South Korea"
    
    def test_memory_search_by_tag(self):
        """Test memory search by tag."""
        service = InMemoryMemoryService()
        
        service.store(content={"data": "1"}, tags=["analysis"])
        service.store(content={"data": "2"}, tags=["analysis"])
        service.store(content={"data": "3"}, tags=["other"])
        
        results = service.search_by_tag("analysis")
        assert len(results) == 2


# ============================================================
# Day 4: Observability Tests
# ============================================================

class TestDay4Observability:
    """Tests for Day 4: Observability & Evaluation."""
    
    def test_logger_creation(self):
        """Test logger can be created."""
        logger = AgentLogger("test")
        assert logger.name == "test"
        assert logger.logs == []
    
    def test_logger_levels(self):
        """Test different log levels."""
        logger = AgentLogger("test")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        assert len(logger.logs) == 4
        assert logger.logs[0]["level"] == "DEBUG"
        assert logger.logs[1]["level"] == "INFO"
        assert logger.logs[2]["level"] == "WARNING"
        assert logger.logs[3]["level"] == "ERROR"
    
    def test_tracer_trace_lifecycle(self):
        """Test trace start and end."""
        tracer = AgentTracer("test-service")
        
        trace_id = tracer.start_trace("test-operation")
        assert trace_id is not None
        assert tracer.current_trace_id == trace_id
        
        tracer.add_span("step1", duration_ms=100)
        tracer.add_span("step2", duration_ms=200)
        
        result = tracer.end_trace(trace_id, "OK")
        assert result["status"] == "OK"
        assert len(result["spans"]) == 2
        assert result["total_duration_ms"] >= 0
    
    def test_metrics_counter(self):
        """Test metrics counter."""
        metrics = MetricsCollector()
        
        metrics.increment("requests")
        metrics.increment("requests")
        metrics.increment("requests", value=5)
        
        summary = metrics.summary()
        assert summary["counters"]["requests"] == 7
    
    def test_metrics_histogram(self):
        """Test metrics histogram."""
        metrics = MetricsCollector()
        
        for i in range(10):
            metrics.record("duration", float(i * 10))
        
        stats = metrics.get_histogram_stats("duration")
        assert stats["count"] == 10
        assert stats["min"] == 0
        assert stats["max"] == 90
        assert stats["avg"] == 45
    
    def test_evaluator_golden_task(self):
        """Test golden task evaluation."""
        evaluator = AgentEvaluator("test-agent")
        
        evaluator.add_golden_task(
            name="test_task",
            input_query="South Korea",
            expected_output="Fine Dust",
            validator=AgentEvaluator.contains_validator
        )
        
        def mock_agent(query):
            return {"policy": "Fine Dust Management Act"}
        
        results = evaluator.evaluate_all(mock_agent)
        assert results["total_tasks"] == 1
        assert results["passed"] == 1
        assert results["pass_rate"] == 1.0


# ============================================================
# Day 5: A2A Protocol Tests
# ============================================================

class TestDay5A2A:
    """Tests for Day 5: A2A Protocol."""
    
    def test_agent_skill_creation(self):
        """Test agent skill can be created."""
        skill = AgentSkill(
            id="analyze",
            name="Analyze Policy",
            description="Analyze environmental policy",
            tags=["analysis"],
            examples=["Analyze South Korea's policy"]
        )
        assert skill.id == "analyze"
        assert skill.name == "Analyze Policy"
    
    def test_agent_card_creation(self):
        """Test agent card can be created."""
        card = AgentCard(
            name="Test Agent",
            description="Test description",
            skills=[
                AgentSkill(id="skill1", name="Skill 1", description="Desc")
            ]
        )
        assert card.name == "Test Agent"
        assert card.protocol_version == "A2A/1.0"
    
    def test_agent_card_to_json(self):
        """Test agent card JSON serialization."""
        card = AgentCard(
            name="Test Agent",
            description="Test description"
        )
        json_str = card.to_json()
        assert "Test Agent" in json_str
        assert "A2A/1.0" in json_str
    
    def test_a2a_protocol_discovery(self):
        """Test A2A protocol agent discovery."""
        agent = Agent(name="local")
        card = AgentCard(name="Local Agent", description="Test")
        protocol = A2AProtocol(agent, card)
        
        remote = RemoteA2aAgent(
            name="remote-agent",
            description="Remote agent",
            agent_card_url="http://example.com/.well-known/agent.json"
        )
        protocol.register_remote(remote)
        
        discovered = protocol.discover()
        assert len(discovered) == 1
        assert discovered[0]["name"] == "remote-agent"


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for PolicyAgentSystem."""
    
    @pytest.fixture
    def system(self):
        """Create PolicyAgentSystem instance."""
        return PolicyAgentSystem()
    
    def test_system_initialization(self, system):
        """Test system initializes correctly."""
        assert system.orchestrator is not None
        assert system.session_service is not None
        assert system.memory_service is not None
        assert system.logger is not None
        assert system.tracer is not None
        assert system.metrics is not None
        assert system.agent_card is not None
    
    def test_system_analyze(self, system):
        """Test full analysis pipeline."""
        # Run async in sync context
        result = asyncio.get_event_loop().run_until_complete(
            system.analyze("South Korea")
        )
        
        assert "country" in result
        assert "air_quality" in result
        assert "policies" in result
        assert "analysis" in result
        assert "report" in result
        assert result["analysis"]["effectiveness_score"] == 100
    
    def test_system_compare(self, system):
        """Test multi-country comparison."""
        result = asyncio.get_event_loop().run_until_complete(
            system.compare(["South Korea", "China"])
        )
        
        assert "comparison" in result
        assert "best_performer" in result
        assert len(result["comparison"]) == 2
    
    def test_system_observability(self, system):
        """Test observability after operations."""
        asyncio.get_event_loop().run_until_complete(
            system.analyze("South Korea")
        )
        
        obs = system.get_observability_summary()
        assert obs["metrics"]["counters"]["analysis_requests"] >= 1
        assert obs["logs"]["total"] >= 1
        assert obs["traces"]["total"] >= 1
        assert obs["memory"]["total_memories"] >= 1
    
    def test_system_evaluation(self, system):
        """Test golden task evaluation."""
        results = system.run_evaluation()
        
        assert "total_tasks" in results
        assert "passed" in results
        assert "pass_rate" in results
        assert results["pass_rate"] >= 0.5  # At least 50% pass rate
    
    def test_system_a2a_card(self, system):
        """Test A2A agent card generation."""
        card_json = system.get_a2a_card()
        
        assert "Environmental Policy Agent" in card_json
        assert "A2A/1.0" in card_json
        assert "analyze_policy" in card_json


# ============================================================
# Run Tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
