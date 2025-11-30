# 🌍 Environmental Policy Impact Agent System

AI-powered multi-agent system for analyzing environmental policy effectiveness worldwide.

**Team Robee** | Kaggle AI Agents Intensive Capstone Project | Track A: Consent Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✅ Course Concepts Implemented

| Day | Topic | Implementation | Status |
|-----|-------|----------------|--------|
| **Day 1** | Multi-Agent Architecture | 4 specialized agents + orchestrator | ✅ |
| **Day 2** | Tools & MCP | 5 custom tools + FastMCP server | ✅ |
| **Day 3** | Memory & Context | Session + Long-term memory | ✅ |
| **Day 4** | Observability | Logger, Tracer, Metrics, Evaluator | ✅ |
| **Day 5** | A2A & Deployment | Agent Cards, A2A Protocol, Configs | ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PolicyAgentSystem (Orchestrator)           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Data      │  │   Policy     │  │  Visualizer  │      │
│  │  Collector   │→ │  Analyzer    │→ │    Agent     │      │
│  │    Agent     │  │    Agent     │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                                    ↓              │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Reporter Agent                       │      │
│  └──────────────────────────────────────────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  Day 3: Memory     │  Day 4: Observability  │  Day 5: A2A  │
│  ├── Session       │  ├── Logger            │  ├── Cards   │
│  └── Long-term     │  ├── Tracer            │  └── Protocol│
│                    │  ├── Metrics           │              │
│                    │  └── Evaluator         │              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agents-intensive-capstone/
├── agents/                    # Day 1: Multi-Agent System
│   ├── data_collector.py      # Fetches WAQI API data
│   ├── policy_analyzer.py     # Statistical analysis
│   ├── visualizer.py          # Chart configurations
│   └── reporter.py            # Report generation
│
├── tools/                     # Day 2: Custom Tools
│   ├── waqi_tool.py           # Air quality API
│   ├── policy_db_tool.py      # Policy database
│   ├── analysis_tool.py       # Statistical tools
│   └── visualization_tool.py  # Viz configs
│
├── memory/                    # Day 3: Memory Systems
│   ├── session_manager.py     # Short-term memory
│   └── long_term_memory.py    # Persistent storage
│
├── observability/             # Day 4: Observability ⭐
│   ├── logger.py              # Structured logging
│   ├── tracer.py              # Distributed tracing
│   ├── metrics.py             # Performance metrics
│   └── evaluator.py           # Agent evaluation
│
├── deployment/                # Day 5: Deployment ⭐
│   ├── a2a_protocol.py        # Agent2Agent protocol
│   └── deployment_config.py   # Production configs
│
├── main.py                    # System orchestrator
├── mcp_server.py              # MCP server
├── config.py                  # Configuration
└── notebooks/
    └── demo_kaggle.ipynb      # Demo notebook
```


---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/joymin5655/Kaggle.git
cd Kaggle/agents-intensive-capstone
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env with your API keys:
# GEMINI_API_KEY=your_key_here
# WAQI_API_KEY=your_key_here
```

### 3. Run Demo

```bash
python main.py
```

---

## 📊 Day 4: Observability

### Logging
```python
from observability.logger import AgentLogger

logger = AgentLogger("MyAgent")
logger.log_tool_call("api_call", {"param": "value"}, {"result": "data"}, duration_ms=150)
```

### Tracing
```python
from observability.tracer import AgentTracer

tracer = AgentTracer()
trace_id = tracer.start_trace("analyze_policy")
with tracer.span(trace_id, "data_collection"):
    # ... your code
tracer.end_trace(trace_id)
```

### Metrics
```python
from observability.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record_tool_call("waqi_api", 150, success=True)
metrics.get_summary()  # Get aggregated stats
```

---

## 🔗 Day 5: A2A Protocol

### Agent Cards
```python
from deployment.a2a_protocol import AgentCard, A2AProtocol

card = AgentCard(
    agent_id="my-agent-001",
    name="My Agent",
    capabilities=["analyze", "report"],
    input_schema={...},
    output_schema={...}
)

protocol = A2AProtocol(card)
protocol.discover_agents(capability="analyze")
```

### Deployment Configs
```python
from deployment.deployment_config import DeploymentConfig

config = DeploymentConfig.for_environment("production")
config.export_json("deploy/config.json")
```

---

## 🧪 Demo Results

**South Korea - 2019 Fine Dust Management Act**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| PM2.5 | 38 μg/m³ | 24 μg/m³ | **-37%** |
| Bad Air Days | 87/year | 43/year | **-51%** |
| Statistical Significance | - | - | **p < 0.001** |

---

## 📚 References

- [Google AI Agents Intensive Course](https://www.kaggle.com/learn-guide/5-day-agents)
- [Google ADK Documentation](https://github.com/google/adk-python)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [WAQI API](https://aqicn.org/api/)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 👥 Team Robee

Built with ❤️ for a cleaner planet 🌍

**GitHub**: https://github.com/joymin5655/Kaggle/tree/main/agents-intensive-capstone
