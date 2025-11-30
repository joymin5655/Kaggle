# 🌍 Environmental Policy Impact Agent System

AI-powered multi-agent system for analyzing environmental policy effectiveness worldwide.

**Team Robee** | Kaggle AI Agents Intensive Capstone | Track A: Consent Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google ADK](https://img.shields.io/badge/Google-ADK-orange.svg)](https://github.com/google/adk-python)

---

## 🎓 Google AI Agents Intensive - All 5 Days Implemented

| Day | Topic | Implementation | Lines |
|-----|-------|----------------|-------|
| **Day 1** | Multi-Agent Architecture | `Agent`, `Runner`, `InMemoryRunner`, Sub-agents | ~200 |
| **Day 2** | Tools & MCP | `FunctionTool`, WAQI API, Policy DB, Analysis | ~300 |
| **Day 3** | Sessions & Memory | `InMemorySessionService`, `InMemoryMemoryService` | ~150 |
| **Day 4** | Observability & Evaluation | `AgentLogger`, `AgentTracer`, `MetricsCollector`, `AgentEvaluator` | ~400 |
| **Day 5** | A2A Protocol | `AgentCard`, `AgentSkill`, `RemoteA2aAgent`, `A2AProtocol` | ~250 |

**Total: ~1,500+ lines of production-ready code**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                PolicyAgentSystem (Orchestrator)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │ DataCollector │  │ PolicyAnalyzer│  │   Reporter    │       │
│  │    Agent      │→ │    Agent      │→ │    Agent      │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│         │                  │                   │                │
│         ▼                  ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Custom Tools                          │   │
│  │  • get_air_quality()     - WAQI API integration         │   │
│  │  • search_policies()     - Policy database              │   │
│  │  • analyze_effectiveness() - Statistical analysis       │   │
│  │  • compare_countries()   - Multi-country comparison     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Day 3: Memory          │  Day 4: Observability  │  Day 5: A2A │
│  ├── SessionService     │  ├── AgentLogger       │  ├── Card   │
│  └── MemoryService      │  ├── AgentTracer       │  ├── Skills │
│                         │  ├── MetricsCollector  │  └── Protocol│
│                         │  └── AgentEvaluator    │             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agents-intensive-capstone/
├── main.py              # 🎯 Complete implementation (~1,500 lines)
│                        #    All 5 days in one file
│
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── README.md            # This file
│
├── agents/              # Day 1: Agent modules (modular version)
├── tools/               # Day 2: Tool modules
├── memory/              # Day 3: Memory modules
├── observability/       # Day 4: Observability modules
├── deployment/          # Day 5: A2A & deployment modules
│
├── mcp_server.py        # Day 2: MCP server for Claude Desktop
├── data/                # Sample policy data
└── .env.example         # API key template
```

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone
git clone https://github.com/joymin5655/Kaggle.git
cd Kaggle/agents-intensive-capstone

# Install
pip install -r requirements.txt

# Set API keys (optional - demo data works without)
export GOOGLE_API_KEY=your_gemini_api_key
export WAQI_API_KEY=your_waqi_api_key

# Run
python main.py
```

### Option 2: Run in Kaggle Notebook

```python
# Install
!pip install -q google-adk httpx

# Load API keys from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["GOOGLE_API_KEY"] = secrets.get_secret("GEMINI_API_KEY")
os.environ["WAQI_API_KEY"] = secrets.get_secret("WAQI_API_KEY")

# Run
!python main.py
```

---

## 📊 Demo Output

```
======================================================================
🌍 Environmental Policy Impact Agent System
   Kaggle AI Agents Intensive Capstone Project - Team Robee
======================================================================

📊 DEMO 1: Single Country Analysis (South Korea)
======================================================================

============================================================
📋 South Korea 환경정책 분석 보고서
============================================================

📌 분석 대상 정책: 미세먼지 저감 및 관리에 관한 특별법
   (Comprehensive Fine Dust Management Act)

────────────────────────────────────────────────────────────
📊 효과성 평가
────────────────────────────────────────────────────────────
   점수: 100/100 🟢 매우 효과적
   목표 감축률: 35%
   실제 감축률: 37%
   목표 달성: ✅ 달성

🏆 Country Rankings:
Rank  Country        Score   Rating
1     South Korea    100     🟢 Highly Effective
2     China          100     🟢 Highly Effective
3     Germany        64      🟠 Moderately Effective
4     Japan          43      🔴 Needs Improvement
```

---

## 🔑 API Keys

| API | Purpose | Required? | Get it from |
|-----|---------|-----------|-------------|
| **Gemini API** | LLM reasoning | Optional* | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| **WAQI API** | Real-time air quality | Optional* | [WAQI](https://aqicn.org/api/) |

*Demo data is provided, so the system works without API keys.

---

## 📚 Course Reference

| Day | Whitepaper | Codelab |
|-----|------------|---------|
| 1 | Introduction to Agents | [1a](https://www.kaggle.com/code/kaggle5daysofai/day-1a-from-prompt-to-action), [1b](https://www.kaggle.com/code/kaggle5daysofai/day-1b-agent-architectures) |
| 2 | Tools & MCP | [2a](https://www.kaggle.com/code/kaggle5daysofai/day-2a-agent-tools) |
| 3 | Sessions & Memory | [3a](https://www.kaggle.com/code/kaggle5daysofai/day-3a-agent-sessions) |
| 4 | Agent Quality | [4a](https://www.kaggle.com/code/kaggle5daysofai/day-4a-agent-observability) |
| 5 | Prototype to Production | [5a](https://www.kaggle.com/code/kaggle5daysofai/day-5a-agent2agent-communication), [5b](https://www.kaggle.com/code/kaggle5daysofai/day-5b-agent-deployment) |

---

## 🏆 Features

### Day 1: Multi-Agent Architecture
- 4 specialized agents with distinct roles
- Orchestrator pattern for coordination
- Runner for execution management

### Day 2: Custom Tools
- `get_air_quality()` - Real WAQI API integration with fallback
- `search_policies()` - Policy database with 4 countries
- `analyze_effectiveness()` - Statistical analysis with Cohen's d
- `compare_countries()` - Multi-country ranking

### Day 3: Memory
- **Session Memory**: Conversation state, preferences, history
- **Long-term Memory**: Persistent storage with tags and search

### Day 4: Observability
- **Logger**: Structured logs with levels and tool call tracking
- **Tracer**: Distributed tracing with spans
- **Metrics**: Counters, gauges, histograms with percentiles
- **Evaluator**: Golden tasks, validators, pass rates

### Day 5: A2A Protocol
- **AgentCard**: Full A2A spec compliance
- **AgentSkill**: Capability definitions with examples
- **RemoteA2aAgent**: Connect to external agents
- **A2AProtocol**: Discovery and delegation

---

## 👥 Team Robee

Built with ❤️ for a cleaner planet 🌍

**GitHub**: https://github.com/joymin5655/Kaggle/tree/main/agents-intensive-capstone

---

## 📄 License

MIT License - see [LICENSE](LICENSE)
