# 🌍 Environmental Policy Impact Agent System

AI-powered multi-agent system for analyzing environmental policy effectiveness worldwide.

**Team Robee** | Kaggle AI Agents Intensive Capstone | Track A: Consent Agents

---

## 🎓 Google AI Agents Intensive - 5일 코스 구현

| Day | Topic | Implementation | Status |
|-----|-------|----------------|--------|
| **Day 1** | Multi-Agent Architecture | `Agent`, `InMemoryRunner`, Sub-agents | ✅ |
| **Day 2** | Tools & MCP | `FunctionTool`, Custom tools | ✅ |
| **Day 3** | Sessions & Memory | `InMemorySessionService`, `InMemoryMemoryService` | ✅ |
| **Day 4** | Observability | `AgentLogger`, `AgentTracer`, `MetricsCollector` | ✅ |
| **Day 5** | A2A Protocol | `AgentCard`, `RemoteA2aAgent`, `A2AProtocol` | ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              PolicyAgentSystem (Orchestrator)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Data      │  │   Policy     │  │   Reporter   │      │
│  │  Collector   │→ │   Analyzer   │→ │    Agent     │      │
│  │    Agent     │  │    Agent     │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│        ↓                  ↓                  ↓              │
│    Tools:            Tools:              Output:           │
│  - get_air_quality   - analyze_         Korean Report     │
│  - search_policies     effectiveness                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Day 3: Memory     │  Day 4: Observability  │  Day 5: A2A  │
│  ├── Session       │  ├── Logger            │  ├── Card    │
│  └── Long-term     │  ├── Tracer            │  └── Protocol│
│                    │  └── Metrics           │              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agents-intensive-capstone/
├── main.py              # 🎯 All 5 days in one file (ADK-compatible)
├── mcp_server.py        # Day 2: MCP server for Claude Desktop
├── config.py            # Configuration
├── requirements.txt     # Dependencies
│
├── agents/              # Day 1: Multi-agent components
├── tools/               # Day 2: Custom tools
├── memory/              # Day 3: Memory services
├── observability/       # Day 4: Logging, tracing, metrics
├── deployment/          # Day 5: A2A protocol, configs
│
├── data/                # Sample policy data
├── README.md            # This file
└── .env.example         # API key template
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/joymin5655/Kaggle.git
cd Kaggle/agents-intensive-capstone
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
cp .env.example .env
# Edit .env:
# GOOGLE_API_KEY=your_gemini_api_key
# WAQI_API_KEY=your_waqi_api_key
```

### 3. Run Demo
```bash
python main.py
```

---

## 📊 Demo Output

```
============================================================
🌍 Environmental Policy Impact Agent System
   Kaggle AI Agents Intensive Capstone - Team Robee
============================================================

📋 South Korea 환경정책 분석 보고서

### 정책: Comprehensive Fine Dust Management Act
### 효과성 점수: 100/100 (🟢 매우 효과적)

#### 📊 분석 결과:
- 목표 감축률: 35%
- 실제 감축률: 37%
- 목표 달성: ✅ 예
- 통계적 유의성: p < 0.001
- 효과 크기: Large
```

---

## 🔑 API Keys

| API | Purpose | Get it from |
|-----|---------|-------------|
| **Gemini API** | LLM for agent reasoning | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| **WAQI API** | Real-time air quality | [WAQI](https://aqicn.org/api/) |

---

## 📚 5-Day Course Reference

| Day | Whitepaper | Codelab |
|-----|------------|---------|
| 1 | Introduction to Agents | [1a](https://www.kaggle.com/code/kaggle5daysofai/day-1a-from-prompt-to-action), [1b](https://www.kaggle.com/code/kaggle5daysofai/day-1b-agent-architectures) |
| 2 | Tools & MCP | [2a](https://www.kaggle.com/code/kaggle5daysofai/day-2a-agent-tools) |
| 3 | Sessions & Memory | [3a](https://www.kaggle.com/code/kaggle5daysofai/day-3a-agent-sessions) |
| 4 | Agent Quality | [4a](https://www.kaggle.com/code/kaggle5daysofai/day-4a-agent-observability) |
| 5 | Prototype to Production | [5a](https://www.kaggle.com/code/kaggle5daysofai/day-5a-agent2agent-communication), [5b](https://www.kaggle.com/code/kaggle5daysofai/day-5b-agent-deployment) |

---

## 👥 Team Robee

Built with ❤️ for a cleaner planet 🌍

**GitHub**: https://github.com/joymin5655/Kaggle/tree/main/agents-intensive-capstone
