# System Architecture

## Multi-Agent System (Day 1)

```
                    ┌─────────────────────┐
                    │   User Interface    │
                    │  (CLI / Natural     │
                    │    Language)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    │  (PolicyAgentSystem)│
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Data Collector  │ │  Policy Analyzer │ │   Visualizer     │
│      Agent       │ │      Agent       │ │      Agent       │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ • WAQI Tool      │ │ • Trend Tool     │ │ • Globe Tool     │
│ • Policy DB Tool │ │ • Compare Tool   │ │ • Chart Tool     │
│ • Search Tool    │ │ • Stats Tool     │ │ • Dashboard Tool │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Reporter Agent    │
                    ├─────────────────────┤
                    │ • Summary Tool      │
                    │ • Insight Tool      │
                    │ • Recommend Tool    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Final Report      │
                    │  (Executive Summary │
                    │   + Visualizations) │
                    └─────────────────────┘
```

## Memory System (Day 3)

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory System                           │
├─────────────────────────────┬───────────────────────────────┤
│      Session Memory         │       Long-Term Memory        │
│      (SessionManager)       │       (LongTermMemory)        │
├─────────────────────────────┼───────────────────────────────┤
│ • User preferences          │ • Past analysis results       │
│ • Query history             │ • User preferences (persist)  │
│ • Conversation context      │ • Historical comparisons      │
│ • Analysis cache            │ • Statistics & metadata       │
│                             │                               │
│ [In-memory, per session]    │ [JSON file, persistent]       │
└─────────────────────────────┴───────────────────────────────┘
```

## MCP Integration (Day 2)

```
┌─────────────────────┐         ┌─────────────────────┐
│   Claude Desktop    │◄───────►│    MCP Server       │
│   (AI Client)       │  stdio  │  (mcp_server.py)    │
└─────────────────────┘         └──────────┬──────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
          ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
          │ get_realtime_   │   │ search_policies │   │ analyze_        │
          │ air_quality     │   │                 │   │ effectiveness   │
          │                 │   │                 │   │                 │
          │ → WAQI API      │   │ → policies.json │   │ → Statistics    │
          └─────────────────┘   └─────────────────┘   └─────────────────┘
```

## Data Flow

```
1. User Query
   │
   ▼
2. Orchestrator parses intent
   │
   ├──► Data Collection Agent
   │    ├── Fetch WAQI data
   │    └── Search policy DB
   │
   ├──► Policy Analysis Agent
   │    ├── Calculate trends
   │    ├── Compare before/after
   │    └── Statistical tests
   │
   ├──► Visualization Agent
   │    ├── Generate globe config
   │    └── Create charts
   │
   └──► Reporter Agent
        ├── Generate summary
        ├── Extract insights
        └── Provide recommendations
   │
   ▼
3. Final Report delivered to user
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| AI Framework | Gemini ADK |
| MCP Server | FastMCP |
| Data APIs | WAQI (Air Quality) |
| Analysis | NumPy, SciPy |
| Storage | JSON (policies, memory) |
| Visualization | Three.js, Chart.js |
