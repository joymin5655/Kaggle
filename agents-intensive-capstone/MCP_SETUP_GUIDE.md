# Claude Desktop MCP 설정 가이드

이 가이드는 Environmental Policy Agent System을 Claude Desktop과 연동하는 방법을 설명합니다.

## 📋 사전 요구사항

1. **Claude Desktop** 설치
   - https://claude.ai/download 에서 다운로드

2. **Python 3.10+** 설치
   ```bash
   python --version  # 3.10 이상 확인
   ```

3. **필요 패키지 설치**
   ```bash
   pip install mcp fastmcp httpx python-dotenv
   ```

## 🔧 MCP 서버 설정

### Step 1: 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집 - API 키 입력
WAQI_API_KEY=your_waqi_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 2: MCP 서버 테스트

```bash
# 서버 실행 테스트
python mcp_server.py
```

정상 실행 시 출력:
```
🌍 Environmental Policy MCP Server
==================================================

Available tools:
  • get_realtime_air_quality - 실시간 대기질 조회
  • search_environmental_policies - 정책 검색
  • analyze_policy_effectiveness - 효과성 분석
  • compare_countries - 국가 비교

Starting server...
```

### Step 3: Claude Desktop 설정

Claude Desktop 설정 파일 위치:
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

설정 파일 열기 (Mac):
```bash
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Step 4: MCP 서버 등록

`claude_desktop_config.json` 파일에 다음 내용 추가:

```json
{
  "mcpServers": {
    "environmental-policy": {
      "command": "python",
      "args": [
        "/Volumes/WD_BLACK SN770M 2TB/My_proj/Kaggle/agents-intensive-capstone/mcp_server.py"
      ],
      "env": {
        "WAQI_API_KEY": "your_waqi_api_key_here"
      }
    }
  }
}
```

> ⚠️ **중요**: `args`의 경로를 실제 mcp_server.py 파일 경로로 수정하세요.

### Step 5: Claude Desktop 재시작

1. Claude Desktop 완전 종료 (Command + Q on Mac)
2. Claude Desktop 다시 실행
3. 좌측 하단 🔌 플러그 아이콘 확인

## 🎯 사용 방법

Claude Desktop에서 다음과 같이 질문하면 MCP 도구가 자동으로 호출됩니다:

### 예시 1: 실시간 대기질 조회
```
서울의 현재 대기질을 알려줘
```

### 예시 2: 정책 검색
```
한국의 환경 정책을 검색해줘
```

### 예시 3: 정책 효과 분석
```
한국의 2023년 배출 감소 정책의 효과를 분석해줘
```

### 예시 4: 국가 비교
```
한국, 중국, 일본의 대기질을 비교해줘
```

## 🔧 문제 해결

### 문제 1: MCP 서버가 표시되지 않음
1. 설정 파일 JSON 유효성 확인
2. 파이썬 경로가 올바른지 확인
3. Claude Desktop 완전 재시작

### 문제 2: API 키 오류
1. `.env` 파일에 WAQI_API_KEY 설정 확인
2. 또는 `claude_desktop_config.json`의 `env`에 직접 입력

### 문제 3: 도구 실행 실패
```bash
# MCP 서버 직접 실행하여 에러 확인
python mcp_server.py
```

## 📚 MCP 도구 목록

| 도구 | 설명 | 예시 |
|------|------|------|
| `get_realtime_air_quality` | 실시간 AQI 조회 | "서울 대기질" |
| `search_environmental_policies` | 정책 검색 | "한국 환경 정책" |
| `analyze_policy_effectiveness` | 효과성 분석 | "정책 효과 분석" |
| `compare_countries` | 국가 비교 | "한중일 비교" |

## 🔗 관련 링크

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [WAQI API](https://aqicn.org/api/)
- [Claude Desktop](https://claude.ai/download)
