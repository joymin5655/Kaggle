# Contributing to Environmental Policy Impact Agent System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

---

## 📜 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- pip

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Kaggle.git
cd Kaggle/agents-intensive-capstone
```

---

## 💻 Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

### Set Up API Keys (Optional)

```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## 🔧 Making Changes

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-new-country` |
| Bug Fix | `fix/description` | `fix/api-timeout` |
| Docs | `docs/description` | `docs/update-readme` |
| Refactor | `refactor/description` | `refactor/memory-service` |

---

## 📝 Code Style

### Python Style Guide

We follow PEP 8 with these additions:

```python
# Use type hints
def analyze_effectiveness(target: float, actual: float) -> Dict[str, Any]:
    """
    Analyze policy effectiveness.
    
    Args:
        target: Target reduction percentage
        actual: Actual reduction percentage
    
    Returns:
        Analysis results with effectiveness score
    """
    pass

# Use descriptive variable names
effectiveness_score = calculate_score(target, actual)  # Good
es = calc(t, a)  # Bad

# Document classes
class AgentLogger:
    """
    Structured logging for agents.
    
    Attributes:
        name: Logger identifier
        logs: List of log entries
    """
    pass
```

### Formatting

```bash
# Format code (if using black)
black main.py

# Check style (if using flake8)
flake8 main.py
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Write Tests

```python
# tests/test_tools.py
import pytest
from main import get_air_quality, search_policies, analyze_effectiveness

def test_get_air_quality_seoul():
    """Test air quality retrieval for Seoul."""
    result = get_air_quality("Seoul")
    assert "aqi" in result
    assert "pm25" in result
    assert result["city"] == "Seoul"

def test_analyze_effectiveness_exceeded():
    """Test effectiveness when target is exceeded."""
    result = analyze_effectiveness(target=35, actual=37)
    assert result["effectiveness_score"] == 100
    assert result["exceeded_target"] == True
    assert result["emoji"] == "🟢"
```

### Golden Task Evaluation

```python
# Use built-in evaluator
system = PolicyAgentSystem()
results = system.run_evaluation()
assert results["pass_rate"] >= 0.8  # 80% pass rate required
```

---

## 📤 Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add Germany to policy database with Climate Action Programme 2030"
git commit -m "Fix WAQI API timeout handling with retry logic"
git commit -m "Update README with detailed API reference"

# Bad
git commit -m "update"
git commit -m "fix bug"
```

### Commit Message Format

```
<type>: <short description>

<optional longer description>

<optional footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear title describing the change
- Description of what changed and why
- Reference to any related issues
- Screenshots if applicable (for UI changes)

---

## 📁 Project Structure for Contributors

```
agents-intensive-capstone/
├── main.py              # Main implementation - edit for core changes
├── config.py            # Configuration - add new settings here
├── requirements.txt     # Dependencies - add new packages here
│
├── agents/              # Add new agent types here
├── tools/               # Add new tools here
├── memory/              # Modify memory behavior here
├── observability/       # Add logging/metrics here
├── deployment/          # A2A and deployment changes
│
├── tests/               # Add tests here
│   └── test_tools.py
│
└── data/                # Add sample data here
    └── policies.json
```

---

## 🆘 Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Tag issues appropriately: `bug`, `enhancement`, `question`

---

## 🙏 Thank You!

Every contribution helps make environmental policy analysis more accessible. Together, we can help create a cleaner planet! 🌍
