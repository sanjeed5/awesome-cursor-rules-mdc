---
description: Definitive guidelines for building robust, performant, and maintainable AI agents using smolagents, focusing on practical patterns and avoiding common pitfalls.
globs: **/*.py
---
# smolagents Best Practices

This guide outlines the definitive best practices for developing `smolagents` applications. Adhering to these principles ensures your agents are reliable, efficient, and easy to maintain.

## 1. Code Organization and Structure

Organize your agent logic and tools into clear, distinct modules. This improves readability, reusability, and testability.

❌ BAD: Monolithic agent file
```python
# agent.py
from smolagents import CodeAgent, HfApiModel, tool
import requests # Direct import in agent file

@tool
def search_web(query: str) -> str:
    # ... complex search logic ...
    return "search result"

@tool
def calculate_math(expression: str) -> float:
    # ... math logic ...
    return eval(expression) # DANGER! Unsafe!

agent = CodeAgent(tools=[search_web, calculate_math], model=HfApiModel())
agent.run("What is 2+2 and also search for 'latest AI news'")
```

✅ GOOD: Modular structure
```python
# src/tools.py
from smolagents import tool
import requests
import json

@tool
def web_search(query: str) -> str:
    """
    Performs a web search for the given query.
    Args:
        query: The search term.
    Returns:
        A summary of the search results or an error message.
    """
    try:
        # Replace with a real search API (e.g., DuckDuckGoSearchTool, custom API)
        response = requests.get(f"https://api.example.com/search?q