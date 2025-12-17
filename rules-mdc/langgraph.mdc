---
description: This guide provides definitive best practices for building robust, maintainable, and performant AI agent workflows using LangGraph.
globs: **/*.py
---
# langgraph Best Practices

## Code Organization and Structure

### 1. Explicit Graph Topology & Modularity
Define clear, single responsibilities for each node. Encapsulate common patterns into reusable subgraphs. Prefer Directed Acyclic Graphs (DAGs); use cycles only when essential for feedback loops.

✅ GOOD: Focused nodes, composable subgraphs
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[str]
    tool_output: str | None

def fetch_data_node(state: AgentState) -> AgentState:
    return {"messages": state["messages"] + ["Data fetched."]}

def analyze_data_node(state: AgentState) -> AgentState:
    return {"messages": state["messages"] + ["Data analyzed."]}

def build_subgraph():
    builder = StateGraph(AgentState)
    builder.add_node("fetch", fetch_data_node)
    builder.add_node("analyze", analyze_data_node)
    builder.add_edge("fetch", "analyze")
    builder.set_entry_point("fetch")
    builder.set_finish_point("analyze")
    return builder.compile()
```

### 2. Functional API Preference
Prior