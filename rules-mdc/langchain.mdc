---
description: Definitive guidelines for writing maintainable, performant, and robust LangChain applications using modern best practices (LCEL, LangGraph, Pydantic, `create_agent`).
globs: **/*.py
---
# LangChain Best Practices

This guide outlines the definitive best practices for developing with LangChain. Adhere to these rules to ensure your LLM applications are modular, scalable, and production-ready.

## 1. Code Organization and Structure

Always structure your LangChain projects around core components, separating concerns into distinct modules. This enhances readability, testability, and maintainability.

**✅ GOOD: Modular Structure**
Organize by component type (models, prompts, tools, agents, memory).

```python
# my_project/
# ├── agents/
# │   └── flight_booking_agent.py
# ├── models/
# │   └── llm_config.py
# ├── prompts/
# │   └── flight_prompts.py
# ├── tools/
# │   └── flight_tools.py
# ├── memory/
# │   └── chat_memory.py
# └── main.py
```

**❌ BAD: Monolithic Files**
Avoid dumping all logic into a single file.

```python
# main.py (containing everything)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# ... many more imports and definitions
# ... LLM, prompt, tools, agent definition all in one file
```

## 2. Leverage LangChain Expression Language (LCEL)

LCEL is the modern, recommended way to compose chains. It offers first-class streaming, async support, and clear debugging. **Never use deprecated `LLMChain` or older chain patterns.**

**✅ GOOD: LCEL for Chains**
Use the `|` operator for clear, composable pipelines.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

# Compose chain with LCEL
chain = prompt | llm | output_parser

# Invoke
response = chain.invoke({"question": "What is the capital of France?"})
print(response)
```

**❌ BAD: Deprecated `LLMChain`**
This pattern is outdated and lacks modern features.

```python
from langchain.chains import LLMChain # DEPRECATED
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("What is the capital of {country}?")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Avoid this
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.invoke({"country": "Germany"})
print(response)
```

## 3. Agents: Use `create_agent` with LangGraph

For stateful, multi-step reasoning, use `create_agent` which is built on LangGraph. This provides durable execution, checkpointing, and human-in-the-loop support.

**✅ GOOD: `create_agent` for Robust Agents**
Leverage the proven ReAct pattern out-of-the-box.

```python
from langchain.agents import create_agent, tool
from langchain_openai import ChatOpenAI
from typing import Literal

@tool
def get_current_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit") -> str:
    """Get the current weather in a given location and unit."""
    if "tokyo" in location.lower():
        return "It's 25 degrees Celsius and sunny in Tokyo."
    elif "san francisco" in location.lower():
        return "It's 60 degrees Fahrenheit and foggy in San Francisco."
    else:
        return f"Weather data for {location} not available."

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_current_weather]
system_prompt = "You are a helpful AI assistant that can answer questions about the weather."

agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

# Run the agent
result = agent_executor.invoke({"messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}]})
print(result["messages"][-1].content)
```

**❌ BAD: Manually Implementing Agent Logic**
Re-inventing the wheel leads to brittle, less robust agents.

```python
# Avoid complex, manual agent loops unless absolutely necessary with LangGraph directly.
# This often lacks built-in persistence, error handling, and standard patterns.
# ... (complex custom loop with if/else for tool calling and response generation)
```

## 4. Enforce Structured Output with Pydantic

Always define structured output schemas for LLM responses using Pydantic. This drastically reduces hallucination risk and simplifies downstream processing.

**✅ GOOD: Pydantic for Reliable Output**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(description="the setup of the joke")
    punchline: str = Field(description="the punch line of the joke")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(Joke)

prompt = ChatPromptTemplate.from_messages([
    ("human", "Tell me a joke about {topic}")
])

chain = prompt | structured_llm
joke_obj = chain.invoke({"topic": "bears"})
print(f"Setup: {joke_obj.setup}\nPunchline: {joke_obj.punchline}")
```

**❌ BAD: Relying on Free-Form Text Output**
Parsing unstructured text is error-prone and fragile.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("human", "Tell me a joke about {topic}. Format it as 'Setup: ... Punchline: ...'")
])
chain = prompt | llm | StrOutputParser()
raw_joke = chain.invoke({"topic": "dogs"})
# This will require manual, brittle parsing logic
print(raw_joke)
```

## 5. Asynchronous Operations and Streaming

For responsive user experiences and efficient resource utilization, always use LangChain's asynchronous APIs and streaming capabilities.

**✅ GOOD: Async and Streaming**

```python
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

async def stream_response():
    prompt = ChatPromptTemplate.from_template("Write a long poem about {topic}.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    print("Streaming response:")
    async for chunk in chain.stream({"topic": "the ocean"}):
        print(chunk, end="", flush=True)
    print("\n--- End Stream ---")

if __name__ == "__main__":
    asyncio.run(stream_response())
```

**❌ BAD: Blocking Calls for Long Operations**
Synchronous calls block the event loop, leading to poor UX.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def get_blocking_response():
    prompt = ChatPromptTemplate.from_template("Write a long poem about {topic}.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()

    print("Getting blocking response...")
    response = chain.invoke({"topic": "the ocean"}) # Blocks until complete
    print(response)

if __name__ == "__main__":
    get_blocking_response()
```

## 6. Type Hints

Strictly use type hints for all functions, variables, and LangChain components. This improves code clarity, enables static analysis, and reduces runtime errors.

**✅ GOOD: Comprehensive Type Hinting**

```python
from typing import List, Dict, Any
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage

def process_chat_history(
    messages: List[BaseMessage],
    agent_chain: Runnable[Dict[str, Any], Dict[str, Any]]
) -> List[BaseMessage]:
    """Processes a list of chat messages using an agent chain."""
    # ... logic
    return messages

# Example usage with proper types
# agent_chain = ... (a Runnable object)
# history: List[BaseMessage] = []
# updated_history = process_chat_history(history, agent_chain)
```

**❌ BAD: Untyped Code**
Makes code harder to understand and refactor.

```python
def process_chat_history(messages, agent_chain): # No type hints
    # ... logic
    return messages
```

## 7. Secure API Key Management

Never hardcode API keys. Always use environment variables, preferably loaded via `python-dotenv`.

**✅ GOOD: Environment Variables with `python-dotenv`**

```python
# .env file (NOT committed to Git)
# OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="sk-..."

# my_app.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv() # Load environment variables from .env

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)
```

**❌ BAD: Hardcoded API Keys**
A major security vulnerability.

```python
# my_app.py
from langchain_openai import ChatOpenAI

# NEVER DO THIS
llm = ChatOpenAI(openai_api_key="sk-YOUR_HARDCODED_KEY_HERE", model="gpt-4o-mini")
```

## 8. Testing Approaches

Implement a robust testing strategy including unit tests for components and integration tests for chains/agents. Leverage LangSmith for tracing and evaluation.

**✅ GOOD: Unit & Integration Tests + LangSmith**

```python
# tests/test_tools.py
import pytest
from tools.flight_tools import get_flight_status # Assuming a tool module

def test_get_flight_status_valid():
    status = get_flight_status("AA123")
    assert "on time" in status.lower()

# tests/test_agent.py
from langchain_core.messages import HumanMessage
from agents.flight_booking_agent import agent_executor # Assuming an agent module

@pytest.mark.asyncio
async def test_flight_booking_agent_query():
    # Set LANGCHAIN_TRACING_V2="true" and LANGCHAIN_API_KEY for LangSmith
    response = await agent_executor.ainvoke({"messages": [HumanMessage(content="What is the status of flight AA123?")]})
    assert "on time" in response["messages"][-1].content.lower()

# For prompt engineering, use golden tests (input/expected output pairs)
```

**❌ BAD: No Tests or Manual Testing Only**
Leads to regressions and unreliable LLM applications.

```python
# No test files, relying on manual `python main.py` to check functionality.
```