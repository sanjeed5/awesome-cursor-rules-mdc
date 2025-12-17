---
description: This guide provides opinionated best practices for structuring, optimizing, and testing autogen multi-agent systems to ensure maintainability, reproducibility, and cost-efficiency.
globs: **/*.py
---
# autogen Best Practices

Building robust `autogen` applications requires disciplined adherence to specific patterns. This guide outlines the definitive best practices for our team, focusing on modularity, performance, and testability.

## 1. Code Organization and Agent Structure

Organize your agents into dedicated modules, ensuring each agent has a single, well-defined responsibility. This mirrors traditional software architecture and enhances maintainability.

❌ BAD: Monolithic agent definitions or generic names.
```python
# agents.py
from autogen import AssistantAgent, UserProxyAgent

# Too many responsibilities, unclear role
def create_complex_agent(llm_config):
    agent = AssistantAgent(
        name="GenericAgent",
        llm_config=llm_config,
        system_message="I can do anything you ask."
    )
    return agent
```

✅ GOOD: Dedicated modules, clear roles, and explicit system messages.
```python
# agents/planner.py
from autogen import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

def create_planner_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Creates an agent responsible for breaking down complex tasks into actionable steps.
    """
    return AssistantAgent(
        name="TaskPlanner",
        model_client=model_client,
        description="An expert in breaking down complex problems into a sequence of manageable sub-tasks.",
        system_message=(
            "You are a meticulous TaskPlanner. Your sole responsibility is to decompose user requests "
            "into a clear, ordered list of steps. Do not execute tasks, only plan them."
        )
    )

# agents/coder.py
from autogen import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

def create_coder_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Creates an agent capable of writing and executing Python code.
    """
    return AssistantAgent(
        name="PythonCoder",
        model_client=model_client,
        description="An expert Python programmer capable of writing, executing, and debugging code.",
        system_message=(
            "You are an expert Python programmer. You write clean, efficient, and well-tested code. "
            "When asked to solve a problem, provide the Python code in a markdown block. "
            "If execution is needed, state 'EXECUTE' after the code block."
        )
    )
```

## 2. LLM Caching for Reproducibility and Cost Efficiency

ALWAYS wrap your agent interactions in a caching context manager. This is non-negotiable for reproducible results and to prevent excessive API costs during development and testing.

❌ BAD: No caching, leading to variable outputs and repeated API calls.
```python
# In your main script
user_proxy.initiate_chat(assistant, message="Analyze data.") # Each run hits the LLM
```

✅ GOOD: Explicit caching with `Cache.disk()` or `Cache.redis()`.
```python
from autogen import Cache
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os

# Configure your LLM client once
llm_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create agents using the configured client
# (Assuming create_planner_agent and UserProxyAgent are defined)
planner = create_planner_agent(llm_client)
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False} # Or True with proper setup
)

# Use disk caching for development and testing
with Cache.disk(cache_path_root=".autogen_cache") as cache:
    user_proxy.initiate_chat(
        planner,
        message="Develop a Python script to calculate the factorial of 5.",
        cache=cache # Pass the cache object
    )

# For production or shared environments, consider RedisCache
# with Cache.redis(redis_url="redis://localhost:6379/0") as cache:
#     user_proxy.initiate_chat(planner, message="...", cache=cache)
```
**Note**: Set `cache_seed=None` in `llm_config` to explicitly disable caching for an agent, but this should be rare and justified.

## 3. Type Hints for Clarity and Maintainability

Utilize Python type hints extensively. This improves code readability, enables static analysis, and reduces bugs in complex multi-agent systems.

❌ BAD: Untyped function signatures.
```python
def setup_conversation(agent1, agent2, task):
    # ... logic ...
    pass
```

✅ GOOD: Fully typed function signatures.
```python
from autogen import Agent, GroupChatManager, UserProxyAgent
from typing import List

def setup_conversation(
    agents: List[Agent],
    manager: GroupChatManager,
    initial_message: str
) -> None:
    """
    Initiates a multi-agent conversation with a given manager and agents.
    """
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    user_proxy.initiate_chat(manager, message=initial_message)
```

## 4. Robust Testing Approaches

Implement a multi-layered testing strategy:
1.  **Unit Tests**: For individual tool functions and agent helper methods.
2.  **Integration Tests**: For multi-agent conversations, ensuring workflows complete as expected. Leverage caching to make these tests fast and deterministic.

```python
# tests/test_tools.py
import pytest

# Example tool function
def calculate_square(number: int) -> int:
    return number * number

def test_calculate_square():
    assert calculate_square(5) == 25
    assert calculate_square(0) == 0
    assert calculate_square(-2) == 4

# tests/test_workflow.py
import pytest
from autogen import Cache, GroupChatManager, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from agents.planner import create_planner_agent
from agents.coder import create_coder_agent

@pytest.fixture(scope="module")
def llm_client():
    return OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )

@pytest.fixture(scope="module")
def planner_agent(llm_client):
    return create_planner_agent(llm_client)

@pytest.fixture(scope="module")
def coder_agent(llm_client):
    return create_coder_agent(llm_client)

def test_factorial_workflow(planner_agent, coder_agent):
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )
    manager = GroupChatManager(
        agents=[user_proxy, planner_agent, coder_agent],
        messages=[],
        llm_config={"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}
    )

    with Cache.disk(cache_path_root=".autogen_test_cache") as cache:
        user_proxy.initiate_chat(
            manager,
            message="Calculate the factorial of 5 using Python.",
            cache=cache
        )
        # Assert that the final message contains the correct answer
        final_message = manager.last_message()["content"]
        assert "120" in final_message
```

## 5. Environment Management and Packaging

While not `autogen`-specific, these are crucial for any Python project.

*   **Virtual Environments**: ALWAYS use `venv` or `conda` for dependency isolation.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
*   **Packaging**: For reusable agents or applications, define `pyproject.toml` (Poetry/Rye) or `setup.py` for proper distribution.
    ```toml
    # pyproject.toml (example snippet)
    [project]
    name = "my-autogen-app"
    version = "0.1.0"
    dependencies = [
        "pyautogen~=0.2.8", # Pin to specific versions
        "autogen-ext[openai]",
        "diskcache",
        "python-dotenv",
        # ... other dependencies
    ]
    ```

## 6. Secure API Key Handling

NEVER hardcode API keys. Use environment variables.

❌ BAD: Hardcoded API key.
```python
llm_config = {"model": "gpt-4", "api_key": "sk-YOUR_HARDCODED_KEY"}
```

✅ GOOD: Environment variables.
```python
import os
from dotenv import load_dotenv

load_dotenv() # Load from .env file in development

llm_config = {
    "model": "gpt-4",
    "api_key": os.environ.get("OPENAI_API_KEY")
}
if not llm_config["api_key"]:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
```
```
# .env file (add to .gitignore)
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

By adhering to these guidelines, our `autogen` projects will be maintainable, performant, and reliable.