---
description: Definitive guidelines for building robust, maintainable, and scalable multi-agent systems using CrewAI, focusing on modularity, clear role definition, and disciplined LLM configuration.
globs: **/*.py
---
# CrewAI Best Practices

This document outlines the definitive best practices for developing with CrewAI, ensuring your multi-agent systems are maintainable, performant, and production-ready. Adhere to these guidelines for all new and existing CrewAI projects.

## 1. Code Organization and Structure

Always modularize your agents, tasks, and crews into dedicated Python modules. This promotes reusability, version control, and clear separation of concerns.

### 1.1. Dedicated Modules for Agents, Tasks, and Crews

**Always separate definitions.** Agents belong in `agents.py`, tasks in `tasks.py`, and the crew orchestration in `crew.py` (or `main.py`).

❌ BAD: Monolithic file
```python
# main.py
from crewai import Agent, Task, Crew
researcher = Agent(role='Researcher', ...)
research_task = Task(description='Research topic', agent=researcher, ...)
crew = Crew(agents=[researcher], tasks=[research_task], ...)
```

✅ GOOD: Modular structure
```python
# agents.py
from crewai import Agent
from tools import search_tool # Assume tools.py exists

class ResearchAgents:
    def senior_researcher(self) -> Agent:
        return Agent(
            role='Senior Researcher',
            goal='Uncover critical insights and data points.',
            backstory='Expert in data analysis and synthesis.',
            verbose=True, allow_delegation=False, tools=[search_tool]
        )

# tasks.py
from crewai import Task
from agents import ResearchAgents

class ResearchTasks:
    def __init__(self):
        self.agents = ResearchAgents()

    def research_topic(self, topic: str) -> Task:
        return Task(
            description=f"Conduct comprehensive research on '{topic}'.",
            expected_output='A detailed report summarizing key findings.',
            agent=self.agents.senior_researcher(),
            async_execution=False
        )

# crew.py
from crewai import Crew, Process
from agents import ResearchAgents
from tasks import ResearchTasks

class MyCrew:
    def __init__(self, topic: str):
        self.topic = topic
        self.agents = ResearchAgents()
        self.tasks = ResearchTasks()

    def run(self):
        crew = Crew(
            agents=[self.agents.senior_researcher()],
            tasks=[self.tasks.research_topic(self.topic)],
            process=Process.sequential,
            verbose=2
        )
        return crew.kickoff()

if __name__ == "__main__":
    result = MyCrew("AI Agent Frameworks in 2025").run()
    print(result)
```

### 1.2. Clear Agent Role Definition

Every agent *must* have explicit `role`, `goal`, and `backstory`. These are the primary drivers of agent behavior.

❌ BAD: Vague agent
```python
researcher = Agent(role='Researcher')
```

✅ GOOD: Detailed agent
```python
researcher = Agent(
    role='Senior Market Analyst',
    goal='Provide actionable insights on emerging AI trends.',
    backstory='With years of experience in tech market analysis, I excel at identifying key drivers and potential disruptors.',
    verbose=True, allow_delegation=False
)
```

## 2. LLM Selection and Configuration

Match LLM capabilities to agent requirements. Always externalize LLM configuration.

### 2.1. Strategic LLM Selection

Choose LLMs based on task complexity, cost, and latency. For complex reasoning, use powerful models (e.g., GPT-4o, Claude Opus). For simple, high-volume tasks, use faster, cheaper models.

❌ BAD: Arbitrary LLM choice
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo") # No justification
```

✅ GOOD: Purpose-driven LLM choice
```python
from langchain_openai import ChatOpenAI
# For complex analysis where accuracy is paramount
complex_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
# For quick, high-volume data retrieval
fast_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Assign to agents based on their role
researcher = Agent(llm=complex_llm, ...)
summarizer = Agent(llm=fast_llm, ...)
```

### 2.2. Externalized LLM Configuration

Never hardcode API keys or sensitive configurations. Use environment variables.

❌ BAD: Hardcoded API key
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key="sk-YOUR_HARDCODED_KEY")
```

✅ GOOD: Environment variables (`.env`)
```python
# .env file (NOT committed to Git)
OPENAI_API_KEY="sk-YOUR_ACTUAL_API_KEY"

# config.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv() # Load environment variables from .env

def get_openai_llm(model: str = "gpt-4o", temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )

# Usage in agents.py or crew.py
from config import get_openai_llm
complex_llm = get_openai_llm("gpt-4o")
```

## 3. Common Patterns and Anti-patterns

### 3.1. Agent Collaboration with `context`

Leverage `context` in tasks to enable agents to build upon each other's work. This is fundamental to multi-agent systems.

❌ BAD: Isolated tasks
```python
task1 = Task(description='Research X', agent=researcher)
task2 = Task(description='Summarize Y', agent=summarizer) # Y is not from X
```

✅ GOOD: Chained tasks with `context`
```python
research_task = Task(
    description='Research quantum computing trends.',
    expected_output='A comprehensive report.', agent=researcher
)

analysis_task = Task(
    description='Analyze the research report and identify key investment opportunities.',
    expected_output='A list of 3-5 high-potential investment opportunities.',
    agent=analyst,
    context=[research_task] # Analyst uses output of researcher
)
```

### 3.2. Effective Tool Usage

Equip agents with specific tools for specific jobs. Agents should not "hallucinate" tool usage.

❌ BAD: Agent without necessary tools
```python
# Agent needs to search but has no search tool
researcher = Agent(role='Researcher', goal='Find data', ...)
```

✅ GOOD: Agent with relevant tools
```python
from crewai_tools import SerperDevTool # Example search tool
search_tool = SerperDevTool()

researcher = Agent(
    role='Data Gatherer',
    goal='Collect up-to-date information on market shifts.',
    backstory='My expertise lies in rapidly extracting relevant data from the web.',
    tools=[search_tool], # Agent can now use search_tool
    verbose=True
)
```

## 4. Performance Considerations

Optimize for cost and speed by managing verbosity, process flow, and asynchronous execution.

### 4.1. Controlled Verbosity

Use `verbose=1` for production, `verbose=2` for development/debugging. Excessive verbosity impacts performance.

❌ BAD: Always `verbose=2` in production
```python
crew = Crew(..., verbose=2)
```

✅ GOOD: Conditional verbosity
```python
import os
is_dev = os.getenv("ENV", "dev") == "dev"
crew = Crew(..., verbose=2 if is_dev else 1)
```

### 4.2. Process Flow and Asynchronous Tasks

Choose `Process.sequential` for dependent tasks and `Process.hierarchical` for complex, multi-layered coordination. Use `async_execution=True` for tasks that can run in parallel.

❌ BAD: Defaulting to sequential for independent tasks
```python
# These tasks could run in parallel
task1 = Task(description='Scrape website A', agent=scraper)
task2 = Task(description='Scrape website B', agent=scraper)
crew = Crew(tasks=[task1, task2], process=Process.sequential)
```

✅ GOOD: Parallel execution for independent tasks
```python
task1 = Task(description='Scrape website A', agent=scraper, async_execution=True)
task2 = Task(description='Scrape website B', agent=scraper, async_execution=True)
crew = Crew(tasks=[task1, task2], process=Process.sequential) # Crew still sequential, but tasks run in parallel
```

## 5. Common Pitfalls and Gotchas

### 5.1. Agent Drift and Hallucination

Agents can stray from their goal or invent information. Mitigate this with clear `goal`, `backstory`, specific `tools`, and `allow_delegation=False` where appropriate.

❌ BAD: Agent frequently goes off-topic
```python
agent = Agent(role='Generalist', goal='Explore AI', temperature=0.8)
```

✅ GOOD: Focused agent
```python
agent = Agent(
    role='Specific Data Extractor',
    goal='Extract financial data from quarterly reports.',
    backstory='I am precise and only focus on numerical data.',
    tools=[pdf_reader_tool],
    allow_delegation=False, # Prevents delegating
    temperature=0.1 # Lower temperature for factual tasks
)
```

### 5.2. Infinite Loops

Agents can get stuck in repetitive cycles. Set `max_rpm` (requests per minute