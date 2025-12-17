---
description: Definitive guidelines for building production-ready LlamaIndex applications, emphasizing modularity, type safety, cloud services, and robust testing.
globs: **/*.py
---
# LlamaIndex Best Practices

LlamaIndex is the definitive framework for building context-augmented LLM applications. Follow these rules to ensure your RAG pipelines and agents are modular, type-safe, performant, and production-ready in 2025.

## 1. Code Organization and Structure

Organize your LlamaIndex components into distinct, testable functions or classes. Leverage the `Workflow` API for explicit data flow.

### ✅ GOOD: Modular Functions & Workflow API

```python
# core_components.py
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.workflow import Workflow, LlamaAgentWorker, AgentInput, AgentOutput
from typing import List

def configure_global_settings() -> None:
    """Configures global LLM and embedding models for consistent behavior."""
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def load_documents(data_path: str) -> List[Document]:
    """Loads documents from a directory using SimpleDirectoryReader."""
    return SimpleDirectoryReader(data_path).load_data()

def build_vector_index(documents: List[Document]) -> VectorStoreIndex:
    """Builds a VectorStoreIndex from documents."""
    return VectorStoreIndex.from_documents(documents)

def create_query_engine(index: VectorStoreIndex):
    """Creates a query engine with optimal response mode for citations."""
    return index.as_query_engine(response_mode="compact", verbose=True)

# main_app.py
from core_components import (
    configure_global_settings, load_documents, build_vector_index, create_query_engine
)

def main_rag_workflow():
    configure_global_settings() # Always configure settings first

    docs = load_documents("./data")
    index = build_vector_index(docs)
    query_engine = create_query_engine(index)

    # Wrap the query engine in an AgentWorker for Workflow compatibility
    agent_worker = LlamaAgentWorker.from_query_engine(query_engine)
    rag_workflow = Workflow(
        name="SimpleRAGWorkflow",
        description="A basic RAG pipeline for document querying.",
        input_type=AgentInput,
        output_type=AgentOutput,
        worker=agent_worker
    )

    result = rag_workflow.run(AgentInput(query="What are the main themes across these documents?"))
    print(result.response)

if __name__ == "__main__":
    main_rag_workflow()
```

### ❌ BAD: Monolithic Script

```python
# app.py (avoid this structure for anything beyond quickstarts)
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

docs = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(docs)
qe = index.as_query_engine()
ans = qe.query("What are the main themes?")
print(ans)
```

## 2. Common Patterns and Anti-patterns

### ✅ GOOD: Leverage LlamaCloud Services & LlamaHub Connectors

For production-grade RAG, prefer managed services like LlamaParse and LlamaCloud, and utilize LlamaHub for data connectors.

```python
import os
from llama_index.core.readers import LlamaParseReader
from llama_index.cloud import LlamaCloudIndex
from llama_index.core import Document

# Ensure LLAMAPARSE_API_KEY and LLAMACLOUD_API_KEY are set in environment
os.environ["LLAMAPARSE_API_KEY"] = os.getenv("LLAMAPARSE_API_KEY")
os.environ["LLAMACLOUD_API_KEY"] = os.getenv("LLAMACLOUD_API_KEY")

# Use LlamaParse for robust PDF/Office document parsing (preserves structure)
parser = LlamaParseReader(result_type="markdown", parsing_instruction="Extract all tables and figures.")
documents: List[Document] = parser.load_data("path/to/complex_report.pdf")

# Store indexes in LlamaCloud for automatic scaling, versioning, and management
cloud_index = LlamaCloudIndex.from_documents(
    documents,
    name="my-production-rag-index",
    project_name="my-enterprise-rag",
    # auto_delete=False # Keep index after script finishes
)
query_engine = cloud_index.as_query_engine()
response = query_engine.query("Summarize the key findings in the report.")
print(response)
```

### ❌ BAD: Custom Scraping & Basic Readers for Complex Data

```python
# Avoid custom web scraping when LlamaHub offers a robust connector.
# Avoid SimpleDirectoryReader for complex PDFs with tables/figures; it will lose structure.
from llama_index.core.readers import SimpleDirectoryReader
# docs = SimpleDirectoryReader("./complex_pdfs").load_data() # Leads to poor retrieval
```

## 3. Performance Considerations

Optimize ingestion, indexing, and retrieval for speed and cost.

### ✅ GOOD: Asynchronous Operations & Efficient Chunking

```python
import asyncio
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from typing import List

async def build_index_async(documents: List[Document]) -> VectorStoreIndex:
    """Asynchronously builds a VectorStoreIndex."""
    # For very large datasets, consider streaming or batching document processing
    index = await asyncio.to_thread(VectorStoreIndex.from_documents, documents)
    return index

async def main_performance_example():
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    # Load documents (can be async if using async readers)
    docs = [Document(text=f"Sample document {i} content. " * 100) for i in range(100)]
    
    index = await build_index_async(docs)
    query_engine = index.as_query_engine()
    response = await asyncio.to_thread(query_engine.query, "What is in the documents?")
    print("Query completed:", response)

if __name__ == "__main__":
    asyncio.run(main_performance_example())
```

### ❌ BAD: Synchronous, Unoptimized Processing

```python
# Synchronous processing of many documents is slow and blocks the event loop.
# Default chunking without consideration for content structure also hurts performance and quality.
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
# docs = SimpleDirectoryReader("./large_data_folder").load_data() # Can block
# index = VectorStoreIndex.from_documents(docs) # Can be memory intensive and slow
```

## 4. Common Pitfalls and Gotchas

### 4.1. Environment Variables for Secrets

Always use environment variables for API keys and sensitive credentials.

### ✅ GOOD: Load from Environment

```python
import os
# Set these in your shell or .env file, e.g., OPENAI_API_KEY="sk-..."
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LLAMAPARSE_API_KEY"] = os.getenv("LLAMAPARSE_API_KEY")
```

### ❌ BAD: Hardcoding Secrets

```python
# NEVER hardcode API keys directly in code
OPENAI_API_KEY = "sk-YOUR_HARDCODED_KEY_HERE"
```

### 4.2. Chunking Strategy

Effective chunking is critical for retrieval quality.

### ✅ GOOD: Semantic Chunking with Metadata & Overlap

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# Configure chunking globally for consistency
Settings.text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator=" " # Prefer space separator for better word boundaries
)
# For complex documents, LlamaParse handles intelligent, structure-aware chunking.
```

### ❌ BAD: Default or Naive Chunking

```python
# Relying solely on default chunking can lead to poor context and retrieval.
# Default is often fine for simple text, but not for structured documents or long texts.
# Settings.text_splitter = None # Don't explicitly disable if you need control
```

## 5. Type Hints

Enforce type safety for maintainable and robust LlamaIndex applications.

### ✅ GOOD: Comprehensive Type Annotations

```python
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from typing import List

def process_and_query_documents(documents: List[Document], query_str: str) -> str:
    """Processes documents and queries the index, returning the response as a string."""
    index: VectorStoreIndex = VectorStoreIndex.from_documents(documents)
    query_engine: BaseQueryEngine = index.as_query_engine()
    response = query_engine.query(query_str)
    return str(response)
```

### ❌ BAD: Untyped Functions

```python
def process_and_query_untyped(documents, query_str): # Lacks clarity on expected types
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(query_str)
    return str(response)
```

## 6. Virtual Environments

Isolate project dependencies to prevent conflicts and ensure reproducible builds.

### ✅ GOOD: Use `venv` or `poetry`

```bash
# Using venv (standard Python approach)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Using Poetry (recommended for robust dependency management)
poetry init --name my-llamaindex-app --python ">=3.9,<3.12"
poetry add llama-index llama-index-llms-openai llama-index-embeddings-openai
poetry shell
```

### ❌ BAD: Global `pip install`

```bash
# Avoid installing directly into your system's Python environment.
# This leads to dependency conflicts and "works on my machine" problems.
pip install llama-index # Can break other projects
```

## 7. Packaging

Define project dependencies clearly for reproducibility and deployment.

### ✅ GOOD: `pyproject.toml` (Poetry/Rye) or `requirements.txt`

```toml
# pyproject.toml (Poetry example - preferred for modern Python projects)
[tool.poetry]
name = "my-llamaindex-app"
version = "0.1.0"
description = "A production-ready LlamaIndex RAG application."
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9,<3.12"
llama-index = "^0.10.0" # Use appropriate, pinned version ranges
llama-index-llms-openai = "^0.1.0"
llama-index-embeddings-openai = "^0.1.0"
# Add other necessary packages like llama-index-cloud, llama-index-readers-llamaparse
```

### ❌ BAD: Undocumented Dependencies

```python
# No explicit dependency list makes it hard for others to set up the project.
# (Implicitly relying on a global environment or manual installs is a recipe for errors.)
```

## 8. Testing Approaches

Implement unit and integration tests for reliability. Mock LLM calls for faster, cheaper unit tests.

### ✅ GOOD: Unit Tests with Mocking

```python
# tests/test_rag_components.py
import pytest
from unittest.mock import MagicMock
from llama_index.core import Document, Settings
from core_components import load_documents, build_vector_index, create_query_engine

@pytest.fixture
def mock_documents():
    return [
        Document(text="LlamaIndex is a data framework for LLM applications."),
        Document(text="It connects LLMs to your private data sources.")
    ]

def test_load_documents_from_dir(tmp_path):
    # Create dummy files for testing SimpleDirectoryReader
    (tmp_path / "doc1.txt").write_text("Content of document one.")
    (tmp_path / "doc2.txt").write_text("Content of document two.")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 2
    assert "Content of document one." in docs[0].text

def test_build_vector_index(mock_documents):
    # Mock LLM and embedding models for fast, isolated unit tests
    Settings.llm = MagicMock()
    Settings.embed_model = MagicMock()
    index = build_vector_index(mock_documents)
    assert index is not None
    # Further assertions can check index properties if needed

def test_query_engine_response_mocked():
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Mocked response: LlamaIndex connects LLMs to your data."
    response = mock_query_engine.query("What is LlamaIndex?")
    assert "Mocked response" in str(response)
    mock_query_engine.query.assert_called_once_with("What is LlamaIndex?")
```

### ❌ BAD: No Tests or Relying on End-to-End Only

```python
# No test files, or only manual testing.
# Relying solely on slow, expensive end-to-end tests makes development cycles long and debugging difficult.
# Unit tests are crucial for isolating issues in specific components.
```