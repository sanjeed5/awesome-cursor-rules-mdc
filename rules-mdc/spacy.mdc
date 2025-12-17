---
description: This guide provides definitive, opinionated best practices for writing maintainable, performant, and robust spaCy code in Python, focusing on modern patterns and avoiding common pitfalls.
globs: **/*.py
---
# spaCy Best Practices

spaCy is the backbone of our NLP systems. Adhering to these guidelines ensures our pipelines are performant, reproducible, and easy to maintain.

## 1. Project Organization & Configuration

Always structure your spaCy projects using `spacy project` and define all pipeline settings in a declarative YAML config. This is non-negotiable for reproducibility and scalability.

**✅ GOOD: Use `spacy project` and declarative configs.**
Organize your pipeline logic in a dedicated `src/pipeline/` package. Use `project.yml` to manage workflows and `config.cfg` for all spaCy pipeline settings.

```python
# src/pipeline/custom_component.py
from spacy.language import Language
from spacy.tokens import Doc

@Language.factory("my_custom_component")
def create_my_component(nlp: Language, name: str):
    return MyCustomComponent(nlp, name)

class MyCustomComponent:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        # Custom logic here
        return doc

# project.yml (simplified)
# ...
workflows:
  train:
    - "python -m spacy train config.cfg --output models/"
  package:
    - "python -m spacy package models/en_core_web_v1.0.0 ./dist --build wheel"

# config.cfg (simplified)
[nlp]
lang = "en"
pipeline = ["tok2vec", "ner", "my_custom_component"]

[components.my_custom_component]
factory = "my_custom_component"
```

**❌ BAD: Ad-hoc scripts and hardcoded parameters.**
Avoid scattering pipeline logic across multiple scripts or hardcoding model paths and hyperparameters. This makes experiments non-reproducible and deployment fragile.

```python
# bad_script.py
import spacy
# Parameters hardcoded or passed via CLI args, not in a central config
MODEL_PATH = "path/to/my/model"
THRESHOLD = 0.7

nlp = spacy.load(MODEL_PATH)
# ... pipeline components added programmatically ...
```

## 2. Type Hinting

Strictly use type hints for all spaCy objects (`Language`, `Doc`, `Span`, `Token`). This improves code readability, enables static analysis, and reduces runtime errors.

**✅ GOOD: Comprehensive type hints.**
Annotate function arguments and return types with precise spaCy types.

```python
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from typing import List

def process_document(nlp: Language, text: str) -> Doc:
    """Processes text with a spaCy pipeline."""
    return nlp(text)

def extract_entities(doc: Doc) -> List[Span]:
    """Extracts named entities from a processed Doc."""
    return list(doc.ents)

def get_token_lemma(token: Token) -> str:
    """Returns the lemma of a single token."""
    return token.lemma_
```

**❌ BAD: Untyped spaCy code.**
Omitting type hints makes code harder to understand and refactor.

```python
# bad_types.py
import spacy

def process_document(nlp, text): # What are nlp and text? What does it return?
    return nlp(text)

def extract_entities(doc): # What is 'doc'?
    return list(doc.ents)
```

## 3. Performance Considerations

Optimize for speed and memory by disabling unused pipeline components and processing text in batches.

**✅ GOOD: Efficient pipeline usage.**
Use `nlp.select_pipes` for inference to only run necessary components. Process multiple documents with `nlp.pipe`.

```python
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from typing import Iterator

def analyze_text_batch(nlp: Language, texts: List[str]) -> Iterator[Doc]:
    """Processes a batch of texts, only running NER."""
    # Temporarily disable unused pipes for performance during inference
    with nlp.select_pipes(enable=["ner"]):
        yield from nlp.pipe(texts, batch_size=50)

# Example usage
nlp_full = spacy.load("en_core_web_sm")
documents = ["Apple is looking at buying U.K. startup.", "Tim Cook is CEO of Apple."]
for doc in analyze_text_batch(nlp_full, documents):
    print(f"Text: {doc.text}, Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
```

**❌ BAD: Inefficient processing.**
Loading full pipelines for simple tasks or processing documents one by one in a loop is wasteful.

```python
# bad_performance.py
import spacy

nlp = spacy.load("en_core_web_lg") # Loads all components, even if only tokenization is needed
texts = ["Text 1", "Text 2", "Text 3"]

for text in texts: # Processes one by one, no batching
    doc = nlp(text)
    # ... only use doc.text or doc.tokens, ignoring NER, POS, etc.
```

## 4. Virtual Environments & Dependencies

Always use a virtual environment and pin your `spacy` version in `requirements.txt`. This prevents dependency conflicts and ensures consistent environments.

**✅ GOOD: Pinned dependencies in a virtual environment.**

```bash
# Terminal
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```ini
# requirements.txt
spacy==3.8.*
spacy-llm==0.8.* # If using LLM integration
# Other dependencies...
```

**❌ BAD: Global installs or unpinned versions.**
Installing `spacy` globally or using `spacy>=3.0` can lead to unexpected behavior and "works on my machine" issues.

```ini
# requirements.txt (BAD)
spacy # No version specified, will install latest
```

## 5. Custom Components & Extension Attributes

Extend spaCy's `Doc`, `Span`, and `Token` objects using custom components and extension attributes. This keeps your custom logic integrated and accessible.

**✅ GOOD: Custom components and extension attributes.**

```python
# src/pipeline/custom_component.py
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

# Register extension attribute
if not Doc.has_extension("custom_sentiment"):
    Doc.set_extension("custom_sentiment", default=0.0)

@Language.factory("sentiment_analyzer")
def create_sentiment_analyzer(nlp: Language, name: str):
    return SentimentAnalyzer(nlp, name)

class SentimentAnalyzer:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        # Simple sentiment logic
        score = sum(1 for token in doc if token.text == "good") - sum(1 for token in doc if token.text == "bad")
        doc._.custom_sentiment = float(score)
        return doc

# config.cfg
# ...
[nlp]
pipeline = ["tok2vec", "ner", "sentiment_analyzer"]

[components.sentiment_analyzer]
factory = "sentiment_analyzer"

# Usage
nlp = spacy.load("path/to/your/trained_model") # Load model with custom component
doc = nlp("This is a good movie, not bad at all.")
print(f"Sentiment: {doc._.custom_sentiment}") # Access custom attribute
```

**❌ BAD: Storing custom data separately or modifying core attributes.**
Avoid maintaining parallel data structures or attempting to directly modify `Doc` attributes that are meant to be read-only.

```python
# bad_extensions.py
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Some text.")

# ❌ BAD: Storing custom data in a separate dictionary
custom_data = {doc.text: {"sentiment": 0.5}}

# ❌ BAD: Attempting to modify read-only attributes (will raise error)
# doc.user_data = {"my_field": "value"}
```

## 6. Testing

Write focused unit tests for custom components and integration tests for pipeline behavior using `pytest`. Use minimal `nlp` objects for unit tests.

**✅ GOOD: Targeted unit tests with `pytest`.**

```python
# tests/test_custom_component.py
import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc

# Assuming custom_component.py is in src/pipeline/
# from src.pipeline.custom_component import create_my_component, MyCustomComponent

# Register factory for testing
@Language.factory("test_my_custom_component")
def create_test_my_component(nlp: Language, name: str):
    return MyCustomComponent(nlp, name)

class MyCustomComponent: # Define a minimal version for testing
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name
    def __call__(self, doc: Doc) -> Doc:
        doc.set_extension("processed_by_custom", default=True, force=True)
        return doc

def test_my_custom_component_adds_extension():
    nlp = spacy.blank("en")
    nlp.add_pipe("test_my_custom_component")
    doc = nlp("Hello world.")
    assert doc._.processed_by_custom is True

def test_my_custom_component_returns_doc():
    nlp = spacy.blank("en")
    nlp.add_pipe("test_my_custom_component")
    doc = nlp("Another test.")
    assert isinstance(doc, Doc)
```

**❌ BAD: Untested components or overly broad tests.**
Avoid relying solely on manual testing or writing integration tests that load full, complex models for every small change.

```python
# tests/bad_test.py
import spacy

def test_full_pipeline_output():
    # ❌ BAD: Loads a large, slow model for a simple test
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("This is a test sentence.")
    assert len(doc.ents) > 0 # Fragile, depends on model behavior
```

## 7. LLM Integration

For integrating Large Language Models, leverage the `spacy-llm` package. It provides a structured, component-based approach that aligns with spaCy's pipeline philosophy.

**✅ GOOD: Use `spacy-llm` for structured LLM integration.**
Define LLM components in your `config.cfg` just like any other spaCy component.

```ini
# config.cfg (simplified for spacy-llm)
[nlp]
lang = "en"
pipeline = ["llm_ner"]

[components.llm_ner]
factory = "spacy.LLM"
task = {"@llm_tasks": "spacy.NER.v3"}
model = {"@llm_models": "spacy.GPT-3-5.v1"}
```

**❌ BAD: Ad-hoc LLM calls outside the pipeline.**
Directly calling LLM APIs in your application code bypasses spaCy's structured pipeline, making it harder to manage, test, and swap models.

```python
# bad_llm.py
import openai # or other LLM client
import spacy

def process_with_llm_and_spacy(text: str):
    # ❌ BAD: LLM call is separate from the spaCy pipeline
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Extract entities from: {text}",
        max_tokens=100
    )
    llm_entities = parse_llm_response(response)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Now you have two separate sets of entities, difficult to reconcile
    return doc, llm_entities
```