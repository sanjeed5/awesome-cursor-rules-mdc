---
description: This guide provides definitive, actionable best practices for writing maintainable, performant, and type-safe NLTK code in Python, emphasizing modern patterns and avoiding common pitfalls.
globs: **/*.py
---
# `nltk` Best Practices

NLTK remains a cornerstone for classic NLP tasks. Integrate it effectively with modern Python by following these guidelines.

## 1. Code Organization and Structure

Organize NLTK operations into modular, reusable components.

### 1.1 Modularize NLP Stages
Encapsulate each NLP step (tokenization, stemming, POS tagging) in a dedicated, pure function or small class. This promotes testability and allows easy swapping of implementations (e.g., NLTK to spaCy).

❌ **BAD: Monolithic Script**
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = "NLTK is a powerful library for natural language processing."
tokens = nltk.word_tokenize(text)
filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print(stemmed_tokens)
```

✅ **GOOD: Modular Functions**
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into words."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: List[str], lang: str = 'english') -> List[str]:
    """Removes common stopwords from a list of tokens."""
    stop_words = set(stopwords.words(lang))
    return [word for word in tokens if word.lower() not in stop_words]

def stem_tokens(tokens: List[str]) -> List[str]:
    """Applies Porter Stemming to a list of tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

# Usage
document = "NLTK is a powerful library for natural language processing."
tokens = tokenize_text(document)
filtered = remove_stopwords(tokens)
stemmed = stem_tokens(filtered)
print(stemmed)
```

## 2. Common Patterns and Anti-patterns

### 2.1 Lazy Data Loading
Download NLTK corpora *once* during setup or on first run, not repeatedly. Guard `nltk.download()` calls.

❌ **BAD: Repeated Downloads**
```python
import nltk
# This will run every time the script executes
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text: str) -> List[str]:
    # ...
    pass
```

✅ **GOOD: Lazy, Guarded Download**
```python
import nltk
import os

def ensure_nltk_data(corpus_name: str):
    """Downloads NLTK data if not already present."""
    try:
        nltk.data.find(f'corpora/{corpus_name}')
    except nltk.downloader.DownloadError:
        print(f"Downloading NLTK data: {corpus_name}...")
        nltk.download(corpus_name)
        print(f"Downloaded {corpus_name}.")

# In your application's entry point or setup script:
# ensure_nltk_data('punkt')
# ensure_nltk_data('stopwords')

# In your processing logic, assume data is available
def process_text(text: str) -> List[str]:
    # ...
    pass
```

### 2.2 Avoid Global State
Pass NLTK objects (like tokenizers, stemmers, lemmatizers) explicitly to functions. This improves testability and thread safety.

❌ **BAD: Global Stemmer Instance**
```python
from nltk.stem import PorterStemmer
_GLOBAL_STEMMER = PorterStemmer() # Global state

def process_document(text: str) -> List[str]:
    tokens = nltk.word_tokenize(text)
    return [_GLOBAL_STEMMER.stem(t) for t in tokens]
```

✅ **GOOD: Pass Objects Explicitly**
```python
from nltk.stem import PorterStemmer
from typing import List, Callable

# Define a type alias for clarity
StemmerFunc = Callable[[str], str]

def process_document(text: str, stemmer: StemmerFunc) -> List[str]:
    tokens = nltk.word_tokenize(text)
    return [stemmer(t) for t in tokens]

# Usage
porter_stemmer = PorterStemmer().stem # Get the bound method
processed_text = process_document("running ran runner", stemmer=porter_stemmer)
print(processed_text)
```

## 3. Performance Considerations

### 3.1 Choose the Right Tokenizer
For large corpora, `RegexpTokenizer` or `ToktokTokenizer` are significantly faster than `nltk.word_tokenize`.

❌ **BAD: Default `word_tokenize` for Large Scale**
```python
import nltk
# Slow for very large texts
tokens = nltk.word_tokenize(long_text_data)
```

✅ **GOOD: Faster Tokenizers for Production**
```python
from nltk.tokenize import RegexpTokenizer, ToktokTokenizer
from typing import List

def tokenize_regex(text: str) -> List[str]:
    """Uses RegexpTokenizer for faster word tokenization."""
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text)

def tokenize_toktok(text: str) -> List[str]:
    """Uses ToktokTokenizer for robust, faster word tokenization."""
    tokenizer = ToktokTokenizer()
    return tokenizer.tokenize(text)

long_text_data = "This is some very long text that needs efficient tokenization." * 1000
# Prefer one of these for performance
tokens_regex = tokenize_regex(long_text_data)
tokens_toktok = tokenize_toktok(long_text_data)
```

## 4. Common Pitfalls and Gotchas

### 4.1 Lemmatization Requires POS Tags
`WordNetLemmatizer` performs better when provided with a Part-of-Speech (POS) tag. Without it, it defaults to noun ('n').

❌ **BAD: Lemmatizing without POS**
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running")) # Output: 'running' (incorrect for verb)
```

✅ **GOOD: Lemmatizing with POS Tags**
```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from typing import List, Tuple

def get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tag to WordNet POS tag."""
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN # Default to noun

def lemmatize_with_pos(tokens: List[str]) -> List[str]:
    """Lemmatizes tokens using WordNetLemmatizer with POS tags."""
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = nltk.pos_tag(tokens) # Requires 'averaged_perceptron_tagger'
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens]

# Ensure data is downloaded:
# ensure_nltk_data('wordnet')
# ensure_nltk_data('averaged_perceptron_tagger')

tokens = ["The", "cats", "are", "running", "fast"]
lemmas = lemmatize_with_pos(tokens)
print(lemmas) # Output: ['The', 'cat', 'be', 'run', 'fast']
```

## 5. Type Hints

Always use type hints for NLTK functions, especially for inputs (`str`, `List[str]`) and outputs (`List[str]`, `nltk.Tree`). This improves readability, enables static analysis, and enhances IDE support.

```python
import nltk
from nltk.tree import Tree
from typing import List, Tuple

def pos_tag_sentence(tokens: List[str]) -> List[Tuple[str, str]]:
    """Performs Part-of-Speech tagging on a list of tokens."""
    # Requires 'averaged_perceptron_tagger'
    return nltk.pos_tag(tokens)

def chunk_entities(tagged_tokens: List[Tuple[str, str]]) -> Tree:
    """Chunks named entities from POS-tagged tokens."""
    # Requires 'maxent_ne_chunker' and 'words'
    return nltk.chunk.ne_chunk(tagged_tokens)

# Usage
sentence_tokens = ["Apple", "Inc.", "is", "headquartered", "in", "Cupertino", "."]
tagged = pos_tag_sentence(sentence_tokens)
entities = chunk_entities(tagged)
print(entities)
```

## 6. Virtual Environments and Packaging

### 6.1 Use Virtual Environments
Always develop NLTK projects within a virtual environment. This isolates dependencies and prevents conflicts.

```bash
# Create a virtual environment
python -m venv .venv
# Activate it
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows
# Install NLTK and other dependencies
pip install nltk
```

### 6.2 Manage Dependencies with `pyproject.toml`
Use `pyproject.toml` for modern dependency management and project configuration.

```toml
# pyproject.toml
[project]
name = "my-nltk-project"
version = "0.1.0"
dependencies = [
    "nltk>=3.8",
    "regex", # Often useful with NLTK
]

[tool.black]
line-length = 79

[tool.ruff]
line-length = 79
select = ["E", "F", "W", "I"] # Common rules
```

## 7. Testing Approaches

### 7.1 Unit Test Pure Functions
Since NLTK operations are often encapsulated in pure functions, unit testing is straightforward.

```python
# test_nlp_utils.py
import unittest
from unittest.mock import patch
from my_module import remove_stopwords, tokenize_text # Assuming your functions are in my_module

class TestNlpUtils(unittest.TestCase):
    def test_remove_stopwords(self):
        tokens = ["This", "is", "a", "test", "sentence", "for", "stopwords"]
        expected = ["test", "sentence", "stopwords"]
        # Mock stopwords to ensure test isolation
        with patch('nltk.corpus.stopwords.words', return_value=['this', 'is', 'a', 'for']):
            self.assertEqual(remove_stopwords(tokens), expected)

    def test_tokenize_text(self):
        text = "Hello, world!"
        # Mock nltk.word_tokenize to control behavior
        with patch('nltk.word_tokenize', return_value=['Hello', ',', 'world', '!']):
            self.assertEqual(tokenize_text(text), ['Hello', ',', 'world', '!'])

if __name__ == '__main__':
    unittest.main()
```