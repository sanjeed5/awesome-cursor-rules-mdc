---
description: This guide outlines definitive best practices for using the gensim library, focusing on reproducibility, efficient corpus construction, robust model training, and maintainable code for NLP topic modeling tasks.
globs: **/*.py
---
# gensim Best Practices

This document is your definitive guide for using `gensim` effectively and correctly within our team. We prioritize reproducibility, performance, and maintainability. Follow these rules to ensure consistent, high-quality NLP pipelines.

## 1. Ensure Reproducibility

Always configure logging and set a random seed at the entry point of any script using `gensim` models. This is non-negotiable for debugging and consistent results.

### ❌ BAD: Unpredictable Runs

```python
import gensim
from gensim import models, corpora

# No logging, no seed
# ... model training ...
lda_model = models.LdaModel(corpus, num_topics=10)
```

### ✅ GOOD: Reproducible and Observable Runs

```python
import logging
import numpy as np
import gensim
from gensim import models, corpora
from gensim.utils import randseed

# 1. Configure logging first
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 2. Set a global random seed for gensim and numpy
# Gensim's randseed sets numpy's seed internally.
randseed = 42
np.random.seed(randseed)

# ... rest of your script ...
# Ensure any model that takes a random_state or seed parameter uses it
lda_model = models.LdaModel(corpus, num_topics=10, random_state=randseed)
```

## 2. Construct Clean and Efficient Corpora

A well-prepared corpus is fundamental to effective topic modeling. Prioritize memory efficiency and intelligent vocabulary pruning.

### 2.1. Preprocessing with `gensim.utils.simple_preprocess` and `spaCy`

Combine `gensim`'s simple preprocessing with `spaCy` for robust tokenization and lemmatization. `simple_preprocess` handles basic tokenization and lowercasing efficiently.

### ❌ BAD: Manual, Inconsistent Preprocessing

```python
import re

documents = ["This is a document.", "Another document here."]
stoplist = set('is a here'.split())

texts = []
for doc in documents:
    # Manual lowercasing, splitting, and stopword removal
    tokens = [word for word in re.findall(r'\b\w+\b', doc.lower()) if word not in stoplist]
    texts.append(tokens)
```

### ✅ GOOD: `gensim.utils.simple_preprocess` + `spaCy` for Production

```python
import spacy
from gensim.utils import simple_preprocess

# Load spaCy model once
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def preprocess_document(text: str) -> list[str]:
    """
    Tokenizes, lowercases, removes stopwords, and lemmatizes text using spaCy.
    """
    # Use gensim's simple_preprocess for initial tokenization and lowercasing
    tokens = simple_preprocess(text, deacc=True) # deacc=True removes accents

    # Process with spaCy for lemmatization and advanced stopword removal
    doc = nlp(" ".join(tokens))
    lemmas = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return lemmas

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time"]

processed_texts = [preprocess_document(doc) for doc in documents]
```

### 2.2. Vocabulary Pruning with `filter_extremes`

Always prune your dictionary using `filter_extremes` to remove very rare and very common tokens. This significantly improves model quality and reduces noise.

**Crucial**: `no_below` is an absolute count (int), `no_above` is a fraction (float).

### ❌ BAD: Unfiltered Dictionary or Misunderstood Parameters

```python
from gensim import corpora

# Dictionary with all tokens, including noise
dictionary = corpora.Dictionary(processed_texts)

# Incorrectly using no_below as a fraction or no_above as an absolute count
# This will lead to unexpected vocabulary sizes.
dictionary.filter_extremes(no_below=0.05, no_above=100) # Incorrect units
```

### ✅ GOOD: Intelligent Vocabulary Pruning

```python
from gensim import corpora

dictionary = corpora.Dictionary(processed_texts)

# Filter out tokens that appear in:
# - less than 5 documents (no_below=5)
# - more than 60% of documents (no_above=0.6)
dictionary.filter_extremes(no_below=5, no_above=0.6)

# Compactify IDs after filtering
dictionary.compactify()
```

### 2.3. Streaming Corpora for Large Datasets

For large text collections, always use streaming corpora to keep only one document in memory at a time. This prevents `MemoryError` and allows processing arbitrarily large datasets.

### ❌ BAD: Loading Entire Corpus into Memory

```python
# This loads all documents into memory, which is fine for small datasets
# but will crash for large ones.
corpus = [dictionary.doc2bow(text) for text in all_documents_in_memory]
```

### ✅ GOOD: Streaming Corpus with `MmCorpus`

```python
from gensim.corpora import MmCorpus
from gensim.utils import simple_preprocess
import os

# Assume 'my_large_corpus.txt' contains one document per line
class StreamCorpus:
    def __init__(self, filepath, dictionary):
        self.filepath = filepath
        self.dictionary = dictionary

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Preprocess each line (document) on the fly
                tokens = simple_preprocess(line, deacc=True)
                yield self.dictionary.doc2bow(tokens)

# Example usage:
# First, build dictionary from a sample or a single pass
# For very large corpora, you might need to build the dictionary from a sample
# or iterate through the full corpus once to build it.
# For simplicity, let's assume `dictionary` is already built.

# Save corpus to disk in Matrix Market format for streaming
# This is a one-time operation if you need to persist the streamed corpus
# and iterate over it multiple times without re-preprocessing.
# For direct streaming from source, the StreamCorpus class is sufficient.
# MmCorpus.serialize('my_corpus.mm', StreamCorpus('my_large_corpus.txt', dictionary))

# Then, load and use the streamed corpus
# streamed_corpus = MmCorpus('my_corpus.mm')
# For direct streaming from source:
streamed_corpus = StreamCorpus('my_large_corpus.txt', dictionary)

# Now you can train models on `streamed_corpus` without loading it all into memory
# lda_model = models.LdaModel(streamed_corpus, num_topics=10)
```

## 3. Robust Model Training and Evaluation

Structure your model training for clarity, maintainability, and objective evaluation.

### 3.1. Code Organization and Type Hints

Separate preprocessing, model building, and evaluation into distinct, type-annotated functions or modules. This mirrors standard Python package structure and facilitates unit testing.

### ❌ BAD: Monolithic Script, Magic Numbers

```python
# main.py
import gensim
# ... lots of code ...
# Preprocessing, model training, and evaluation all mixed
lda = gensim.models.LdaModel(corpus, num_topics=5, passes=10)
# ... more code ...
```

### ✅ GOOD: Modular, Type-Hinted Functions

```python
# my_project/preprocessing.py
from typing import List
from gensim import corpora
import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def create_dictionary(documents: List[List[str]]) -> corpora.Dictionary:
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    dictionary.compactify()
    return dictionary

# my_project/models.py
from gensim import models, corpora
from typing import Optional

def train_lda_model(corpus: corpora.MmCorpus, dictionary: corpora.Dictionary,
                    num_topics: int, passes: int, random_state: Optional[int] = None) -> models.LdaModel:
    """Trains an LDA model with specified parameters."""
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state
    )

# my_project/evaluation.py
from gensim import models, corpora
from gensim.models import CoherenceModel

def evaluate_coherence(model: models.LdaModel, texts: List[List[str]],
                       dictionary: corpora.Dictionary, coherence_metric: str = 'c_v') -> float:
    """Calculates topic coherence for an LDA model."""
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence_metric
    )
    return coherence_model.get_coherence()

# main.py (entry point)
import logging
import numpy as np
from gensim.utils import randseed
from my_project.preprocessing import preprocess_document, create_dictionary
from my_project.models import train_lda_model
from my_project.evaluation import evaluate_coherence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
randseed = 42
np.random.seed(randseed)

if __name__ == "__main__":
    raw_documents = ["Doc 1 content...", "Doc 2 content..."] # Load from file in real app
    processed_texts = [preprocess_document(doc) for doc in raw_documents]
    dictionary = create_dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts] # Use streaming for large corpus

    # Hyperparameters documented explicitly
    NUM_TOPICS = 10
    NUM_PASSES = 15

    lda_model = train_lda_model(corpus, dictionary, NUM_TOPICS, NUM_PASSES, randseed)
    coherence_score = evaluate_coherence(lda_model, processed_texts, dictionary)

    logging.info(f"LDA Model Coherence (c_v): {coherence_score:.4f}")
```

### 3.2. Prefer LDA or LDA-Mallet

For topic modeling, `gensim`'s Latent Dirichlet Allocation (LDA) or its wrapper for [LDA-Mallet](https://radimrehurek.com/gensim/models/ldamallet.html) are the recommended choices. Evaluate topic quality using coherence scores.

```python
from gensim.models import LdaModel, LdaMallet
from gensim.models import CoherenceModel

# After training your LDA model:
# lda_model = LdaModel(...)

# Calculate coherence score
coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts,
                                     dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
logging.info(f"LDA Coherence: {coherence_lda}")

# If using LDA-Mallet (requires Mallet installation)
# mallet_path = '/path/to/mallet-2.0.8/bin/mallet' # Update this path
# ldamallet_model = LdaMallet(mallet_path, corpus=corpus, num_topics=NUM_TOPICS,
#                             id2word=dictionary, iterations=NUM_PASSES, random_state=randseed)

# coherence_model_ldamallet = CoherenceModel(model=ldamallet_model, texts=processed_texts,
#                                            dictionary=dictionary, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# logging.info(f"LDA-Mallet Coherence: {coherence_ldamallet}")
```

## 4. Performance Considerations

### 4.1. Serialize Transformed Corpora

If you apply a transformation (e.g., TF-IDF) to a corpus and iterate over the transformed corpus multiple times, serialize the result to disk. `gensim` transformations are often lazy, recomputing on each iteration.

```python
from gensim import models, corpora

# tfidf_model = models.TfidfModel(corpus)
# corpus_tfidf = tfidf_model[corpus] # This is a lazy wrapper

# If you iterate `corpus_tfidf` multiple times, serialize it:
# corpora.MmCorpus.serialize('/tmp/corpus_tfidf.mm', corpus_tfidf)
# Then load and use:
# persistent_corpus_tfidf = corpora.MmCorpus('/tmp/corpus_tfidf.mm')
```

## 5. Dependency Management

Always lock your dependencies using `requirements.txt` or `pyproject.toml` to ensure reproducibility across environments.

### ❌ BAD: Unspecified Dependencies

```
# No requirements.txt or pyproject.toml
# Relying on global environment or implicit versions
```

### ✅ GOOD: Locked Dependencies

```toml
# pyproject.toml (preferred for modern Python projects)
[project]
name = "my-gensim-project"
version = "0.1.0"
dependencies = [
    "gensim==4.3.2",
    "spacy==3.7.4",
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl",
    "numpy==1.26.4",
    "pyldavis==3.4.1",
]
```

## 6. Testing Approaches

Implement unit tests for your preprocessing functions, dictionary creation, and model evaluation logic. This ensures that changes don't silently break your NLP pipeline.

```python
# tests/test_preprocessing.py
import unittest
from my_project.preprocessing import preprocess_document, create_dictionary

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_document(self):
        text = "This is a test document with some stopwords."
        expected_tokens = ['test', 'document', 'stopword'] # Assuming 'this', 'is', 'a', 'with', 'some' are stopwords
        self.assertEqual(preprocess_document(text), expected_tokens)

    def test_create_dictionary_filtering(self):
        texts = [
            ['apple', 'banana', 'apple'],
            ['banana', 'orange'],
            ['apple', 'orange', 'grape'],
            ['kiwi', 'kiwi', 'kiwi'] # Appears in 1 doc, freq 3
        ]
        # no_below=2, no_above=0.5 (filter 'kiwi' and 'apple' if it appears in > 2 docs)
        dictionary = create_dictionary(texts)
        self.assertIn('banana', dictionary.token2id)
        self.assertIn('orange', dictionary.token2id)
        self.assertNotIn('kiwi', dictionary.token2id) # filtered by no_below=2
        # 'apple' appears in 2/4 docs = 0.5, so it should be kept if no_above > 0.5, or filtered if no_above <= 0.5
        # With default no_above=0.6, 'apple' should be kept.
        self.assertIn('apple', dictionary.token2id)

if __name__ == '__main__':
    unittest.main()
```