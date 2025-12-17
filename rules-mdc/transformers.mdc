---
description: Definitive guidelines for writing high-quality, maintainable, and performant code with 🤗 Transformers, ensuring consistency and adherence to 2025 best practices.
globs: **/*.py
---
# transformers Best Practices

This guide outlines the definitive best practices for developing with 🤗 Transformers, focusing on reliability, readability, and production readiness. Adhere to these rules to ensure your code integrates seamlessly with the ecosystem and passes all CI checks.

## 1. Code Organization and Structure

Always follow the "Modular Transformers" framework for new models. This reduces boilerplate and promotes reusability.

### 1.1. Modular Model Definitions
Place new model definitions in `src/transformers/models/<model_name>/` using a modular file (e.g., `modular_<model_name>.py`). Inherit from existing base classes and import components from other models to minimize code duplication.

❌ BAD: Re-implementing common layers or full models from scratch.
```python
# src/transformers/models/my_model/modeling.py
class MyModelAttention(nn.Module):
    # ... duplicate attention logic ...

class MyModel(PreTrainedModel):
    # ... full model re-implementation ...
```

✅ GOOD: Inheriting and reusing components.
```python
# src/transformers/models/my_model/modular_my_model.py
from ..llama.modeling_llama import LlamaAttention, LlamaModel
from ..llama.configuration_llama import LlamaConfig

class MyModelConfig(LlamaConfig):
    model_type = "my_model"
    # Add/override specific config parameters

class MyModelAttention(LlamaAttention):
    # Only override specific methods if needed, otherwise inherit directly
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Custom logic if different from LlamaAttention
        return super().forward(hidden_states, attention_mask, **kwargs)

class MyModel(LlamaModel):
    # Override the attention layer if MyModelAttention is different
    def __init__(self, config):
        super().__init__(config)
        self.attention = MyModelAttention(config) # Use custom attention
```
After creating the modular file, generate the single-file structure:
```bash
python utils/modular_model_converter.py my_model
```

## 2. Python Style and Formatting

Strictly adhere to PEP 8 and the Google Python Style Guide. Automated tools enforce this.

### 2.1. Automated Formatting
Always run `make style` and `make quality` locally before committing. For changes in your current branch, use `make fixup`.

❌ BAD: Manual formatting, inconsistent spacing, un-sorted imports.
```python
import os, sys
from transformers import AutoModel
def my_func(arg1,arg2):
    if arg1 == True:
        return arg2
```

✅ GOOD: Consistent, auto-formatted code.
```python
import os
import sys

from transformers import AutoModel


def my_func(arg1: bool, arg2: str) -> str:
    if arg1:  # Use direct boolean evaluation
        return arg2
    return ""
```

### 2.2. Line Length
Limit all lines to a maximum of 79 characters. Docstrings and comments should be 72 characters. Use Python's implicit line continuation.

❌ BAD: Long, unreadable lines.
```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10, output_attentions=True, output_hidden_states=True, return_dict=True)
```

✅ GOOD: Wrapped lines using parentheses.
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=10,
    output_attentions=True,
    output_hidden_states=True,
    return_dict=True,
)
```

## 3. Type Hints and Docstrings

Type hints are mandatory for all public methods. Docstrings must follow the Google style.

### 3.1. Type Annotations
Use `from __future__ import annotations` for forward references. Annotate all function arguments, return values, and class attributes.

❌ BAD: Missing type hints, unclear intent.
```python
def process_data(data, config):
    """Processes input data."""
    # ...
    return processed_data
```

✅ GOOD: Clear, explicit type hints.
```python
from __future__ import annotations
from typing import Any, Dict

def process_data(data: list[str], config: Dict[str, Any]) -> list[int]:
    """Processes input data according to the provided configuration.

    Args:
        data: A list of string inputs to be processed.
        config: A dictionary containing processing parameters.

    Returns:
        A list of integers representing the processed data.
    """
    # ...
    return [len(d) for d in data]
```

### 3.2. Docstring Conventions
Follow the Google Python Style Guide for docstrings. Include a concise summary, Args, Returns, and Raises sections where applicable.

## 4. Performance Considerations

Prioritize efficiency and leverage built-in optimizations.

### 4.1. Optimized Training with `Trainer`
For training, always use `transformers.Trainer`. It provides out-of-the-box support for:
*   Mixed precision training (`fp16=True` or `bf16=True`).
*   `torch.compile` for graph compilation (`torch_compile=True`).
*   FlashAttention (if available for your model/hardware).

❌ BAD: Writing custom training loops that re-implement `Trainer` features.
```python
# Manual training loop
optimizer = AdamW(model.parameters(), lr=1e-5)
for epoch in range(num_epochs):
    # ... manual gradient accumulation, mixed precision, etc. ...
```

✅ GOOD: Leveraging `Trainer` for robust and optimized training.
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    fp16=True,  # Enable mixed precision
    torch_compile=True, # Enable torch.compile
    # ... other args ...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # ...
)
trainer.train()
```

### 4.2. Efficient Inference with `Pipeline`
For common inference tasks, use `transformers.pipeline`. It handles tokenization, model inference, and post-processing efficiently.

❌ BAD: Manually managing tokenizers, models, and post-processing.
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
# ... manual label mapping ...
```

✅ GOOD: Using `pipeline` for streamlined inference.
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this movie!")
# result: [{'label': 'POSITIVE', 'score': 0.99987}]
```

## 5. Testing Approaches

Thorough testing is non-negotiable. Every new feature or model requires unit tests.

### 5.1. Unit Tests
Write unit tests for every new model, utility, and significant code change. Verify:
*   **Forward-pass shapes**: Ensure model outputs have expected dimensions.
*   **Serialization**: Test saving and loading with `safetensors`.
*   **Edge cases**: Test with various input sizes, empty inputs, etc.

❌ BAD: No tests, or only manual verification.
```python
# No test file for new_model.py
# Developer manually runs `new_model.py` to check if it works.
```

✅ GOOD: Dedicated test files (`test_new_model.py`) verifying core functionality.
```python
# tests/test_new_model.py
import unittest
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.new_model import NewModel, NewModelConfig

class NewModelTester(unittest.TestCase):
    def setUp(self):
        self.config = NewModelConfig(vocab_size=100, hidden_size=16)
        self.model = NewModel(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # or a dummy tokenizer

    def test_forward_pass(self):
        input_ids = torch.randint(0, self.config.vocab_size, (2, 10))
        outputs = self.model(input_ids)
        self.assertIn("last_hidden_state", outputs)
        self.assertEqual(outputs.last_hidden_state.shape, (2, 10, self.config.hidden_size))

    def test_safetensors_serialization(self):
        # Save and load with safetensors
        tmpdir = self.create_temporary_dir()
        self.model.save_pretrained(tmpdir, safe_serialization=True)
        loaded_model = NewModel.from_pretrained(tmpdir)
        self.assertTrue(torch.equal(self.model.state_dict()["embeddings.word_embeddings.weight"],
                                    loaded_model.state_dict()["embeddings.word_embeddings.weight"]))
```

### 5.2. Local Test Execution
Use `utils/tests_fetcher.py` to identify relevant tests and run them with `pytest`.
```bash
python utils/tests_fetcher.py
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

## 6. Packaging and Deployment

Leverage `safetensors` and MLflow for robust model packaging.

### 6.1. `safetensors` for Model Serialization
Always use `safetensors` for saving and loading model weights. It's faster and more secure than traditional PyTorch checkpoints.

❌ BAD: Using `torch.save` or `model.save_pretrained(safe_serialization=False)`.
```python
torch.save(model.state_dict(), "model.pt")
```

✅ GOOD: Using `model.save_pretrained` with `safe_serialization=True`.
```python
model.save_pretrained("./my_model_dir", safe_serialization=True)
# To load:
loaded_model = AutoModel.from_pretrained("./my_model_dir")
```

### 6.2. MLflow Integration
For production deployments, integrate with `mlflow.transformers` to log models, configurations, and prompt templates. This ensures reproducible and traceable deployments.

```python
import mlflow
from transformers import pipeline

# Log a pipeline
with mlflow.start_run():
    text_generator = pipeline("text-generation", model="gpt2")
    mlflow.transformers.log_model(
        transformers_model=text_generator,
        artifact_path="text_generator_pipeline",
        model_config={"max_new_tokens": 50, "do_sample": True},
        input_example="Hello, my name is",
    )

# Load for inference
logged_model = mlflow.pyfunc.load_model("runs:/<run_id>/text_generator_pipeline")
result = logged_model.predict(["Hello, my name is"])
```

## 7. Virtual Environments

Always work within a dedicated virtual environment.

❌ BAD: Installing dependencies globally.
```bash
pip install transformers[dev] # Installs into system Python
```

✅ GOOD: Using `venv` or `conda` for isolated environments.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" # Editable install with dev dependencies
```