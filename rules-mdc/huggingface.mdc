---
description: Definitive guidelines for writing robust, performant, and maintainable Python code using Hugging Face Transformers and the Hugging Face Hub.
globs: **/*.py
---
# Hugging Face Best Practices

This guide establishes the definitive coding standards for developing with Hugging Face libraries, particularly `transformers`, and interacting with the Hugging Face Hub. Adherence ensures high-quality, performant, and maintainable ML code.

## 1. Code Organization and Structure

Always structure your Hugging Face projects for clarity, modularity, and Hub compatibility.

### 1.1 Model and Tokenizer Loading
Use `Auto` classes and `from_pretrained` for standard model and tokenizer loading. This ensures compatibility and leverages the Hub's versioning.

❌ **BAD** (Hardcoding specific model classes, limits flexibility)
```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

✅ **GOOD** (Flexible, auto-detecting model types and versions)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### 1.2 Project Structure for Hub Repositories
Embrace the "Git-first" workflow. Every model, dataset, or Space lives in a Git repository on the Hub. Structure your local project to mirror this, ensuring `README.md` (Model Card) and `.gitattributes` are present.

✅ **GOOD** (Standard Hub repository structure)
```
my-awesome-model/
├── src/
│   ├── model.py            # Custom model definition (if any)
│   └── utils.py            # Helper functions
├── scripts/
│   └── train.py            # Training script
├── data/                   # Small datasets, or pointers to Hub datasets
├── config.json             # Model configuration
├── tokenizer.json          # Tokenizer configuration
├── model.safetensors       # Model weights (or .bin)
├── README.md               # Crucial Model Card with metadata
├── .gitattributes          # For large file handling (Xet)
└── requirements.txt        # Project dependencies
```

## 2. Common Patterns and Anti-patterns

### 2.1 Inference with `pipeline`
For quick, high-level inference tasks, `pipeline` is the most efficient and robust choice. It handles preprocessing and postprocessing automatically.

❌ **BAD** (Manual steps for common tasks, error-prone)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("I love Hugging Face!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(model.config.id2label[predictions.item()])
```

✅ **GOOD** (Leverage `pipeline` for simplicity and robustness)
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love Hugging Face!")
print(result)
```

### 2.2 Saving and Loading Models
Always use `save_pretrained()` and `from_pretrained()` for models and tokenizers. This ensures all necessary files (config, weights, tokenizer files) are handled correctly for Hub compatibility.

❌ **BAD** (Framework-specific saving, loses crucial metadata)
```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
torch.save(model.state_dict(), "my_model_weights.pt")
# To load, you'd still need the config and potentially other files.
```

✅ **GOOD** (Hugging Face standard, Hub-ready)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save to a local directory
save_directory = "./my_fine_tuned_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Load later
loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
```

## 3. Performance Considerations

Prioritize performance from the start, especially for training and large model inference.

### 3.1 Hardware-Aware Libraries
Leverage `Optimum`, `PEFT`, `bitsandbytes`, and `accelerate` for memory and speed optimizations.

✅ **GOOD** (Using `accelerate` for distributed training/inference)
```python
# In train.py
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

class MyDataset(Dataset):
    def __len__(self): return 100
    def __getitem__(self, idx): return {"input_ids": torch.randint(0, 1000, (128,)), "labels": torch.randint(0, 2, (1,))}

accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=1e-5)
train_dataloader = DataLoader(MyDataset(), batch_size=8)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# Training loop
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

✅ **GOOD** (Using `bitsandbytes` for 4-bit quantization with `PEFT` for LoRA)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Apply PEFT LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 3.2 Mixed Precision Training
Always enable mixed precision (e.g., `bfloat16` or `float16`) for GPU training to reduce memory footprint and speed up computation.

✅ **GOOD** (With `Trainer`)
```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    fp16=True, # Enable mixed precision (or bf16=True for bfloat16)
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)
trainer.train()
```

## 4. Common Pitfalls and Gotchas

### 4.1 Device Placement
Ensure your model and inputs are on the same device (CPU/GPU) to avoid runtime errors. Use `model.to(device)` and `inputs.to(device)`.

❌ **BAD** (Mixing devices, leads to runtime errors)
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").cuda() # Model on GPU
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello", return_tensors="pt") # Inputs on CPU by default
outputs = model(**inputs) # ERROR: Inputs on CPU, model on GPU
```

✅ **GOOD** (Consistent device placement)
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello", return_tensors="pt").to(device) # Move inputs to device
outputs = model(**inputs)
```

### 4.2 Tokenizer Mismatch
Always use the tokenizer associated with the specific pretrained model you are using. Different models often have different tokenization schemes.

❌ **BAD** (Using a generic tokenizer, leads to incorrect results)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Using a tokenizer not designed for Llama-2
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from