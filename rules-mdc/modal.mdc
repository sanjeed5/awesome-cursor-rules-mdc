---
description: This guide provides opinionated, actionable best practices for writing high-performance, cost-effective, and maintainable AI/ML applications on Modal.
globs: **/*
---
# modal Best Practices

Modal is the definitive platform for deploying AI/ML workloads. To leverage its full potential – sub-second cold starts, instant autoscaling, and GPU acceleration – you *must* adhere to these best practices. This guide cuts through the noise, providing the exact patterns your team will use daily.

## Code Organization and Structure

A well-structured Modal application is modular, explicit, and easy to debug.

### 1. Centralize Your `modal.Stub`

Always define a single, well-named `modal.Stub` at the top level of your main application file. This `Stub` is the entry point for all your Modal functions, images, and volumes.

❌ **BAD: Multiple `Stub` definitions or generic names**
```python
# my_module_a.py
import modal
stub_a = modal.Stub("my-app-part-a") # Don't do this

# my_module_b.py
import modal
stub_b = modal.Stub("my-app-part-b") # Or this
```

✅ **GOOD: Single, descriptive `Stub`**
```python
# src/my_ml_app/app.py
import modal

# Define the stub for your entire application
# Use a clear, unique name for your project/service
stub = modal.Stub("my-inference-service")

# All modal.Functions, Images, and Volumes will be attached to this stub
```

### 2. Modularize Your Application

For larger applications, separate your core logic (e.g., model loading, inference pipeline) into distinct Python modules. Import these modules into your main `app.py` where your `modal.Function`s are defined. This keeps your Modal definitions clean and your business logic testable.

```python
# src/my_ml_app/model_loader.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelLoader:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is None:
            print(f"Loading model {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
            print("Model loaded.")
        return self.model, self.tokenizer

# src/my_ml_app/app.py
import modal
from .model_loader import ModelLoader # Relative import for modularity

stub = modal.Stub("my-inference-service")

# Define your image and volumes here (see sections below)
inference_image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:23.09-py3") \
    .pip_install("torch", "transformers")

model_volume = modal.Volume.from_name("my-llm-weights", create_if_missing=True)

@stub.function(image=inference_image, volumes={"/models": model_volume})
def generate_text(prompt: str):
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # Lazy load the model inside the function
    model_loader = ModelLoader(model_id)
    model, tokenizer = model_loader.load()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@stub.local_entrypoint()
def main():
    print(generate_text.remote("Hello, my name is"))

```

## Common Patterns and Anti-patterns

These patterns ensure your Modal functions are robust, performant, and cost-efficient.

### 3. Explicit Resource Specification

Always declare the exact GPU type, CPU cores, and memory your function needs. This allows Modal's scheduler to provision precisely what's required, optimizing performance and cost. Never rely on defaults for production workloads.

❌ **BAD: Default (unspecified) resources**
```python
@stub.function() # No GPU, minimal CPU/memory
def process_data(data):
    # This will run on a CPU, potentially slowly
    pass
```

✅ **GOOD: Specific GPU, CPU, and Memory**
```python
@stub.function(
    gpu="A10G",          # Specify GPU type (e.g., "A10G", "A100", "L4")
    cpu=4,               # Allocate 4 CPU cores
    memory="16Gi",       # Allocate 16 GB of memory
    timeout=600          # Set a clear timeout for long-running tasks
)
def run_gpu_inference(input_data):
    # Your GPU-accelerated code here
    pass
```

### 4. Reproducible Environments with `modal.Image`

Build custom Docker images or pin existing ones to guarantee identical dependencies across runs. This is critical for reproducibility and avoiding "works on my machine" issues. Use `modal.Image` to define your environment once.

❌ **BAD: Installing dependencies inside the function or relying on `pip_install` directly on the function decorator**
```python
# This rebuilds the environment for every function or on every code change,
# leading to slow cold starts and non-deterministic environments.
@stub.function(image=modal.Image.debian_slim().pip_install("numpy", "pandas"))
def analyze_data(df):
    import numpy as np # This is bad, environment should be ready
    # ...
```

✅ **GOOD: Pre-build your `modal.Image` with all dependencies**
```python
# Define your base image and install all necessary packages
# Use a specific registry image for stability (e.g., PyTorch CUDA images)
inference_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:23.09-py3") # Pin a specific base image
    .pip_install(
        "torch==2.1.0",
        "transformers==4.35.2",
        "accelerate==0.24.1",
        "sentencepiece==0.1.99"
    )
    .apt_install("git", "ffmpeg") # Install system-level dependencies
)

@stub.function(image=inference_image, gpu="A10G")
def run_llm_inference(prompt: str):
    # All dependencies are pre-installed and ready
    import torch
    from transformers import pipeline
    # ...
```

### 5. Persistent Model Weights with `modal.Volume`

Store large model checkpoints in Modal's persistent `modal.Volume`s. Load them lazily inside your functions. This avoids repeated uploads, speeds up cold starts, and decouples model versions from code deployments.

❌ **BAD: Downloading models on every cold start or bundling in the image**
```python
# This downloads the model every time a new container spins up. Slow and wasteful.
@stub.function(image=inference_image, gpu="A10G")
def infer_with_model(input_data):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id) # Downloads here!
    # ...
```

✅ **GOOD: Lazy loading from a `modal.Volume`**
```python
import modal
from pathlib import Path

stub = modal.Stub("my-inference-service")

# 1. Define your persistent volume
# Create it once: `modal volume create my-llm-weights`
model_volume = modal.Volume.from_name("my-llm-weights", create_if_missing=True)

# 2. Define a function to download/update weights (run this once or on model updates)
# This function saves the model to the volume.
@stub.function(image=inference_image, volumes={"/models": model_volume}, gpu="A10G")
def download_model_to_volume(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    local_model_path = Path("/models") / model_id
    if not local_model_path.exists():
        print(f"Downloading {model_id} to volume...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)
        print("Download complete.")
    else:
        print(f"{model_id} already exists in volume.")

# 3. Your inference function loads lazily from the volume
@stub.function(image=inference_image, volumes={"/models": model_volume}, gpu="A10G")
def perform_inference(prompt: str, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    local_model_path = Path("/models") / model_id
    
    # Ensure model is in volume (could be pre-downloaded or downloaded on first run)
    if not local_model_path.exists():
        # This is a fallback; ideally, pre-download with download_model_to_volume
        print(f"Model {model_id} not found in volume, attempting to download...")
        download_model_to_volume.remote(model_id) # Call the download function
        
    print(f"Loading model {model_id} from volume...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    # ... perform inference
    return "Inference result"

@stub.local_entrypoint()
def main():
    # Run this once to populate the volume
    download_model_to_volume.remote("mistralai/Mistral-7B-Instruct-v0.2")
    print(perform_inference.remote("What is the capital of France?"))
```

### 6. Optimize Concurrency and Dynamic Batching

For high-throughput inference, configure `max_concurrent_inputs` and enable dynamic batching (beta) to let Modal automatically group requests. This significantly reduces per-request latency and increases GPU utilization.

❌ **BAD: Default concurrency for a busy API**
```python
@stub.function(gpu="A10G") # Default max_concurrent_inputs=1, no batching
def process_single_request(input_data):
    # Only one request processed at a time, GPU underutilized
    pass
```

✅ **GOOD: High concurrency with dynamic batching**
```python
@stub.function(
    image=inference_image,
    gpu="A10G",
    volumes={"/models": model_volume},
    # Allow up to 100 concurrent requests to this function
    max_concurrent_inputs=100,
    # Enable dynamic batching for better GPU utilization
    # Requests will be batched up to `batch_size` or for `batch_timeout` ms
    allow_concurrent_inputs=True, # Required for dynamic batching
    batch_size=32,
    batch_timeout=100, # milliseconds
    idle_timeout=300 # Shut down container after 5 min idle
)
def batched_inference(inputs: list[str]):
    # Your model should be designed to accept a list of inputs (a batch)
    # Load model once per container lifecycle
    # ... model loading logic (from volume) ...
    
    print(f"Processing batch of size {len(inputs)}")
    # Perform batched inference
    results = [f"Processed: {i}" for i in inputs]
    return results

@stub.local_entrypoint()
def main():
    # Example of calling a batched function
    # Modal will automatically batch these calls if `allow_concurrent_inputs` is True
    results = list(batched_inference.map(["prompt 1", "prompt 2", "prompt 3"]))
    print(results)
```

### 7. Secure Secrets Management

Never hardcode API keys, tokens, or sensitive configuration directly in your code or environment variables. Use `modal.Secret` to securely inject secrets into your functions.

❌ **BAD: Hardcoding secrets or using insecure environment variables**
```python
# In code:
OPENAI_API_KEY = "sk-..." # NEVER do this

# Or via insecure env vars (if not managed by Modal Secrets):
# os.environ["OPENAI_API_KEY"]
```

✅ **GOOD: Use `modal.Secret.from_name`**
```python
import modal
import os

stub = modal.Stub("my-secure-app")

# 1. Create your secret once via `modal secret create my-openai-secret`
#    and add key-value pairs like OPENAI_API_KEY=sk-...
openai_secret = modal.Secret.from_name("my-openai-secret")

@stub.function(secrets=[openai_secret])
def call_openai_api(prompt: str):
    import openai
    # Access the secret as an environment variable
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.Completion.create(engine="davinci", prompt=prompt)
    return response.choices[0].text

@stub.local_entrypoint()
def main():
    print(call_openai_api.remote("Tell me a joke."))
```

## Performance Considerations

### 8. Optimize Cold Starts with Snapshotting

Modal excels at cold starts, but you can further optimize by ensuring your image is lean and by using `modal.Image.run_commands` for heavy pre-computation if needed. For large models, lazy loading from `modal.Volume` is paramount (see #5).

```python
# Optimize image build time and size
optimized_image = (
    modal.Image.from_registry("ubuntu:22.04", force_build=True) # Start from a minimal base
    .apt_install("python3-pip", "git")
    .pip_install("fastapi", "uvicorn")
    # Run commands that pre-compile or pre-process things that don't change often
    # This gets baked into the image snapshot.
    .run_commands(["python -c 'import torch; print(torch.__version__)'"]) 
)

@stub.function(image=optimized_image)
def fast_startup_function():
    # ...
    pass
```

## Common Pitfalls and Gotchas

### 9. Don't Forget `stub.local_entrypoint()` or `stub.serve()`/`stub.deploy()`

Your Modal application won't run or deploy without a designated entry point. For local testing and development, use `local_entrypoint`. For deploying as a persistent service, use `serve` (for webhooks) or `deploy`.

❌ **BAD: Missing an entry point**
```python
# app.py
import modal
stub = modal.Stub("my-app")
@stub.function()
def my_func():
    print("Hello")
# This file does nothing when run directly or deployed without an entrypoint.
```

✅ **GOOD: Define a `local_entrypoint` for development**
```python
# app.py
import modal
stub = modal.Stub("my-app")

@stub.function()
def my_func():
    print("Hello from Modal!")

@stub.local_entrypoint() # This makes `python app.py` work locally and `modal run app.py` work remotely
def main():
    my_func.remote() # Call the remote function
```

✅ **GOOD: Define a `serve` entrypoint for web services**
```python
# app.py
import modal
stub = modal.Stub("my-api")

@stub.function()
@modal.web_endpoint(method="GET")
def hello():
    return {"message": "Hello, world!"}

# To deploy: `modal deploy app.py`
# To run locally with a dev server: `modal serve app.py`
```

### 10. Debugging Remote Functions

Debugging on Modal requires a different mindset. Leverage `print` statements, Modal's integrated logs, and `modal.lookup()` for inspecting deployed objects. Avoid complex interactive debugging directly on remote functions.

```python
# In your remote function
@stub.function(image=inference_image, gpu="A10G")
def debuggable_function(input_data):
    print(f"Received input: {input_data}") # Use print for logging
    try:
        result = some_complex_calculation(input_data)
        print(f"Calculation successful: {result}")
        return result
    except Exception as e:
        print(f"Error during calculation: {e}") # Log errors explicitly
        raise # Re-raise to see full traceback in Modal logs
```
After running, check `modal logs <app-id>` or the Modal dashboard for detailed output.

## Testing Approaches

### 11. Unit Test Your Core Logic Locally

Before deploying to Modal, thoroughly unit test the pure Python logic of your application. This includes model loading, data preprocessing, and post-processing. Modal functions should primarily orchestrate these well-tested components.

```python
# src/my_ml_app/model_loader.py (as defined in #2)
# ...

# tests/test_model_loader.py
import pytest
from src.my_ml_app.model_loader import ModelLoader

def test_model_loader_initialization():
    loader = ModelLoader("test/model")
    assert loader.model_id == "test/model"
    assert loader.model is None

# Mocking external calls for true unit tests
# def test_model_loading(mocker):
#     mocker.patch('transformers.AutoTokenizer.from_pretrained')
#     mocker.patch('transformers.AutoModelForCausalLM.from_pretrained')
#     loader = ModelLoader("mock/model")
#     model, tokenizer = loader.load()
#     assert model is not None
#     assert tokenizer is not None
```

### 12. Use `f.local()` for Quick Local Modal Function Testing

For quick checks of how your `modal.Function` interacts with your local environment and stub, use `f.local()`. This executes the function directly on your machine, bypassing Modal's cloud infrastructure.

```python
# app.py
import modal
stub = modal.Stub("my-app")

@stub.function()
def add_one(x: int) -> int:
    return x + 1

@stub.local_entrypoint()
def main():
    # Call the function locally, without Modal cloud overhead
    result = add_one.local(5)
    print(f"Local result: {result}") # Expected: 6
```

### 13. Integration Test Deployed Endpoints

For end-to-end validation, deploy your application and write integration tests that hit the live Modal endpoints (webhooks or deployed functions). This verifies the entire stack, including environment setup, resource allocation, and data flow.

```python
# After deploying your app (e.g., `modal deploy app.py`)
# You can get the URL for webhooks or use `modal.lookup()` for functions.

# Example: Testing a webhook
import requests

app_url = "https://<your-app-name>-<user-id>.modal.run" # Get this from `modal deploy` output

def test_hello_webhook():
    response = requests.get(f"{app_url}/hello")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, world!"}

# Example: Testing a remote function via lookup
# import modal
# stub = modal.Stub.from_name("my-inference-service", namespace=modal.get_env())
# remote_generate_text = stub.lookup("generate_text")
# result = remote_generate_text.remote("Test prompt")
# assert "Test" in result
```