---
description: Definitive guidelines for writing high-performance, maintainable, and production-ready LLM inference code using vLLM.
globs: **/*.py
---
# vLLM Best Practices

vLLM is the gold standard for high-throughput LLM inference. Adhere to these guidelines to maximize performance, ensure reproducibility, and maintain robust LLM services.

## 1. Environment & Installation

**Always use isolated `conda` environments and pin `vLLM` to your CUDA version.** This prevents binary incompatibilities and ensures reproducible deployments.

❌ BAD:
```python
# Unreliable global install, prone to CUDA/PyTorch mismatches
pip install vllm
```

✅ GOOD:
```bash
# For NVIDIA GPUs (CUDA 12.1 is default, check vLLM docs for current default)
conda create -n vllm_env python=3.10 -y
conda activate vllm_env
pip install vllm==0.5.2 # Pin the exact version in requirements.txt

# For NVIDIA GPUs (CUDA 11.8 specific version, e.g., v0.4.0)
conda create -n vllm_cu118 python=3.10 -y
conda activate vllm_cu118
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```
*   **Action**: Ensure `requirements.txt` explicitly lists `vllm==X.Y.Z`.
*   **Hardware**: Target GPUs with compute capability ≥ 7.0 (V100, A100, H100, etc.).

## 2. Code Organization & Structure

**Separate inference logic from data preprocessing and business logic.** Use a modular approach for clarity and testability.

❌ BAD:
```python
# main.py - monolithic script, hard to test or scale
from vllm import LLM, SamplingParams
# ... data loading, preprocessing, model init, inference, postprocessing ...
```

✅ GOOD:
```python
# llm_service/inference_engine.py
from vllm import LLM, SamplingParams
from typing import List

class InferenceEngine:
    def __init__(self, model_path: str, **kwargs):
        """Initializes the vLLM engine with specified model and configurations."""
        self.llm = LLM(model=model_path, **kwargs)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)

    def generate(self, prompts: List[str]) -> List[str]:
        """Generates responses for a list of prompts using configured sampling parameters."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

# llm_service/main.py (or API endpoint, e.g., with FastAPI)
from .inference_engine import InferenceEngine
from typing import Dict

def serve_llm_request(request_data: Dict) -> Dict:
    """Handles an incoming LLM request, orchestrating preprocessing, inference, and postprocessing."""
    # 1. Preprocessing (e.g., validate input, format prompt from request_data)
    prompts = [request_data["text"]] # Simplified example
    
    # 2. Inference
    # Model path and parallelism should be loaded from config, not hardcoded here
    engine = InferenceEngine(model_path="mistralai/Mistral-7B-Instruct-v0.2",
                             tensor_parallel_size=2) # Explicitly configure parallelism
    results = engine.generate(prompts)
    
    # 3. Postprocessing (e.g., format output for API response, add metadata)
    return {"generated_text": results[0]}
```
*   **Action**: Define `LLM` engine parameters explicitly (e.g., `tensor_parallel_size` for distributed inference).

## 3. Performance & Scalability

**Leverage vLLM's core optimizations: continuous batching, parallel sampling, and quantization.**

### 3.1 Continuous Batching & Parallel Sampling
These are enabled by default in `vLLM`'s `LLM` class and `SamplingParams`. Ensure your client sends multiple requests or uses `n` for parallel sampling.

✅ GOOD:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Use n > 1 for parallel sampling (generates multiple outputs per prompt)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    n=4, # Generate 4 independent samples per prompt for diversity/evaluation
    max_tokens=128
)

prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)
# outputs will contain multiple results for each prompt due to n=4
```

### 3.2 Quantization
**Always use quantization for production deployments** unless specific precision is critical. FP8 KV-cache is highly recommended for memory efficiency.

❌ BAD:
```python
# No quantization, higher memory/compute footprint, less throughput
llm = LLM(model="meta-llama/Llama-2-7b-hf")
```

✅ GOOD:
```python
# Use FP8 KV-cache for significant memory efficiency and throughput gains
llm = LLM(model="meta-llama/Llama-2-7b-hf", kv_cache_dtype="fp8")

# Or load a pre-quantized model (e.g., AWQ) for further optimization
llm_awq = LLM(model="casperhansen/llama-2-7b-chat-hf-awq", quantization="awq")
```

## 4. Observability

**Instrument your vLLM services with OpenLIT for OpenTelemetry.** This is non-negotiable for production-grade LLM applications, providing crucial traces, metrics, and cost tracking.

✅ GOOD:
```python
# Install: pip install openlit
# Run your app with OpenLIT CLI for zero-code instrumentation (recommended for quick setup):
# openlit-instrument --service-name my-vllm-app --environment production --otlp-endpoint http://localhost:4318 python your_vllm_app.py

# Or initialize OpenLIT directly in your Python code for more programmatic control:
from openlit import openlit
from vllm import LLM

# Initialize OpenLIT *before* any vLLM operations to ensure full instrumentation
openlit.init(
    service_name="my-vllm-service",
    environment="production",
    otlp_endpoint="http://localhost:4318" # Your OpenTelemetry Collector endpoint
)

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
# ... subsequent vLLM operations will now be automatically traced and monitored
```
*   **Action**: Ensure OpenLIT is initialized *before* `vllm.LLM` instantiation.

## 5. Deployment & MLOps

**Deploy vLLM using containerization (Docker) and orchestration (Kubernetes).** Automate CI/CD and track experiments for robust LLMOps.

*   **Containerization**: Build Docker images with your specific `vLLM` installation and model weights. This ensures consistent, isolated, and reproducible environments across development and production.
*   **Orchestration**: Use Kubernetes with GPU device plugins (e.g., AMD K8s device plugin for ROCm, NVIDIA device plugin for CUDA) for dynamic scaling, autoscaling, and high availability. Leverage Helm charts for declarative deployments.
*   **Version Control**: Version all models, configurations, and inference code. Implement experiment tracking tools (e.g., MLflow, DVC) for model artifacts and metrics.
*   **Testing**: Implement rigorous testing for model serving endpoints (latency, throughput, correctness, data sanity checks, and A/B testing in production).

## 6. Type Hints

**Use type hints extensively.** This improves code readability, maintainability, and enables static analysis, crucial for complex ML systems and team collaboration.

❌ BAD:
```python
def process_output(output):
    # What is 'output'? What does it contain? This is ambiguous.
    return output.outputs[0].text
```

✅ GOOD:
```python
from vllm.outputs import RequestOutput

def process_output(output: RequestOutput) -> str:
    """Extracts the generated text from a vLLM RequestOutput object, handling empty outputs."""
    if not output.outputs:
        return "" # Handle cases with no generated text gracefully
    return output.outputs[0].text
```

## 7. Cursor Interaction (Prompt Engineering)

When interacting with Cursor for `vLLM` code generation or refactoring, apply prompt engineering best practices for optimal results:

*   **Role**: "Act as a senior ML engineer specializing in vLLM inference and MLOps."
*   **Context**: Provide relevant code snippets, `vLLM` version, GPU type, and the specific problem you're solving.
*   **Task**: Be explicit: "Generate a `vLLM` inference script that loads `Llama-3-8B-Instruct` with FP8 KV-cache and serves it via an OpenAI-compatible API endpoint using FastAPI."
*   **Format**: Specify desired output (e.g., "Provide a complete Python script," "Show only the `LLM` initialization block," "Include `requirements.txt`").
*   **Constraints**: "Ensure `tensor_parallel_size` is set to 4," "Do not include any `transformers` code unless explicitly requested," "Prioritize `AsyncLLMEngine` for non-blocking operations."