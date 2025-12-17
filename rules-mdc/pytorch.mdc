---
description: Definitive guidelines for writing clean, performant, and maintainable PyTorch code, emphasizing modern best practices, explicit device management, and efficient training patterns.
globs: **/*.py
---
# PyTorch Best Practices

This guide outlines the definitive best practices for developing with PyTorch, ensuring your code is readable, performant, and production-ready. We prioritize usability, explicit control, and modern tooling.

## 1. Code Organization and Structure

Structure your PyTorch projects for clarity, testability, and scalability. Encapsulate logical blocks into distinct functions or classes.

### 1.1. Modularize Your Codebase

Separate data loading, model definition, training, and evaluation into dedicated modules or functions. This makes components reusable and testable.

❌ BAD: Monolithic script
```python
# train.py
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ... data loading, model definition, training loop all in one file ...

class MyModel(nn.Module):
    # ...
    pass

def main():
    # Data loading
    train_data = TensorDataset(...)
    train_loader = DataLoader(train_data, batch_size=32)

    # Model, optimizer, loss
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            # ... training logic ...
            pass

if __name__ == "__main__":
    main()
```

✅ GOOD: Modularized structure
```python
# src/data.py
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    # Example: Create synthetic data
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X[:100], y[:100]) # Smaller val set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# src/model.py
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), # Assuming 28x28 input, adjust for other sizes
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# src/train.py
import torch
import torch.nn as nn
from torch.optim import Adam
from src.model import SimpleCNN
from src.data import get_dataloaders

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: Adam, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# main.py
import torch
from src.model import SimpleCNN
from src.data import get_dataloaders
from src.train import train_epoch, evaluate_model

def run_experiment(epochs: int = 10, batch_size: int = 64, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(batch_size)
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {val_accuracy:.2f}%")

    torch.save(model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    run_experiment()
```

## 2. Common Patterns and Anti-patterns

Adopt patterns that enhance clarity and performance, and avoid common pitfalls.

### 2.1. Explicit Device Placement

Always explicitly move tensors and models to the correct device (`cpu` or `cuda`). Never rely on implicit device handling.

❌ BAD: Implicit device assumption
```python
model = MyModel() # Model is on CPU by default
data = torch.randn(1, 3, 224, 224) # Data is on CPU
output = model(data) # This will run on CPU, even if CUDA is available
```

✅ GOOD: Explicit device placement
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = torch.randn(1, 3, 224, 224).to(device)
output = model(data) # Model and data are on the same device
```

### 2.2. Disable Gradient Calculation for Inference

Use `torch.no_grad()` for validation and inference to save memory and speed up computation.

❌ BAD: Calculating unnecessary gradients
```python
model.eval()
for data, target in val_loader:
    output = model(data) # Gradients are still computed and stored
    loss = criterion(output, target)
    # ...
```

✅ GOOD: Disabling gradients
```python
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        output = model(data) # No gradients computed
        loss = criterion(output, target)
        # ...
```

### 2.3. Avoid In-Place Operations (Unless Profiled)

In-place operations (`.add_()`, `.mul_()`, etc.) can break PyTorch's autograd engine and make debugging difficult. Prefer out-of-place operations unless profiling explicitly shows an in-place operation is a critical bottleneck for memory or speed.

❌ BAD: In-place operation
```python
x = torch.randn(5, requires_grad=True)
y = x * 2
x.add_(1) # Modifies x in-place, can cause autograd issues
```

✅ GOOD: Out-of-place operation
```python
x = torch.randn(5, requires_grad=True)
y = x * 2
x = x + 1 # Creates a new tensor, preserving autograd graph
```

## 3. Performance Considerations

Optimize your PyTorch code for speed and memory efficiency.

### 3.1. Leverage `torch.compile`

Use `torch.compile` to significantly accelerate your models by tracing and optimizing the computation graph. Apply it to your `nn.Module` instances.

❌ BAD: Running model in eager mode only
```python
model = MyModel().to(device)
# Training loop
for data, target in loader:
    output = model(data)
    # ...
```

✅ GOOD: Compiling the model
```python
model = MyModel().to(device)
compiled_model = torch.compile(model) # Compile once
# Training loop
for data, target in loader:
    output = compiled_model(data) # Use the compiled model
    # ...
```

### 3.2. Mixed Precision Training with `torch.cuda.amp`

For CUDA devices, use automatic mixed precision (AMP) to halve memory bandwidth and speed up computation without significant accuracy loss.

❌ BAD: Full precision training
```python
optimizer = torch.optim.Adam(model.parameters())
for data, target in loader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

✅ GOOD: Mixed precision training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters())

for data, target in loader:
    with autocast(): # Operations within this context run in mixed precision
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward() # Scale gradients
    scaler.step(optimizer) # Update optimizer
    scaler.update() # Update scaler for next iteration
```

### 3.3. Optimize `DataLoader`

Configure `DataLoader` for efficient asynchronous data loading.

❌ BAD: Default `DataLoader` settings
```python
DataLoader(dataset, batch_size=32) # num_workers=0, pin_memory=False
```

✅ GOOD: Optimized `DataLoader`
```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=min(os.cpu_count(), 8), # Tune based on CPU cores
    pin_memory=True, # Faster host-to-GPU transfer
    persistent_workers=True # Keep workers alive between epochs
)
```

## 4. Common Pitfalls and Gotchas

Be aware of these common issues to avoid frustrating debugging sessions.

### 4.1. Device Mismatches

Ensure all tensors involved in an operation are on the same device. This is the most frequent error.

❌ BAD: Device mismatch
```python
model = MyModel().cuda()
input_tensor = torch.randn(1, 3, 224, 224).cpu() # Input on CPU
output = model(input_tensor) # ERROR: Expected all tensors to be on the same device
```

✅ GOOD: Consistent device placement
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
input_tensor = torch.randn(1, 3, 224, 224).to(device)
output = model(input_tensor) # All good
```

### 4.2. Multiprocessing Start Method

When using multiprocessing with CUDA, always set the start method to `"spawn"` or `"forkserver"` to avoid "poison fork" issues.

❌ BAD: Default `fork` start method with CUDA
```python
# This might implicitly use 'fork' on Linux, leading to CUDA errors
import torch.multiprocessing as mp
mp.set_start_method('fork') # Explicitly setting bad practice
```

✅ GOOD: Safe multiprocessing start method
```python
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True) # Recommended for cross-platform safety
except RuntimeError:
    pass # Already set
```

## 5. Type Hints

Use type hints extensively for improved readability, maintainability, and static analysis.

❌ BAD: Untyped functions
```python
def train_step(model, data, target, optimizer, criterion):
    # ...
    pass
```

✅ GOOD: Type-hinted functions
```python
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer: Optimizer,
    criterion: nn.Module
) -> float:
    # ...
    return loss.item()
```

## 6. Virtual Environments

Always use virtual environments (`venv` or `conda`) to manage dependencies and ensure reproducible setups.

❌ BAD: Global package installation
```bash
pip install torch torchvision # Pollutes global Python environment
```

✅ GOOD: Virtual environment setup
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision

# Using conda
conda create -n my_env python=3.10
conda activate my_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 7. Packaging

For reusable components or libraries, package your code properly using `setuptools` or `poetry`.

❌ BAD: Just a collection of scripts
```
my_project/
├── train.py
├── model.py
└── data.py
```

✅ GOOD: Packaged structure
```
my_project/
├── pyproject.toml # Or setup.py
├── src/
│   └── my_project_lib/
│       ├── __init__.py
│       ├── model.py
│       ├── data.py
│       └── train_utils.py
├── scripts/
│   └── run_training.py # Entry point that imports from src/
└── tests/
    ├── test_model.py
    └── test_data.py
```

## 8. Testing Approaches

Implement unit tests for individual modules and integration tests for training loops.

### 8.1. Unit Tests for Modules

Test `nn.Module` definitions, data transformations, and utility functions independently.

❌ BAD: No tests, or only end-to-end testing
```python
# No dedicated tests for SimpleCNN
```

✅ GOOD: Unit test for a model
```python
# tests/test_model.py
import unittest
import torch
from src.model import SimpleCNN

class TestSimpleCNN(unittest.TestCase):
    def test_forward_pass(self):
        model = SimpleCNN(num_classes=10)
        input_tensor = torch.randn(1, 1, 28, 28) # Batch, Channels, H, W
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10))

    def test_output_range(self):
        model = SimpleCNN(num_classes=2)
        input_tensor = torch.randn(1, 1, 28, 28)
        output = model(input_tensor)
        # Check if output is float
        self.assertTrue(output.dtype == torch.float32)

if __name__ == '__main__':
    unittest.main()
```

### 8.2. Integration Tests for Training Loops

Use small, mock datasets to quickly verify the training loop's functionality without long training times.

❌ BAD: Only testing with full datasets
```python
# Training script only runs with full data, slow to verify changes
```

✅ GOOD: Integration test with mock data
```python
# tests/test_training.py
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model import SimpleCNN
from src.train import train_epoch, evaluate_model

class TestTrainingLoop(unittest.TestCase):
    def test_training_run(self):
        device = torch.device("cpu") # Use CPU for faster testing
        model = SimpleCNN(num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Create a small mock dataset
        mock_X = torch.randn(10, 1, 28, 28)
        mock_y = torch.randint(0, 2, (10,))
        mock_dataset = TensorDataset(mock_X, mock_y)
        mock_loader = DataLoader(mock_dataset, batch_size=2)

        initial_loss = train_epoch(model, mock_loader, optimizer, criterion, device)
        # Run a second epoch to ensure loss decreases (basic sanity check)
        second_epoch_loss = train_epoch(model, mock_loader, optimizer, criterion, device)

        self.assertLess(second_epoch_loss, initial_loss * 1.1) # Expect some decrease or stability
        accuracy = evaluate_model(model, mock_loader, device)
        self.assertGreaterEqual(accuracy, 0.0) # Basic check that accuracy is calculated

if __name__ == '__main__':
    unittest.main()
```