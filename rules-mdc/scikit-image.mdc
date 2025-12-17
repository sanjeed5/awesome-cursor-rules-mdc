---
description: This guide defines the definitive best practices for developing with scikit-image, ensuring maintainable, performant, and robust image processing code.
globs: **/*.py
---
# scikit-image Best Practices

This document outlines the mandatory guidelines for all `scikit-image` development within our team. Adhering to these rules ensures consistency, performance, and long-term maintainability of our image processing pipelines.

## 1. Core Principles

`scikit-image` is built on NumPy. All functions must accept and return plain `ndarray` objects.

### 1.1. Immutability of Input Images
**Always** treat input images as immutable. Functions must return new arrays, never modify inputs in-place. This prevents unexpected side effects and simplifies debugging.

❌ BAD: In-place modification
```python
import numpy as np
from skimage import exposure

def normalize_image_bad(image: np.ndarray) -> None:
    """❌ BAD: Modifies the input image directly."""
    image[:] = exposure.rescale_intensity(image, out_range=(0, 1))

img = np.array([[0, 100], [50, 200]], dtype=np.uint8)
original_img_id = id(img)
normalize_image_bad(img)
print(f"Image ID changed? {id(img) != original_img_id}") # False, same object modified
```

✅ GOOD: Return a new array
```python
import numpy as np
from skimage import exposure
from numpy.typing import NDArray, Any

def normalize_image_good(image: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """✅ GOOD: Returns a new, processed image array."""
    return exposure.rescale_intensity(image, out_range=(0, 1))

img = np.array([[0, 100], [50, 200]], dtype=np.uint8)
original_img_id = id(img)
processed_img = normalize_image_good(img)
print(f"Image ID changed? {id(processed_img) != original_img_id}") # True, new object returned
```

### 1.2. Public API Usage
**Always** use the documented public API. Avoid internal or private functions (prefixed with `_`) to ensure forward compatibility and stability.

## 2. Code Organization and Structure

### 2.1. Modular Functions
Break down complex image processing tasks into small, focused, and reusable functions. Each function should do one thing well.

❌ BAD: Monolithic function
```python
import numpy as np
from skimage import io, filters, exposure

def process_image_bad(path: str) -> np.ndarray:
    """❌ BAD: Combines loading, filtering, and normalization."""
    image = io.imread(path, as_gray=True)
    blurred = filters.gaussian(image, sigma=1)
    normalized = exposure.rescale_intensity(blurred, out_range=(0, 1))
    return normalized
```

✅ GOOD: Chained, focused functions
```python
import numpy as np
from skimage import io, filters, exposure
from numpy.typing import NDArray, Any

def load_grayscale_image(path: str) -> NDArray[Any, Any]:
    """Loads an image from path as grayscale."""
    return io.imread(path, as_gray=True)

def apply_gaussian_blur(image: NDArray[Any, Any], sigma: float) -> NDArray[Any, Any]:
    """Applies a Gaussian blur to the image."""
    return filters.gaussian(image, sigma=sigma)

def normalize_intensity(image: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Rescales image intensity to [0, 1]."""
    return exposure.rescale_intensity(image, out_range=(0, 1))

# Usage:
# image_path = "path/to/your/image.png"
# img = load_grayscale_image(image_path)
# blurred_img = apply_gaussian_blur(img, sigma=1.5)
# final_img = normalize_intensity(blurred_img)
```

### 2.2. Docstrings and Type Hints
**Every** function, class, and method must have a comprehensive docstring following PEP 257 and explicit type hints following PEP 484. Use `numpy.typing.NDArray` for NumPy arrays.

```python
from typing import Tuple, Literal
import numpy as np
from numpy.typing import NDArray, DTypeLike

def convert_to_float(
    image: NDArray[DTypeLike, Any],
    *,
    force_copy: bool = False
) -> NDArray[np.float64, Any]:
    """Converts an image to floating-point representation.

    Parameters
    ----------
    image : NDArray
        The input image array.
    force_copy : bool, optional
        If True, always return a new array, even if the dtype is already float.
        Defaults to False.

    Returns
    -------
    NDArray[np.float64, Any]
        The image converted to float64, with pixel values scaled to [0, 1]
        if the original dtype was integer.
    """
    # Implementation details...
    return image.astype(np.float64) # Simplified for example
```

## 3. Common Patterns and Anti-patterns

### 3.1. Image Data Types and Ranges
**Always** process images in floating-point format (e.g., `np.float64`) with values normalized to `[0, 1]`. Convert to `uint8` or `uint16` only for display or saving. Use `skimage.util.img_as_float` for reliable conversion.

❌ BAD: Processing with `uint8`
```python
import numpy as np
from skimage import filters

def process_uint8_bad(image: np.ndarray) -> np.ndarray:
    """❌ BAD: Operations on uint8 can lead to clipping and precision loss."""
    # e.g., subtracting 50 from a pixel value of 20 results in 0, not -30
    return filters.gaussian(image, sigma=1) # Gaussian filter expects float for proper behavior
```

✅ GOOD: Processing with `float`
```python
import numpy as np
from skimage import filters, util
from numpy.typing import NDArray, Any

def process_float_good(image: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """✅ GOOD: Converts to float for accurate processing."""
    float_image = util.img_as_float(image) # Converts to float64, scales to [0, 1]
    processed_float = filters.gaussian(float_image, sigma=1)
    return processed_float
```

### 3.2. Channel Order
**Always** use `(Height, Width, Channels)` (HWC) order for color images to maintain consistency with `matplotlib` and common deep learning frameworks. `scikit-image` functions generally expect channels as the last dimension.

### 3.3. Loading and Saving Images
**Always** use `skimage.io.imread` and `skimage.io.imsave`. These functions handle various image formats and integrate well with `scikit-image`'s `ndarray` expectations.

```python
from skimage import io
import numpy as np
from numpy.typing import NDArray, Any

def load_and_save_image(input_path: str, output_path: str) -> None:
    """Loads an image, processes it (placeholder), and saves it."""
    image: NDArray[Any, Any] = io.imread(input_path)
    # Perform processing here (e.g., convert_to_float, apply_filter, etc.)
    processed_image = image # Placeholder
    io.imsave(output_path, processed_image)
```

## 4. Performance Considerations

### 4.1. Vectorized Operations
**Always** leverage NumPy's vectorized operations. Avoid explicit Python loops over pixel values, as they are significantly slower.

❌ BAD: Pixel-wise loop
```python
import numpy as np

def invert_image_bad(image: np.ndarray) -> np.ndarray:
    """❌ BAD: Slow pixel-wise iteration."""
    inverted = np.zeros_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            inverted[r, c] = 255 - image[r, c] # Assuming uint8
    return inverted
```

✅ GOOD: Vectorized operation
```python
import numpy as np
from numpy.typing import NDArray, Any

def invert_image_good(image: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """✅ GOOD: Fast vectorized operation."""
    # Works for both uint8 (255 - pixel) and float (1.0 - pixel)
    return image.max() - image
```

### 4.2. Memory Management
Be mindful of memory usage with large images. Operations that create many intermediate copies can consume significant RAM. Where possible, use functions that operate in-place (if the function is designed for it and explicitly documented to do so, which is rare in `scikit-image`'s public API for image data).

## 5. Common Pitfalls and Gotchas

### 5.1. Incorrect Data Type Scaling
Forgetting to convert to float `[0, 1]` before processing, or failing to convert back to `uint8` `[0, 255]` (or `uint16` `[0, 65535]`) before saving, leads to incorrect results or corrupted images.

❌ BAD: Saving float image directly
```python
import numpy as np
from skimage import io, util

float_img = np.random.rand(100, 100) # Image in [0, 1] range
io.imsave("bad_float_image.png", float_img) # Will save as uint8, clipping values
```

✅ GOOD: Converting to appropriate integer type before saving
```python
import numpy as np
from skimage import io, util

float_img = np.random.rand(100, 100)
uint8_img = util.img_as_ubyte(float_img) # Converts float [0, 1] to uint8 [0, 255]
io.imsave("good_uint8_image.png", uint8_img)
```

### 5.2. Misinterpreting `skimage.data`
`skimage.data` provides example images. These are often `uint8` or `uint16`. **Always** convert them to `float` using `util.img_as_float` before performing image processing operations.

```python
from skimage import data, util, filters
import matplotlib.pyplot as plt

# Load example image
coins = data.coins()

# Process directly (BAD) vs. convert first (GOOD)
# bad_edges = filters.sobel(coins) # Sobel expects float for proper gradient calculation
good_edges = filters.sobel(util.img_as_float(coins))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(coins, cmap='gray')
axes[0].set_title('Original (uint8)')
axes[1].imshow(good_edges, cmap='gray') # Displaying float [0, 1] is fine for matplotlib
axes[1].set_title('Sobel Edges (float)')
plt.show()
```

## 6. Virtual Environments and Packaging

### 6.1. Virtual Environments
**Always** use a virtual environment (`venv` or `conda`) for `scikit-image` projects. This ensures reproducible environments and avoids dependency conflicts.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install scikit-image and other dependencies
pip install -U scikit-image numpy scipy matplotlib
```

### 6.2. Packaging
For project-specific dependencies, use `pyproject.toml` or `requirements.txt`. For development, install in editable mode:

```bash
# After activating your virtual environment
pip install -e .
```

## 7. Testing Approaches

### 7.1. Unit Tests
**Every** new function and significant change must be covered by unit tests using `pytest`. Tests should verify correctness across various inputs, edge cases, and data types. Use `numpy.testing.assert_array_equal` or `assert_allclose` for comparing image arrays.

```python
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from skimage import util
import pytest

# Assuming normalize_intensity is defined as in Section 2.1
def test_normalize_intensity_uint8():
    img = np.array([[0, 127, 255]], dtype=np.uint8)
    expected = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    result = normalize_intensity(util.img_as_float(img))
    assert_allclose(result, expected)

def test_normalize_intensity_float():
    img = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    expected = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    result = normalize_intensity(img)
    assert_allclose(result, expected)

def test_normalize_intensity_empty_image():
    img = np.array([], dtype=np.uint8).reshape(0, 0)
    expected = np.array([], dtype=np.float64).reshape(0, 0)
    result = normalize_intensity(util.img_as_float(img))
    assert_allclose(result, expected)
```