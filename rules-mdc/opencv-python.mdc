---
description: This guide provides definitive, opinionated best practices for writing high-performance, maintainable, and robust `opencv-python` code in Python applications.
globs: **/*.py
---
# `opencv-python` Best Practices

`opencv-python` is the de-facto standard for computer vision in Python. To write efficient, reliable, and maintainable code, treat OpenCV objects as NumPy arrays and prioritize vectorized operations. This guide outlines the essential practices for our team.

## Core Philosophy: Vectorize Everything

OpenCV functions are C++ optimized. Leverage them directly or use NumPy's vectorized operations instead of explicit Python loops for pixel-wise processing. This is the single most important performance rule.

❌ **BAD: Pixel-wise loop**
```python
import cv2
import numpy as np

def invert_image_slow(image: np.ndarray) -> np.ndarray:
    """Inverts an image pixel by pixel (slow)."""
    h, w, c = image.shape
    inverted_image = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            inverted_image[y, x] = 255 - image[y, x]
    return inverted_image
```

✅ **GOOD: Vectorized operation**
```python
import cv2
import numpy as np

def invert_image_fast(image: np.ndarray) -> np.ndarray:
    """Inverts an image using vectorized NumPy (fast)."""
    return 255 - image

# Or using OpenCV's built-in function for more complex ops
def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts an image to grayscale using cv2 (fast)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

## Code Organization and Structure

Organize your vision pipelines into small, focused functions. Each function should perform a single, well-defined image processing step.

❌ **BAD: Monolithic script**
```python
import cv2
import numpy as np

# main.py
img = cv2.imread('input.jpg')
if img is None:
    print("Error loading image")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_edges.jpg', edges)
```

✅ **GOOD: Modular functions**
```python
import cv2
import numpy as np
from typing import Optional

def load_image(path: str) -> Optional[np.ndarray]:
    """Loads an image, checking for errors."""
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image from {path}")
    return img

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Applies grayscale, blur, and Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def display_image(window_name: str, image: np.ndarray) -> None:
    """Displays an image and waits for a key press."""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# main.py
if __name__ == "__main__":
    input_path = 'input.jpg'
    output_path = 'output_edges.jpg'

    original_img = load_image(input_path)
    if original_img is not None:
        processed_img = preprocess_image(original_img)
        display_image('Processed Edges', processed_img)
        cv2.imwrite(output_path, processed_img)
```

## Common Pitfalls and Gotchas

### 1. `cv2.imread` Returns `None`

Always check the return value of `cv2.imread`. It returns `None` if the image cannot be found or loaded.

❌ **BAD: Assuming image load success**
```python
img = cv2.imread('non_existent.jpg')
# This will crash if img is None
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

✅ **GOOD: Robust image loading**
```python
img = cv2.imread('non_existent.jpg')
if img is None:
    raise FileNotFoundError("Image not found or could not be loaded.")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 2. Color Space Mismatches (BGR vs RGB)

OpenCV uses BGR (Blue, Green, Red) as its default color order for images. Matplotlib and many other libraries use RGB. Be explicit with conversions.

❌ **BAD: Displaying BGR as RGB**
```python
import matplotlib.pyplot as plt
img_bgr = cv2.imread('image.jpg')
plt.imshow(img_bgr) # Colors will be wrong
plt.show()
```

✅ **GOOD: Explicit color conversion**
```python
import matplotlib.pyplot as plt
img_bgr = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()
```

### 3. Data Type Mismatches

OpenCV functions expect specific data types (e.g., `np.uint8` for most image operations, `np.float32` for some calculations like `cv2.filter2D`). Always ensure your image data has the correct `dtype`.

❌ **BAD: Incorrect data type for operations**
```python
img_float = np.random.rand(100, 100, 3) # float64
# This will likely fail or produce unexpected results
edges = cv2.Canny(img_float, 50, 150)
```

✅ **GOOD: Explicit type casting**
```python
img_float = np.random.rand(100, 100, 3).astype(np.float32) * 255
img_uint8 = img_float.astype(np.uint8)
edges = cv2.Canny(img_uint8, 50, 150)
```

## Type Hints

Always use Python type hints, especially for function arguments and return values involving `numpy.ndarray`. This improves code readability, enables static analysis, and reduces bugs.

```python
import numpy as np
import cv2
from typing import Tuple

def resize_image(image: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Resizes an image to the specified (width, height).
    Args:
        image: The input image as a NumPy array.
        new_size: A tuple (width, height) for the new dimensions.
    Returns:
        The resized image.
    """
    return cv2.resize(image, new_size)

# Example usage
img = cv2.imread('input.jpg')
if img is not None:
    resized_img = resize_image(img, (640, 480))
```

## Performance Considerations

1.  **Prefer `cv2` functions**: They are highly optimized C++ implementations.
2.  **Vectorize with NumPy**: For operations not directly available in `cv2`, use NumPy's vectorized functions.
3.  **Avoid unnecessary copies**: Operations like `img[y1:y2, x1:x2]` create views, which are efficient. `img.copy()` creates a new array. Use `copy()` only when you intend to modify the subset independently.
4.  **Image `dtype`**: Stick to `np.uint8` for most image storage. Convert to `np.float32` or `np.float64` only for mathematical operations requiring higher precision, then convert back if needed.
5.  **Batch Processing**: If processing many images, consider loading and processing them in batches where possible, especially for deep learning inference.

## Virtual Environments and Packaging

Always use virtual environments (`venv` or `conda`) to isolate project dependencies. Pin `opencv-python` and `numpy` versions in `requirements.txt` for reproducibility.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Linux/macOS
# .venv\Scripts\activate # Windows

# Install packages
pip install opencv-python==4.12.0.88 numpy<2.0 # For Python < 3.9
# pip install opencv-python==4.12.0.88 numpy<2.3.0,>=2 # For Python >= 3.9
```

`requirements.txt`:
```
opencv-python==4.12.0.88
numpy<2.0; python_version < "3.9"
numpy<2.3.0,>=2; python_version >= "3.9"
```

## Testing Approaches

Implement unit and integration tests for your image processing pipelines. `pytest` is the recommended framework.

1.  **Unit Tests**: Test individual image processing functions with known inputs and expected outputs.
2.  **Integration Tests**: Test the entire pipeline with sample images, verifying the final output.
3.  **Image Comparison**: For asserting image correctness, load a reference image and compare it pixel-wise using `numpy.array_equal` or a perceptual difference metric.

```python
# tests/test_image_processing.py
import pytest
import numpy as np
import cv2
from my_module import preprocess_image, load_image

def test_preprocess_image_grayscale():
    """Test if preprocess_image correctly grayscales and blurs."""
    # Create a dummy color image (BGR)
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_img[:, :, 0] = 255 # Blue channel
    dummy_img[:, :, 1] = 128 # Green channel
    dummy_img[:, :, 2] = 64  # Red channel

    processed_img = preprocess_image(dummy_img)

    # Expected output for grayscale of (255, 128, 64) is ~125
    # Canny output is binary, so check shape and type
    assert processed_img.shape == (10, 10) # Grayscale, then Canny
    assert processed_img.dtype == np.uint8
    # Further checks could involve comparing with a pre-computed reference image
    # For example:
    # expected_edges = cv2.imread('tests/ref_edges.png', cv2.IMREAD_GRAYSCALE)
    # assert np.array_equal(processed_img, expected_edges)
```