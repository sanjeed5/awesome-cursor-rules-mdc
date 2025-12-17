---
description: Definitive guidelines for using Pillow (Python Imaging Library) effectively, focusing on modern best practices, performance, and maintainability.
globs: **/*.py
---
# Pillow Best Practices

Pillow is the definitive image processing library for Python. Adhere to these guidelines to ensure your image manipulation code is robust, performant, and maintainable.

## Critical Guidelines:

### 1. Resource Management: Always Use Context Managers

Images opened from files consume system resources. Ensure they are properly closed using `with` statements.

❌ BAD:
```python
from PIL import Image

im = Image.open("input.jpg")
# ... process im ...
# Resource might leak if not explicitly closed or on error
# im.close() # Often forgotten, especially on exceptions
```

✅ GOOD:
```python
from PIL import Image

def process_image_with_thumbnail(path: str) -> None:
    with Image.open(path) as im:
        # Image is automatically closed when exiting the 'with' block
        im.thumbnail((128, 128))
        im.save("thumbnail.jpg")
```

### 2. Prefer High-Level API for Common Operations

Pillow provides a rich, high-level API for most common image transformations. Use these methods instead of manual pixel manipulation or low-level `ImageFile` operations.

❌ BAD:
```python
from PIL import Image
# Avoid manual pixel access for transformations; it's inefficient and error-prone
def resize_manual(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    new_im = Image.new(im.mode, size)
    # ... complex, slow loop to copy pixels ...
    return new_im
```

✅ GOOD:
```python
from PIL import Image, ImageFilter

def apply_common_transforms(im: Image.Image) -> Image.Image:
    # Use built-in methods for efficiency and correctness
    im_resized = im.resize((256, 256), Image.Resampling.LANCZOS)
    im_rotated = im_resized.rotate(45, expand=True)
    im_blurred = im_rotated.filter(ImageFilter.BLUR)
    return im_blurred
```

### 3. Explicitly Import Custom Plugins

Pillow no longer auto-loads plugins (since 2.1.0). If you're using a custom image format or a third-party plugin, you *must* import it explicitly before use.

❌ BAD:
```python
from PIL import Image
# Assuming 'MyCustomImagePlugin.py' is on PYTHONPATH
# This will fail if MyCustomImagePlugin is not explicitly imported
try:
    with Image.open("custom.myformat") as im:
        im.save("output.png")
except Exception as e:
    print(f"Error: {e}") # Likely 'cannot identify image file'
```

✅ GOOD:
```python
from PIL import Image
import MyCustomImagePlugin # Explicitly import your plugin

def load_custom_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.copy() # Return a copy if further processing is needed outside the 'with' block
```

### 4. Optimize Image Saving for Web/Performance

When saving images, especially for web use, optimize for file size and quality. Always convert to `RGB` for JPEG to avoid issues with alpha channels.

❌ BAD:
```python
from PIL import Image
# Default JPEG quality (often 75) might not be optimal, no explicit optimization
with Image.open("input.png") as im:
    im.save("output.jpg")
```

✅ GOOD:
```python
from PIL import Image

def save_optimized_jpeg(im: Image.Image, path: str, quality: int = 85) -> None:
    # Convert to RGB if not already, as JPEG doesn't support RGBA
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    im.save(path, quality=quality, optimize=True)

def save_optimized_png(im: Image.Image, path: str, compress_level: int = 9) -> None:
    # PNG compression level 0-9 (9 is highest compression, slowest)
    im.save(path, compress_level=compress_level)
```

### 5. Use Type Hints for Clarity and Maintainability

Always type-hint Pillow objects, especially `Image.Image`, to improve code readability and enable static analysis.

❌ BAD:
```python
from PIL import Image

def process_img(img_obj, scale): # What is img_obj? What does it return?
    return img_obj.resize((img_obj.width // scale, img_obj.height // scale))
```

✅ GOOD:
```python
from PIL import Image

def process_img(img_obj: Image.Image, scale: int) -> Image.Image:
    """Resizes an image by a given scale factor."""
    new_size = (img_obj.width // scale, img_obj.height // scale)
    return img_obj.resize(new_size, Image.Resampling.BICUBIC)
```

### 6. Manage Dependencies with Virtual Environments

Always install Pillow within an isolated virtual environment to prevent dependency conflicts and ensure reproducible builds.

❌ BAD:
```bash
pip install Pillow # Global installation, pollutes system Python
```

✅ GOOD:
```bash
python3 -m venv .venv
source .venv/bin/activate # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install --upgrade Pillow defusedxml olefile # Include optional dependencies for broader format support
```

### 7. Test Image Operations Rigorously

Write unit tests for functions that process images. Use small, controlled input images and assert expected output properties or pixel differences.

❌ BAD:
```python
# No tests for image processing logic, leading to potential regressions
def watermark_image(im: Image.Image, watermark_text: str) -> Image.Image:
    # ... complex drawing logic ...
    return im
```

✅ GOOD:
```python
from PIL import Image, ImageDraw, ImageChops
import io

def watermark_image(im: Image.Image, watermark_text: str) -> Image.Image:
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), watermark_text, fill=(255, 255, 255))
    return im

# Example test function (integrate into your test suite, e.g., pytest)
def test_watermark_image():
    original_image = Image.new("RGB", (100, 100), "blue")
    watermarked_image = watermark_image(original_image.copy(), "TEST")

    # Create a reference image with the expected watermark
    expected_image = Image.new("RGB", (100, 100), "blue")
    draw_expected = ImageDraw.Draw(expected_image)
    draw_expected.text((10, 10), "TEST", fill=(255, 255, 255))

    # Assert that the images are identical using ImageChops.difference
    diff = ImageChops.difference(watermarked_image, expected_image)
    assert not diff.getbbox(), "Watermarked image does not match expected output"
```

### 8. Stay Current with Pillow Releases

Regularly update Pillow to benefit from performance improvements, bug fixes, and new features, and to avoid deprecated APIs. Always check release notes for breaking changes (e.g., the 12.x series deprecates older `ImageFile` shortcuts).

❌ BAD:
```python
# Sticking to an old version (e.g., < 7.0)
# Relying on auto-loading plugins or deprecated ImageFile shortcuts
```

✅ GOOD:
```bash
# Periodically update your environment to the latest stable version
pip install --upgrade Pillow
```
```python
# Consult release notes (e.g., Pillow 12.1.0.dev0) and adapt code
# to new APIs and deprecation warnings as they arise.
```