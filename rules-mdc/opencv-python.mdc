---
description: This rule outlines best practices for developing with the opencv-python library, focusing on code organization, performance, security, testing, and common pitfalls. It provides comprehensive guidelines for efficient and maintainable opencv-python projects.
globs: *.py
---
- Always use UV or Pipenv when installing dependencies for consistent environments.
- Always specify Python 3.10 or higher, as older versions may lack feature parity and security updates.
- Use type hints extensively for improved code readability and maintainability.

# Library Information:
- Name: opencv-python
- Tags: python, image-processing, computer-vision, opencv

## 1. Code Organization and Structure

- **Directory Structure:**
    - 
    project_root/
    ├── data/           # Store image, video, and other data files.
    ├── src/            # Source code directory.
    │   ├── modules/     # Reusable modules and utility functions.
    │   │   ├── image_processing/  # Specific image processing functions.
    │   │   │   ├── __init__.py
    │   │   │   ├── filters.py  # Implement filters as functions or classes
    │   │   │   ├── transforms.py # Implement image transforms as functions or classes
    │   │   ├── utils.py       # Utility functions for data loading, etc.
    │   ├── main.py        # Entry point of your application.
    ├── tests/          # Test suite.
    │   ├── unit/        # Unit tests for individual components.
    │   ├── integration/   # Integration tests for combined components.
    ├── models/          # Saved model files
    ├── notebooks/      # Jupyter notebooks for experimentation.
    ├── requirements.txt # Project dependencies.
    ├── pyproject.toml   # Project metadata and build configuration.
    └── README.md        # Project documentation.
    

- **File Naming Conventions:**
    - Use descriptive and consistent naming.
    - `module_name.py` (e.g., `image_utils.py`, `feature_extraction.py`).
    - Class names: `PascalCase` (e.g., `ImageProcessor`, `FeatureDetector`).
    - Function names: `snake_case` (e.g., `load_image`, `apply_filter`).

- **Module Organization:**
    - Group related functions and classes into modules within the `src/modules/` directory.
    - Use `__init__.py` files to make directories packages.
    - Employ clear and concise module names.

- **Component Architecture:**
    - Follow a layered architecture:
        - **Data Access Layer:** Handles image loading, saving, and data source interactions.
        - **Processing Layer:** Implements image processing algorithms and feature extraction.
        - **Application Layer:** Orchestrates the processing and presents the results.

- **Code Splitting:**
    - Decompose complex functions into smaller, well-defined units.
    - Create separate modules for distinct functionalities (e.g., filtering, feature detection, object tracking).
    - Utilize classes to encapsulate related state and behavior.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Strategy Pattern:** To swap image processing algorithms at runtime.
    - **Factory Pattern:** To create different image processing objects based on configuration.
    - **Observer Pattern:** To notify subscribers of image processing events.

- **Recommended Approaches:**
    - **Image Loading:** Use `cv2.imread()` for loading images.  Handle potential file errors gracefully.
    - **Image Display:** Use `cv2.imshow()` for displaying images, and remember to call `cv2.waitKey()` to prevent freezing.
    - **Video Processing:** Use `cv2.VideoCapture()` to capture video from a file or camera.
    - **Iterating Pixels:** Access pixel values directly using NumPy array indexing for performance.  Avoid slow Python loops when possible.

- **Anti-patterns and Code Smells:**
    - **Deeply Nested Loops:** Indicate inefficient pixel-level operations. Consider vectorization using NumPy.
    - **Global Variables:** Lead to unpredictable state. Encapsulate state within classes.
    - **Hardcoded Paths:** Make code inflexible. Use relative paths and configurable settings.
    - **Ignoring Errors:** Can lead to unexpected crashes. Implement robust error handling.

- **State Management:**
    - Encapsulate state within classes.
    - Minimize mutable state; favor immutable data structures where appropriate.
    - Use dependency injection to manage external dependencies.

- **Error Handling:**
    - Use `try...except` blocks to handle potential exceptions (e.g., file not found, invalid image format).
    - Log errors for debugging and monitoring.
    - Raise exceptions to propagate errors up the call stack.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Vectorization:** Use NumPy's vectorized operations instead of loops for faster processing.
    - **Pre-allocation:** Pre-allocate NumPy arrays instead of repeatedly resizing them.
    - **Caching:** Cache frequently accessed data (e.g., lookup tables) to avoid recalculation.
    - **Multiprocessing/Multithreading:** Use `multiprocessing` or `threading` to parallelize independent image processing tasks.
    - **Cython:** Implement performance-critical sections in Cython for C-like speeds.
    - **GPU Acceleration:** Leverage OpenCV's CUDA module for GPU-accelerated processing (if applicable).

- **Memory Management:**
    - Release unused memory by explicitly deleting large arrays or objects.
    - Be mindful of memory copies. Use `view()` or `ascontiguousarray()` to avoid unnecessary copies.
    - Use generators to process large images in chunks.

- **Rendering Optimization:**
    - Minimize the number of draw calls.
    - Use efficient drawing functions (e.g., `cv2.polylines()` instead of individual `cv2.line()` calls).

- **Bundle Size Optimization:**
    - Not directly applicable to opencv-python. For applications that use opencv-python, dependencies should be managed effectively. 

- **Lazy Loading:**
    - Load images only when needed.
    - Use lazy initialization for computationally expensive objects.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **Denial of Service (DoS):** Prevent processing extremely large or malformed images that can consume excessive resources.
    - **Code Injection:** Avoid executing arbitrary code based on user input (e.g., constructing file paths dynamically without proper sanitization).
    - **Buffer Overflows:** Check image dimensions and data types to prevent buffer overflows when accessing pixel data directly.

- **Input Validation:**
    - Validate image file extensions and formats.
    - Check image dimensions and data types.
    - Sanitize filenames to prevent path traversal attacks.

- **Authentication and Authorization:**
    - Not typically applicable to core opencv-python functionality. These are relevant for applications leveraging opencv-python for tasks like facial recognition or video surveillance, requiring access control mechanisms.

- **Data Protection:**
    - Encrypt sensitive image data at rest and in transit (if applicable).
    - Implement access control mechanisms to restrict access to sensitive image data.
    - Anonymize or redact personally identifiable information (PII) from images.

- **Secure API Communication:**
    - Enforce HTTPS for all API communication.
    - Use secure authentication mechanisms (e.g., API keys, OAuth).
    - Validate all API requests and responses.

## 5. Testing Approaches

- **Unit Testing:**
    - Test individual functions and classes in isolation.
    - Use mocking to isolate components from external dependencies (e.g., file system, camera).
    - Test edge cases and boundary conditions.
    - Use libraries like `pytest` and `unittest`.

- **Integration Testing:**
    - Test the interaction between different components.
    - Verify that data flows correctly between modules.
    - Use realistic test data.

- **End-to-End Testing:**
    - Test the entire application workflow.
    - Simulate user interactions.
    - Verify that the application meets the requirements.

- **Test Organization:**
    - Structure tests to mirror the source code organization.
    - Use descriptive test names.
    - Keep tests concise and focused.

- **Mocking and Stubbing:**
    - Use `unittest.mock` or `pytest-mock` to create mock objects for external dependencies.
    - Stub out functions that perform I/O operations or interact with hardware.

## 6. Common Pitfalls and Gotchas

- **BGR vs. RGB:** Be aware that OpenCV uses BGR color format by default, while other libraries (e.g., Matplotlib, PIL) use RGB. Convert between formats using `cv2.cvtColor()`.
- **Image Data Types:** Understand the different image data types (e.g., `uint8`, `float32`) and their implications for processing.
- **Memory Layout:** Be aware of the memory layout of NumPy arrays and use `ascontiguousarray()` when necessary.
- **`cv2.waitKey()`:** Remember to call `cv2.waitKey()` after `cv2.imshow()` to display the image and allow the window to respond to events. A value of 0 will wait indefinitely for a key press.
- **Incorrect File Paths:** Double-check file paths when loading images or videos.
- **Version Compatibility:** Be aware of compatibility issues between different versions of opencv-python.
- **Global Interpreter Lock (GIL):** Understand the implications of the GIL for multithreading in Python. Use multiprocessing for CPU-bound tasks.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **IDE:** VS Code, PyCharm, Jupyter Notebook.
    - **Virtual Environment:** `venv`, `conda`.
    - **Package Manager:** `pip`, `uv`.

- **Build Configuration:**
    - Use `pyproject.toml` for project metadata and build configuration.
    - Specify dependencies in `requirements.txt` or `pyproject.toml`.
    - Use a build system like `setuptools` or `poetry`.

- **Linting and Formatting:**
    - Use `flake8` for linting.
    - Use `black` for formatting.
    - Configure your IDE to automatically format code on save.

- **Deployment:**
    - Containerize your application using Docker.
    - Use a deployment platform like AWS, Azure, or Google Cloud.

- **CI/CD:**
    - Integrate with a CI/CD system like Jenkins, GitLab CI, or GitHub Actions.
    - Automate testing, linting, and deployment.