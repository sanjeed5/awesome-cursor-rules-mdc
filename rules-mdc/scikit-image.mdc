---
description: This rule provides guidelines for best practices and coding standards when using the scikit-image library for image processing in Python. It covers code organization, performance, security, testing, and common pitfalls.
globs: *.py
---
- Always use UV when installing dependencies for faster and more deterministic dependency resolution.
- Always use Python 3.12 or later to leverage the latest language features and performance improvements.
- Utilize classes instead of standalone functions where appropriate for better code organization and encapsulation, especially when dealing with stateful image processing operations.

## Scikit-image Best Practices and Coding Standards

This document outlines the best practices and coding standards for using the scikit-image library in Python for image processing. Following these guidelines will help ensure code clarity, maintainability, performance, and security.

### Library Information:
- Name: scikit-image
- Tags: python, image-processing, scientific-computing

### 1. Code Organization and Structure:

- **Directory Structure:**
    - Adopt a modular directory structure to organize your scikit-image projects.
    - Example:
      
      project_name/
          ├── data/          # Contains image data (e.g., input images, sample datasets)
          ├── src/           # Source code directory
          │   ├── __init__.py  # Marks src as a Python package
          │   ├── modules/
          │   │   ├── __init__.py
          │   │   ├── image_io.py   # Image input/output related functions
          │   │   ├── processing.py # Core image processing algorithms
          │   │   ├── segmentation.py # Segmentation algorithms
          │   │   └── feature.py    # Feature extraction modules
          │   ├── utils.py       # Utility functions
          │   └── main.py        # Main application entry point
          ├── tests/         # Unit and integration tests
          │   ├── __init__.py
          │   ├── test_image_io.py
          │   ├── test_processing.py
          │   └── test_segmentation.py
          ├── notebooks/    # Jupyter notebooks for exploration
          ├── requirements.txt # Project dependencies
          ├── pyproject.toml   # Project metadata and build system
          └── README.md      # Project documentation
      

- **File Naming Conventions:**
    - Use descriptive and consistent file names.
    - Module files: `image_io.py`, `processing.py`, `segmentation.py`
    - Test files: `test_image_io.py`, `test_processing.py`
    - Utility files: `utils.py`

- **Module Organization:**
    - Group related functions and classes into modules.
    - Avoid monolithic modules; break down large modules into smaller, more manageable ones.
    - Use `__init__.py` files to define packages and control namespace exposure.
    - Example (in `src/modules/processing.py`):
      python
      from skimage import filters
      from skimage import morphology
      import numpy as np

      def apply_threshold(image, threshold_value=128):
          """Applies a simple threshold to an image."""
          return image > threshold_value

      def remove_small_objects(binary_image, min_size=100):
          """Removes small connected components from a binary image."""
          return morphology.remove_small_objects(binary_image, min_size=min_size)
      

- **Component Architecture:**
    - Design components with clear responsibilities and well-defined interfaces.
    - Favor composition over inheritance for greater flexibility.
    - Use abstract base classes (ABCs) to define interfaces for components.
    - Example:
      python
      from abc import ABC, abstractmethod

      class ImageProcessor(ABC):
          @abstractmethod
          def process_image(self, image):
              pass

      class GrayscaleConverter(ImageProcessor):
          def process_image(self, image):
              from skimage.color import rgb2gray
              return rgb2gray(image)
      

- **Code Splitting Strategies:**
    - Decompose complex image processing pipelines into smaller, reusable functions.
    - Use generators or iterators for processing large images in chunks.
    - Consider using multiprocessing or multithreading for parallel processing of image regions.
    - Utilize lazy loading techniques for large image datasets.

### 2. Common Patterns and Anti-patterns:

- **Design Patterns:**
    - **Factory Pattern:** Use factory functions or classes to create image processing objects.
    - **Strategy Pattern:** Implement different image processing algorithms as strategies that can be swapped at runtime.
    - **Observer Pattern:** Notify observers when an image processing operation completes.

- **Recommended Approaches:**
    - Use NumPy arrays as the primary data structure for image representation.
    - Leverage scikit-image's functional API for modular and composable image processing pipelines.
    - Use `img_as_float` or other data type conversion utilities to ensure consistent data types.
    - Document image processing functions and classes using docstrings.

- **Anti-patterns and Code Smells:**
    - **Global State:** Avoid using global variables to store image data or processing parameters.
    - **Magic Numbers:** Use named constants instead of hardcoded numerical values.
    - **Deeply Nested Loops:** Optimize image processing loops using NumPy's vectorized operations.
    - **Ignoring Data Types:** Always be aware of the data types of images and intermediate results.
    - **Over-Complicating Code:** Aim for simplicity and readability in your image processing code.

- **State Management:**
    - Encapsulate state within classes or data structures.
    - Use immutable data structures where possible.
    - Avoid modifying image data in-place unless necessary.

- **Error Handling:**
    - Use try-except blocks to handle potential exceptions (e.g., file I/O errors, invalid image formats).
    - Log errors and warnings using the `logging` module.
    - Provide informative error messages to the user.
    - Consider custom exception types for scikit-image related errors.
    - Example:
        python
        import logging
        from skimage import io

        logging.basicConfig(level=logging.ERROR)

        def load_image(filepath):
            try:
                image = io.imread(filepath)
                return image
            except FileNotFoundError:
                logging.error(f"File not found: {filepath}")
                return None
            except Exception as e:
                logging.exception(f"Error loading image: {e}")
                return None
        

### 3. Performance Considerations:

- **Optimization Techniques:**
    - Vectorize image processing operations using NumPy's broadcasting and array manipulation features.
    - Use Cython to optimize performance-critical sections of code.
    - Explore Numba for just-in-time (JIT) compilation of image processing functions.
    - Utilize `skimage.util.apply_parallel` for parallel processing of image regions.
    - Example using NumPy vectorization:
      python
      import numpy as np

      def brighten_image(image, factor=1.5):
          """Brightens an image by multiplying each pixel by a factor."""
          return np.clip(image * factor, 0, 255).astype(image.dtype) # Clip values to valid range
      

- **Memory Management:**
    - Use appropriate data types to minimize memory usage (e.g., `uint8` for grayscale images).
    - Release large image arrays when they are no longer needed.
    - Avoid creating unnecessary copies of image data.
    - Consider using memory-mapped arrays for very large images.

- **Rendering Optimization:**
    - Optimize image display using appropriate colormaps and scaling.
    - Use hardware acceleration (e.g., OpenGL) for faster rendering.

- **Bundle Size Optimization:**
    - Minimize dependencies in your scikit-image projects.
    - Use tree shaking to remove unused code from your bundles.
    - Compress image assets using appropriate compression algorithms.

- **Lazy Loading:**
    - Load large images only when they are needed.
    - Use generators or iterators to process images in chunks.
    - Implement caching mechanisms to avoid redundant image loading.

### 4. Security Best Practices:

- **Common Vulnerabilities:**
    - **Denial-of-Service (DoS) Attacks:** Protect against DoS attacks by limiting the size of input images.
    - **Code Injection:** Sanitize user-provided image processing parameters to prevent code injection attacks.

- **Input Validation:**
    - Validate the format, size, and data type of input images.
    - Check for malicious image headers or metadata.
    - Sanitize user-provided parameters to prevent code injection.
    - Example:
      python
      from skimage import io

      def process_image(filepath, resize_factor):
          if not isinstance(resize_factor, (int, float)):
              raise ValueError("Resize factor must be a number.")
          if resize_factor <= 0:
              raise ValueError("Resize factor must be positive.")

          try:
              image = io.imread(filepath)
              # Perform image processing operations using resize_factor
          except Exception as e:
              print(f"Error processing image: {e}")
      

- **Authentication and Authorization:**
    - Implement authentication and authorization mechanisms to control access to image processing resources.
    - Use secure protocols (e.g., HTTPS) for API communication.

- **Data Protection:**
    - Encrypt sensitive image data at rest and in transit.
    - Implement access control policies to protect image data.
    - Use secure storage mechanisms for image data.

- **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Implement rate limiting to prevent abuse.
    - Use input validation to prevent injection attacks.

### 5. Testing Approaches:

- **Unit Testing:**
    - Write unit tests for individual functions and classes.
    - Use mocking and stubbing to isolate components during testing.
    - Test edge cases and boundary conditions.

- **Integration Testing:**
    - Write integration tests to verify the interaction between components.
    - Test complex image processing pipelines.
    - Use realistic test data.

- **End-to-End Testing:**
    - Write end-to-end tests to verify the entire application workflow.
    - Use automated testing frameworks (e.g., Selenium).

- **Test Organization:**
    - Organize tests into separate directories for unit, integration, and end-to-end tests.
    - Use descriptive test names.
    - Follow a consistent testing style.

- **Mocking and Stubbing:**
    - Use mocking libraries (e.g., `unittest.mock`) to replace external dependencies with mock objects.
    - Use stubbing to provide predefined outputs for specific function calls.

### 6. Common Pitfalls and Gotchas:

- **Data Type Issues:**
    - Be aware of the data types of images and intermediate results.
    - Use `img_as_float` or other data type conversion utilities to ensure consistent data types.
    - Pay attention to data type ranges (e.g., 0-255 for `uint8`, 0.0-1.0 for float).

- **Coordinate Conventions:**
    - Understand the coordinate conventions used by scikit-image (row, col) and NumPy.

- **Memory Consumption:**
    - Avoid creating unnecessary copies of image data.
    - Process large images in chunks or tiles.

- **Version Compatibility:**
    - Be aware of version-specific API changes and deprecations.
    - Check scikit-image's changelog for breaking changes.

- **Image I/O Issues:**
    - Use appropriate image formats for your application.
    - Handle file I/O errors gracefully.

### 7. Tooling and Environment:

- **Recommended Development Tools:**
    - IDE: VS Code, PyCharm
    - Debugger: pdb, ipdb
    - Profiler: cProfile, line_profiler

- **Build Configuration:**
    - Use `pyproject.toml` to manage project metadata and build dependencies.
    - Use `requirements.txt` to specify project dependencies.

- **Linting and Formatting:**
    - Use a linter (e.g., flake8, pylint) to enforce coding style and detect errors.
    - Use a formatter (e.g., black, autopep8) to automatically format code.
    - Configure your IDE to run linters and formatters automatically.

- **Deployment:**
    - Use virtual environments to isolate project dependencies.
    - Containerize your scikit-image applications using Docker.
    - Deploy your applications to cloud platforms (e.g., AWS, Azure, GCP).

- **CI/CD Integration:**
    - Use a CI/CD pipeline (e.g., GitHub Actions, GitLab CI, Jenkins) to automate testing, building, and deployment.
    - Run linters, formatters, and tests in your CI/CD pipeline.
    - Use code coverage tools to measure the effectiveness of your tests.

By adhering to these best practices, you can develop robust, maintainable, and performant image processing applications using the scikit-image library.