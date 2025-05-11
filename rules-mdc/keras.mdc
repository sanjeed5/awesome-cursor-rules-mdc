---
description: This rule enforces Keras library best practices, focusing on code clarity, modularity, performance optimization, and security considerations. It provides actionable guidance for developers to improve the quality and maintainability of Keras-based machine learning projects.
globs: *.py
---
# Keras Development Best Practices

This document outlines best practices for developing Keras applications. It covers various aspects of software engineering, including code organization, common patterns, performance, security, testing, and tooling.

Library Information:
- Name: keras
- Tags: ai, ml, machine-learning, python, deep-learning

## 1. Code Organization and Structure

### 1.1. Directory Structure

Adopt a well-defined directory structure to enhance maintainability and collaboration.


project_root/
├── data/                      # Contains datasets (raw, processed)
├── models/                    # Saved models (weights, architectures)
├── src/                       # Source code
│   ├── layers/                # Custom Keras layers
│   ├── models/                # Model definitions
│   ├── utils/                 # Utility functions
│   ├── callbacks/            # Custom Keras Callbacks
│   ├── preprocessing/        # Data preprocessing scripts
│   └── __init__.py          # Makes 'src' a Python package
├── notebooks/                 # Jupyter notebooks for experimentation
├── tests/                     # Unit and integration tests
├── requirements.txt           # Project dependencies
├── README.md                  # Project overview
└── .gitignore                 # Specifies intentionally untracked files that Git should ignore


### 1.2. File Naming Conventions

Use descriptive and consistent file names.

-   `model_name.py`:  For defining Keras models.
-   `layer_name.py`: For custom Keras layers.
-   `utils.py`: For utility functions.
-   `data_preprocessing.py`: For data preprocessing scripts.
-   `training_script.py`: Main training script.

### 1.3. Module Organization

-   **Single Responsibility Principle:** Each module should have a clear and specific purpose.
-   **Loose Coupling:** Minimize dependencies between modules.
-   **High Cohesion:**  Keep related functions and classes within the same module.

python
# src/models/my_model.py

import keras
from keras import layers

def create_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


### 1.4. Component Architecture

-   **Layers:** Encapsulate reusable blocks of computation (e.g., custom convolutional layers, attention mechanisms).
-   **Models:** Define the overall architecture by combining layers.
-   **Callbacks:** Implement custom training behaviors (e.g., early stopping, learning rate scheduling).
-   **Preprocessing:** Separate data loading, cleaning, and transformation logic.

### 1.5. Code Splitting

-   **Functions:** Break down complex logic into smaller, well-named functions.
-   **Classes:** Use classes to represent stateful components (e.g., custom layers with trainable parameters).
-   **Packages:** Organize modules into packages for larger projects.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **Functional API:** Use the Keras Functional API for building complex, multi-input/output models.

    python
    input_tensor = keras.Input(shape=(784,))
    hidden_layer = layers.Dense(units=64, activation='relu')(input_tensor)
    output_tensor = layers.Dense(units=10, activation='softmax')(hidden_layer)
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    

-   **Subclassing:**  Subclass `keras.Model` or `keras.layers.Layer` for maximum customization.

    python
    class MyLayer(layers.Layer):
        def __init__(self, units=32, **kwargs):
            super(MyLayer, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True)
            self.b = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)

        def call(self, inputs):
            return keras.activations.relu(tf.matmul(inputs, self.w) + self.b)
    

-   **Callbacks:** Implement custom training behaviors (e.g., custom logging, model checkpointing).

    python
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f'Epoch {epoch}: Loss = {logs['loss']}')
    

### 2.2. Recommended Approaches

-   **Data Input Pipelines:** Use `tf.data.Dataset` for efficient data loading and preprocessing.
-   **Model Checkpointing:**  Save model weights during training to prevent data loss and allow for resuming training.
-   **Early Stopping:**  Monitor validation loss and stop training when it plateaus to prevent overfitting.
-   **Learning Rate Scheduling:**  Adjust the learning rate during training to improve convergence.

### 2.3. Anti-Patterns

-   **Hardcoding:** Avoid hardcoding values directly into your code. Use variables and configuration files instead.
-   **Global Variables:** Minimize the use of global variables to prevent namespace pollution and unexpected side effects.
-   **Over-Engineering:**  Don't overcomplicate your code with unnecessary abstractions or complex patterns.
-   **Ignoring Warnings:** Pay attention to warnings and deprecation messages, as they often indicate potential problems.
-   **Training on the entire dataset without validation:** Always split your data into training, validation and testing sets to avoid overfitting.

### 2.4. State Management

-   **Stateless Operations:**  Prefer stateless operations whenever possible to simplify testing and debugging.
-   **Model Weights:**  Store model weights separately from the model architecture.
-   **Configuration Files:**  Use configuration files (e.g., JSON, YAML) to manage hyperparameters and other settings.

### 2.5. Error Handling

-   **Exception Handling:**  Use `try...except` blocks to handle potential exceptions gracefully.
-   **Logging:**  Log errors and warnings to help diagnose problems.
-   **Validation:** Validate input data to prevent unexpected errors.
-   **Assertions:** Use `assert` statements to check for conditions that should always be true.

python
try:
    model = keras.models.load_model('my_model.h5')
except FileNotFoundError:
    logging.error('Model file not found.')
    raise


## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **GPU Acceleration:**  Utilize GPUs for faster training and inference.
-   **Data Preprocessing:**  Optimize data preprocessing pipelines to reduce overhead.
-   **Batch Size:**  Adjust the batch size to maximize GPU utilization.
-   **Model Pruning:**  Remove unnecessary weights from the model to reduce its size and improve its speed.
-   **Quantization:**  Reduce the precision of model weights to reduce memory consumption and improve inference speed.
-   **Mixed Precision Training:** Use `tf.keras.mixed_precision.Policy` to enable mixed precision training for faster training on modern GPUs.

### 3.2. Memory Management

-   **Garbage Collection:**  Be mindful of memory leaks and use garbage collection to reclaim unused memory.
-   **Data Types:**  Use appropriate data types to minimize memory consumption (e.g., `tf.float16` instead of `tf.float32`).
-   **Generators:**  Use generators to load data in batches, reducing memory usage.

### 3.3. Rendering Optimization (If applicable)

Not directly applicable to Keras itself, but relevant when visualizing model outputs or training progress.  Use libraries like `matplotlib` or `seaborn` efficiently and consider downsampling large datasets before plotting.

### 3.4. Bundle Size Optimization

-   **Model Pruning and Quantization:** as above. 
-   **Selectively Import Keras Modules**: Only import the specific Keras modules needed to reduce the overall bundle size, e.g., `from keras.layers import Dense, Conv2D` instead of `import keras.layers`.

### 3.5. Lazy Loading

-   **Lazy Initialization:**  Defer the initialization of resources until they are actually needed.
-   **Data Loading:** Load data on demand rather than loading the entire dataset into memory.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

-   **Adversarial Attacks:**  Protect against adversarial attacks that can fool models into making incorrect predictions.
-   **Data Poisoning:**  Ensure the integrity of training data to prevent data poisoning attacks.
-   **Model Extraction:** Protect against model extraction attacks that can steal intellectual property.

### 4.2. Input Validation

-   **Sanitize Input:**  Sanitize input data to prevent injection attacks.
-   **Validate Input:**  Validate input data to ensure that it conforms to the expected format and range.

python
def predict(model, input_data):
    if not isinstance(input_data, np.ndarray):
        raise TypeError('Input data must be a NumPy array.')
    if input_data.shape != (1, 784):
        raise ValueError('Input data must have shape (1, 784).')
    return model.predict(input_data)


### 4.3. Authentication and Authorization

-   **Secure API:**  Implement secure API communication using HTTPS.
-   **Authentication:**  Require authentication for access to sensitive data and functionality.
-   **Authorization:**  Enforce authorization policies to control access to resources.

### 4.4. Data Protection

-   **Encryption:**  Encrypt sensitive data at rest and in transit.
-   **Anonymization:** Anonymize data to protect privacy.
-   **Data Governance:** Implement data governance policies to ensure data quality and security.

### 4.5. Secure API Communication

-   **HTTPS:**  Use HTTPS for all API communication to encrypt data in transit.
-   **API Keys:**  Use API keys to authenticate requests.
-   **Rate Limiting:**  Implement rate limiting to prevent denial-of-service attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

-   **Test-Driven Development:** Write unit tests before writing code to ensure that the code meets the requirements.
-   **Test Cases:**  Create test cases for different scenarios, including edge cases and error conditions.
-   **Assertions:** Use assertions to verify that the code behaves as expected.

### 5.2. Integration Testing

-   **Component Interaction:**  Test the interaction between different components of the application.
-   **Data Flow:**  Test the flow of data through the application.
-   **System Integration:** Test the integration of the application with other systems.

### 5.3. End-to-End Testing

-   **User Interface:**  Test the user interface to ensure that it is functional and user-friendly.
-   **Workflow:**  Test the entire workflow from start to finish.
-   **Real-World Scenarios:** Test the application in real-world scenarios to ensure that it meets the needs of the users.

### 5.4. Test Organization

-   **Test Directory:** Create a dedicated `tests` directory to store test files.
-   **Test Modules:** Organize tests into modules based on the components they test.
-   **Test Naming Conventions:**  Use clear and consistent naming conventions for test files and functions.

### 5.5. Mocking and Stubbing

-   **Mock Objects:** Use mock objects to simulate the behavior of external dependencies.
-   **Stub Functions:** Use stub functions to replace complex or time-consuming operations with simple, predictable results.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Incorrect Input Shapes:**  Ensure that the input shapes match the expected dimensions.
-   **Data Type Mismatches:** Use consistent data types throughout the application.
-   **Gradient Vanishing/Exploding:**  Use appropriate activation functions and weight initialization techniques to prevent gradient problems.
-   **Overfitting:**  Use regularization techniques (e.g., dropout, L1/L2 regularization) to prevent overfitting.

### 6.2. Edge Cases

-   **Empty Datasets:**  Handle empty datasets gracefully.
-   **Missing Values:**  Handle missing values appropriately (e.g., imputation, deletion).
-   **Outliers:**  Identify and handle outliers in the data.

### 6.3. Version-Specific Issues

-   **API Changes:** Be aware of API changes between different versions of Keras and TensorFlow.
-   **Compatibility:**  Ensure that your code is compatible with the versions of Keras and TensorFlow that you are using.

### 6.4. Compatibility Concerns

-   **TensorFlow Compatibility:** Verify the compatibility between Keras and TensorFlow versions. Keras 3 can run on TensorFlow 2.16 onwards but there can be backwards compatibility issues.
-   **Hardware Compatibility:** Ensure compatibility with different hardware platforms (e.g., CPU, GPU, TPU).

### 6.5. Debugging Strategies

-   **Logging:** Use logging to track the execution of the code and identify potential problems.
-   **Debugging Tools:** Use debugging tools (e.g., `pdb`, `TensorBoard`) to inspect the state of the application.
-   **Print Statements:** Use print statements to display intermediate values and debug the code.
-   **TensorBoard:** Use TensorBoard to visualize the model architecture, training progress, and performance metrics.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

-   **IDE:**  Use an IDE (e.g., VS Code, PyCharm) with Keras and TensorFlow support.
-   **Virtual Environment:** Use a virtual environment (e.g., `venv`, `conda`) to isolate project dependencies.
-   **Jupyter Notebook:** Use Jupyter notebooks for experimentation and prototyping.

### 7.2. Build Configuration

-   **Requirements File:**  Use a `requirements.txt` file to specify project dependencies.
-   **Setup Script:**  Use a `setup.py` script to define the project metadata and installation instructions.

### 7.3. Linting and Formatting

-   **PEP 8:** Adhere to PEP 8 style guidelines for Python code.
-   **Linters:** Use linters (e.g., `flake8`, `pylint`) to enforce code style and identify potential problems.
-   **Formatters:** Use formatters (e.g., `black`, `autopep8`) to automatically format code.

### 7.4. Deployment

-   **Containerization:**  Use containerization (e.g., Docker) to package the application and its dependencies.
-   **Cloud Platforms:** Deploy the application to a cloud platform (e.g., AWS, Google Cloud, Azure).
-   **Serving Frameworks:** Use serving frameworks (e.g., TensorFlow Serving, KServe) to deploy models for inference.

### 7.5. CI/CD Integration

-   **Continuous Integration:**  Automate the build, test, and integration process using CI tools (e.g., Jenkins, Travis CI, GitHub Actions).
-   **Continuous Deployment:**  Automate the deployment process using CD tools (e.g., AWS CodePipeline, Google Cloud Build, Azure DevOps).

This comprehensive guide provides a strong foundation for developing high-quality, maintainable, and secure Keras applications. By following these best practices, developers can improve their productivity, reduce the risk of errors, and build robust machine learning systems.