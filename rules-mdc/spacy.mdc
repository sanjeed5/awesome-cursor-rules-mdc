---
description: This rule file provides comprehensive best practices and coding standards for developing projects using spaCy, covering code organization, performance, security, testing, and more. It aims to guide developers in building maintainable, efficient, and secure NLP applications with spaCy.
globs: *.py
---
# spaCy Best Practices and Coding Standards

This document outlines the recommended best practices and coding standards for developing projects using spaCy. Adhering to these guidelines will result in more maintainable, efficient, and secure NLP applications.

## Library Information:
- Name: spaCy
- Tags: python, nlp, text-processing

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

Organizing your spaCy project with a clear directory structure enhances maintainability and collaboration. Here’s a recommended structure:


project_root/
├── data/                  # Contains raw and processed datasets
│   ├── raw/              # Raw data files (e.g., .txt, .csv)
│   ├── processed/        # Processed data (e.g., spaCy Doc objects)
│   └── README.md        # Description of the data and preprocessing steps
├── models/                # Trained spaCy models
│   ├── en_core_web_sm/   # Example model directory
│   └── README.md        # Information about the trained models
├── scripts/               # Python scripts for training, evaluation, and data processing
│   ├── train.py           # Script for training the spaCy model
│   ├── evaluate.py        # Script for evaluating the model
│   ├── preprocess.py      # Script for preprocessing the data
│   └── utils.py           # Utility functions
├── components/            # Custom spaCy components
│   ├── custom_ner.py      # Custom NER component
│   └── custom_tokenizer.py # Custom tokenizer
├── tests/                 # Unit and integration tests
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── conftest.py        # Pytest configuration file
├── notebooks/             # Jupyter notebooks for exploration and prototyping
│   └── data_exploration.ipynb # Example notebook
├── requirements.txt       # Project dependencies
├── pyproject.toml         # Project configuration and build dependencies (recommended over setup.py)
├── README.md            # Project overview and instructions
├── .gitignore             # Specifies intentionally untracked files that Git should ignore
└── .cursor/              # Cursor IDE configuration


### 1.2 File Naming Conventions

Follow these conventions for consistency and readability:

- **Python Files:** `lowercase_with_underscores.py`
- **Data Files:** `lowercase-with-hyphens.csv`
- **Model Files:** `descriptive_name.spacy`
- **Configuration Files:** `config.cfg` or `config.yaml`

### 1.3 Module Organization

Organize your code into modules based on functionality:

- **Data Processing:**  Modules for loading, cleaning, and transforming data.
- **Model Training:** Modules for training spaCy models.
- **Custom Components:** Modules for custom spaCy components (e.g., custom NER, custom tokenizer).
- **Evaluation:** Modules for evaluating model performance.
- **Utilities:** Modules for helper functions and shared logic.

### 1.4 Component Architecture Recommendations

When building custom spaCy components, follow these guidelines:

- **Encapsulation:** Each component should have a well-defined purpose and encapsulate its logic.
- **Modularity:**  Components should be designed to be reusable in different parts of your project.
- **Configuration:** Make components configurable through parameters passed during initialization.
- **Testing:**  Write unit tests for each component to ensure its correctness.

Example of a custom component structure:

python
# components/custom_ner.py
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

@Language.factory(
    "custom_ner",
    default_config={
        "label": "CUSTOM",
        "patterns": []
    },
)
def create_custom_ner(
    nlp: Language,
    name: str,
    label: str,
    patterns: list
):
    return CustomNer(nlp, name, label, patterns)


class CustomNer:
    def __init__(self, nlp: Language, name: str, label: str, patterns: list):
        self.label = label
        self.patterns = patterns
        self.ruler = nlp.add_pipe("entity_ruler")
        self.ruler.add_patterns([{"label": self.label, "pattern": p} for p in self.patterns])


    def __call__(self, doc: Doc) -> Doc:
        #Add the custom patterns to the entity recognizer
        doc.ents = filter_spans(doc.ents)
        return doc


### 1.5 Code Splitting Strategies

For large projects, split your code into smaller, manageable files:

- **Functional Decomposition:** Split code based on functionality (e.g., data loading, preprocessing, model training).
- **Component-Based Splitting:**  Each custom spaCy component should reside in its own file.
- **Layered Architecture:** If applicable, separate the presentation, business logic, and data access layers.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

- **Pipeline Pattern:** Leverage spaCy's pipeline architecture for sequential processing of text.
- **Factory Pattern:** Use factory functions to create spaCy components with specific configurations.
- **Strategy Pattern:** Implement different strategies for text processing (e.g., different tokenization methods) and switch between them based on the input.

### 2.2 Recommended Approaches for Common Tasks

- **Text Preprocessing:** Use spaCy's built-in tokenization, lemmatization, and stop word removal for efficient text preprocessing.
- **Named Entity Recognition:** Utilize spaCy's NER capabilities or train custom NER models for specific entity types.
- **Dependency Parsing:**  Leverage spaCy's dependency parser to extract relationships between words in a sentence.
- **Similarity Analysis:**  Use spaCy's word vectors to compute similarity between documents or phrases.

### 2.3 Anti-patterns and Code Smells

- **Ignoring Errors:** Always handle exceptions and log errors appropriately.
- **Global Variables:**  Avoid using global variables; use dependency injection or configuration files instead.
- **Hardcoding Paths:**  Use relative paths or environment variables for file paths.
- **Overly Complex Functions:**  Keep functions short and focused on a single task.
- **Lack of Documentation:**  Document your code with docstrings and comments.

### 2.4 State Management Best Practices

- **Stateless Components:** Design spaCy components to be stateless whenever possible to avoid unexpected side effects.
- **Configuration Objects:** Use configuration objects to manage component state.
- **Immutability:**  Prefer immutable data structures to prevent unintended modifications.

### 2.5 Error Handling Patterns

- **Try-Except Blocks:** Use try-except blocks to handle potential exceptions during text processing.
- **Logging:**  Log errors and warnings to a file for debugging and monitoring.
- **Custom Exceptions:** Define custom exceptions for specific error conditions.
- **Graceful Degradation:**  Implement fallback mechanisms to handle errors gracefully.

## 3. Performance Considerations

### 3.1 Optimization Techniques

- **Model Selection:** Use smaller spaCy models (e.g., `en_core_web_sm`) for faster processing if accuracy is not critical.
- **Batch Processing:** Process large texts in batches to improve throughput.
- **GPU Acceleration:** Utilize GPU acceleration for faster processing of large datasets (requires `cupy`).
- **Disable Unnecessary Components:** Disable pipeline components that are not needed for a specific task.
- **Cython Optimization:** Write performance-critical code in Cython for faster execution.
- **Vector Data:** Utilize vector data whenever possible for performance as it can be GPU accelerated.

### 3.2 Memory Management

- **Object Reuse:** Reuse spaCy Doc objects to reduce memory allocation overhead.
- **Data Streaming:** Stream large data files instead of loading them into memory at once.
- **Garbage Collection:**  Explicitly trigger garbage collection when memory usage is high (use with caution).

### 3.3 Bundle Size Optimization (If applicable)

- **Tree shaking** If deploying spaCy models to the web or other bundle-size-sensitive environments, use tree shaking to remove unused code from the spaCy library.

### 3.4 Lazy Loading Strategies

- **Model Loading on Demand:** Load spaCy models only when they are needed, rather than at application startup.
- **Component Initialization on Demand:** Initialize custom spaCy components only when they are used.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities

- **Injection Attacks:** Prevent injection attacks by sanitizing user inputs before passing them to spaCy.
- **Denial of Service:** Limit the size of input texts to prevent denial-of-service attacks.
- **Model Poisoning:** Protect against model poisoning by verifying the integrity of trained spaCy models.

### 4.2 Input Validation

- **Text Length Limits:** Limit the length of input texts to prevent excessive processing time and memory usage.
- **Character Encoding:**  Validate that input texts are encoded in a supported character encoding (e.g., UTF-8).
- **Whitelist Validation:**  Use whitelist validation to allow only specific characters or patterns in input texts.

### 4.3 Authentication and Authorization (If applicable)

- **API Keys:** Use API keys to authenticate clients accessing spaCy services.
- **Role-Based Access Control:** Implement role-based access control to restrict access to sensitive spaCy features.

### 4.4 Data Protection

- **Data Encryption:** Encrypt sensitive data at rest and in transit.
- **Data Masking:** Mask or redact sensitive data in log files and debugging output.  Use the retokenizer features to create [REDACTED] names or private identifiable information
- **Anonymization:**  Anonymize data before storing it for analysis or research purposes.

### 4.5 Secure API Communication (If applicable)

- **HTTPS:** Use HTTPS for all API communication to encrypt data in transit.
- **Input Sanitization:**  Sanitize any data received from the user to prevent injections of malicious code.
- **Rate limiting** Implement rate limiting to mitigate denial-of-service (DoS) attacks by restricting the number of requests a user can make within a specific timeframe.

## 5. Testing Approaches

### 5.1 Unit Testing

- **Component Testing:** Write unit tests for each custom spaCy component to ensure its correctness.
- **Function Testing:** Test individual functions in your code to verify their behavior.
- **Mocking:** Use mocking to isolate components and functions during testing.

### 5.2 Integration Testing

- **Pipeline Testing:** Test the integration of multiple spaCy components in a pipeline.
- **Data Flow Testing:** Test the flow of data through your application to ensure that it is processed correctly.
- **API Testing:** Test the integration of your spaCy application with external APIs.

### 5.3 End-to-End Testing

- **User Interface Testing (if applicable):** Test the user interface of your spaCy application to ensure that it is user-friendly and functions as expected.
- **System Testing:** Test the entire system to ensure that all components work together correctly.

### 5.4 Test Organization

- **Separate Test Directory:** Keep tests in a separate `tests/` directory.
- **Test Modules:**  Organize tests into modules based on the functionality being tested.
- **Test Fixtures:**  Use test fixtures to set up test data and dependencies.
- **Coverage Analysis:**  Use coverage analysis to identify untested code.

### 5.5 Mocking and Stubbing Techniques

- **Mocking External Dependencies:** Mock external dependencies (e.g., APIs, databases) to isolate your code during testing.
- **Stubbing Function Calls:** Stub function calls to control the behavior of dependencies.
- **Monkey Patching:** Use monkey patching to replace functions or objects during testing (use with caution).

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

- **Not Handling Exceptions:** Failing to handle exceptions properly can lead to unexpected application crashes.
- **Ignoring Performance:**  Not optimizing your code for performance can result in slow processing times.
- **Lack of Testing:**  Insufficient testing can lead to bugs and regressions.
- **Incorrect model versions:** Using the wrong model version for your spaCy version may cause incompatibility errors.

### 6.2 Edge Cases

- **Unicode Characters:**  Handle Unicode characters correctly to avoid encoding issues.
- **Rare Words:**  Be aware of rare words that may not be in spaCy's vocabulary.
- **Out-of-Vocabulary Words:**  Handle out-of-vocabulary words gracefully.

### 6.3 Version-Specific Issues

- **API Changes:**  Be aware of API changes in spaCy versions and update your code accordingly.
- **Model Compatibility:**  Ensure that trained spaCy models are compatible with the spaCy version you are using.

### 6.4 Compatibility Concerns

- **Dependency Conflicts:**  Be aware of dependency conflicts between spaCy and other libraries.
- **Operating System Compatibility:**  Ensure that your spaCy application is compatible with the target operating systems.

### 6.5 Debugging Strategies

- **Logging:** Use logging to track the execution flow of your code and identify errors.
- **Debugging Tools:**  Use a debugger to step through your code and inspect variables.
- **Print Statements:**  Use print statements to output debugging information.
- **Profiling:**  Use a profiler to identify performance bottlenecks in your code.
- **Error Messages:**  Read and understand error messages to diagnose problems.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

- **IDE:** Visual Studio Code, PyCharm, or other Python IDE.
- **Virtual Environment:** Use virtual environments (e.g., `venv`, `conda`) to isolate project dependencies.
- **Package Manager:** Use `pip` or `conda` to manage project dependencies.

### 7.2 Build Configuration

- **`pyproject.toml`:** Use `pyproject.toml` for build configuration and dependency management (recommended).
- **`setup.py`:** Use `setup.py` for older projects or when `pyproject.toml` is not supported.
- **`requirements.txt`:**  Use `requirements.txt` to specify project dependencies.

### 7.3 Linting and Formatting

- **`flake8`:** Use `flake8` for linting Python code.
- **`pylint`:** Use `pylint` for static code analysis.
- **`black`:** Use `black` for automatic code formatting.
- **`isort`:**  Use `isort` for sorting imports.

### 7.4 Deployment

- **Docker:** Use Docker to containerize your spaCy application.
- **Cloud Platforms:** Deploy your spaCy application to cloud platforms such as AWS, Google Cloud, or Azure.
- **Serverless Functions:** Deploy your spaCy application as serverless functions (e.g., AWS Lambda, Google Cloud Functions).

### 7.5 CI/CD Integration

- **GitHub Actions:** Use GitHub Actions for continuous integration and continuous deployment.
- **Jenkins:** Use Jenkins for continuous integration.
- **Travis CI:** Use Travis CI for continuous integration.
- **Automated Testing:** Automate unit, integration, and end-to-end tests as part of your CI/CD pipeline.
- **Automated Deployment:**  Automate the deployment of your spaCy application to staging and production environments.

By following these best practices, you can build robust, scalable, and maintainable NLP applications with spaCy.