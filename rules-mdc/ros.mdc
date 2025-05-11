---
description: This rule provides comprehensive best practices and coding standards for ROS (Robot Operating System) development, covering code organization, common patterns, performance, security, testing, and tooling. It aims to enhance code quality, maintainability, and interoperability in ROS projects.
globs: *.cpp,*.h,*.py,*.launch,*.msg,*.srv,*.action,*.yaml
---
# ROS Best Practices and Coding Standards

This document outlines best practices and coding standards for developing robust, maintainable, and interoperable ROS (Robot Operating System) code. It covers various aspects of ROS development, from code organization to security considerations, based on community best practices and official guidelines.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

A standard ROS package directory structure helps maintainability and discoverability. Here's a recommended structure:


my_package/
├── CMakeLists.txt          # CMake build file
├── package.xml             # Package manifest
├── include/                # Header files (C++)
│   └── my_package/
│       └── my_node.h
├── src/                    # Source code (C++ and Python)
│   ├── my_node.cpp
│   └── my_python_node.py
├── scripts/                # Executable scripts (e.g., Python)
│   └── my_script.py
├── msg/                    # Message definitions (.msg files)
│   └── MyMessage.msg
├── srv/                    # Service definitions (.srv files)
│   └── MyService.srv
├── action/                 # Action definitions (.action files)
│   └── MyAction.action
├── launch/                 # Launch files (.launch or .launch.py)
│   └── my_launch.launch.py
├── config/                 # Configuration files (YAML)
│   └── my_config.yaml
├── urdf/                   # URDF files for robot models (.urdf or .xacro)
│   └── my_robot.urdf.xacro
├── rviz/                   # RViz configuration files (.rviz)
│   └── my_rviz_config.rviz
├── test/                    # Test files
│   ├── my_node_test.cpp
│   └── my_python_node_test.py
└── doc/                     # Documentation
    └── README.md


### 1.2 File Naming Conventions

-   **Packages:** Use underscore-separated lowercase names (e.g., `my_package`).
-   **Source Files:** Use underscore-separated lowercase names with `.cpp` or `.py` extension (e.g., `my_node.cpp`, `my_python_node.py`).
-   **Header Files:** Use underscore-separated lowercase names with `.h` extension, typically matching the class name (e.g., `my_class.h`).
-   **Message/Service/Action Definitions:** Use CamelCase for names (e.g., `MyMessage.msg`, `MyService.srv`, `MyAction.action`).
-   **Launch Files:** Use underscore-separated lowercase names with `.launch.py` or `.launch` extension (e.g., `my_launch.launch.py`).
-   **Configuration Files:** Use underscore-separated lowercase names with `.yaml` extension (e.g., `my_config.yaml`).

### 1.3 Module Organization

-   **Logical Grouping:** Group related functionalities into separate modules or classes.
-   **Namespaces (C++):** Use namespaces to avoid naming conflicts and organize code (e.g., `namespace my_package`).
-   **Python Modules:** Organize code into Python modules within the `src` directory.

### 1.4 Component Architecture

-   **Node Design:** Design nodes to be modular and focused on specific tasks. Avoid monolithic nodes.
-   **Communication:** Use ROS topics, services, and actions appropriately for communication between nodes. Consider using ROS 2 actions over services when a task could take a long time to perform. Using actionlib, nodes can execute goals asynchronously, provide feedback, and be cancelled.
-   **Composition:** Use ROS 2 component composition to combine multiple nodes into a single process for improved performance.

### 1.5 Code Splitting

-   **Libraries:** Extract reusable code into libraries within the package (e.g., in a `lib` subdirectory).
-   **Configuration Files:** Use configuration files (YAML) to parameterize node behavior without modifying code.
-   **Launch Files:** Separate node execution and configuration into launch files for easy deployment and modification.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

-   **Publisher-Subscriber:** Use topics for asynchronous, one-to-many communication.
-   **Request-Reply:** Use services for synchronous, one-to-one communication requiring a response.
-   **Actionlib:** Use actions for long-running tasks with feedback and cancellation capabilities.
-   **State Machine:** Implement state machines for complex node behavior.
-   **Parameterization:** Externalize configurable parameters to YAML files loaded at runtime.
-   **Lifecycle Management:** Implement node lifecycle management for controlled startup and shutdown (ROS 2).
-   **Sensor Drivers:** Create dedicated nodes that abstract the hardware and provide a unified interface to the sensor.  This also provides a single point of failure and restart for sensor data.

### 2.2 Recommended Approaches

-   **Configuration:** Prefer using ROS parameters and dynamic reconfigure for node configuration.
-   **Logging:** Use `ROS_INFO`, `ROS_WARN`, `ROS_ERROR`, and `ROS_DEBUG` macros (C++) or `rospy.loginfo`, `rospy.logwarn`, `rospy.logerr`, and `rospy.logdebug` (Python) for logging messages.
-   **Error Handling:** Use exceptions (C++) or handle errors gracefully in Python with `try...except` blocks.
-   **Time Handling:** Use `ros::Time::now()` (C++) or `rospy.Time.now()` (Python) for getting the current ROS time.
-   **Transformations:** Use the `tf2` library for managing coordinate transformations.
-   **Unit Testing:** Create comprehensive unit tests for individual components.

### 2.3 Anti-patterns and Code Smells

-   **God Nodes:** Avoid creating nodes that perform too many unrelated tasks.
-   **Hardcoded Values:** Avoid hardcoding values in the code; use parameters instead.
-   **Spinning Too Much:** Avoid busy-waiting loops; use `ros::Rate` or `rospy.Rate` to control loop frequency.
-   **Ignoring Errors:** Handle potential errors and exceptions properly.
-   **Global Variables:** Avoid using global variables; use class members or local variables instead.
-   **Magic Numbers:** Define constants for important values instead of using magic numbers.

### 2.4 State Management

-   **Internal State:** Encapsulate node state within class members or local variables.
-   **External State:** Use ROS parameters to store and manage persistent state.
-   **State Machines:** Employ state machines for complex state transitions and behavior.

### 2.5 Error Handling

-   **Exceptions (C++):** Use exceptions to signal errors and handle them appropriately.
-   **Try...Except (Python):** Use `try...except` blocks to catch and handle exceptions.
-   **Return Codes:** In cases where exceptions are not appropriate, use return codes to indicate success or failure.
-   **Logging:** Log error messages using `ROS_ERROR` or `rospy.logerr`.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Minimize Data Copying:** Use references or pointers to avoid unnecessary data copying.
-   **Efficient Data Structures:** Use appropriate data structures (e.g., `std::vector`, `std::map`) for efficient data storage and retrieval.
-   **Pre-allocation:** Pre-allocate memory for frequently used data structures.
-   **Node Composition (ROS 2):** Compose multiple nodes into a single process for improved inter-node communication.
-   **ZeroMQ (ROS 2):** Use the ZeroMQ transport for low-latency communication.

### 3.2 Memory Management

-   **Smart Pointers (C++):** Use smart pointers (`std::shared_ptr`, `std::unique_ptr`) to manage memory automatically and prevent memory leaks.
-   **Resource Acquisition Is Initialization (RAII):** Use RAII to ensure that resources are released when they are no longer needed.
-   **Memory Pools:** Use memory pools for allocating and deallocating memory efficiently.

### 3.3 Rendering Optimization (If Applicable)

-   **Minimize Polygons:** Use fewer polygons in 3D models.
-   **Texture Optimization:** Optimize textures for size and resolution.
-   **Culling:** Implement view frustum culling to avoid rendering objects that are not visible.

### 3.4 Bundle Size Optimization

-   **Minimal Dependencies:** Only include the necessary dependencies in the package manifest.
-   **Code Stripping:** Strip unnecessary symbols and debug information from binaries.

### 3.5 Lazy Loading

-   **On-Demand Loading:** Load resources (e.g., 3D models, textures) only when they are needed.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities

-   **Unvalidated Input:** Protect against injection attacks by validating all input data.
-   **Buffer Overflows:** Prevent buffer overflows by using safe string manipulation functions.
-   **Denial of Service (DoS):** Protect against DoS attacks by limiting resource consumption.
-   **Man-in-the-Middle (MitM):** Use secure communication protocols (e.g., TLS/SSL) to prevent MitM attacks.
-   **Remote Code Execution (RCE):** Avoid RCE vulnerabilities by carefully sanitizing user input and preventing the execution of arbitrary code.

### 4.2 Input Validation

-   **Data Type Validation:** Ensure that input data matches the expected data type.
-   **Range Checking:** Verify that input values are within the allowed range.
-   **String Sanitization:** Sanitize string input to prevent injection attacks.

### 4.3 Authentication and Authorization

-   **ROS 2 Security Features:** Utilize the built-in security features of ROS 2, such as authentication and access control.
-   **Secure Communication:** Use secure communication protocols (e.g., TLS/SSL) for sensitive data.
-   **Access Control:** Implement access control mechanisms to restrict access to sensitive resources.

### 4.4 Data Protection

-   **Encryption:** Encrypt sensitive data at rest and in transit.
-   **Data Masking:** Mask sensitive data to protect it from unauthorized access.
-   **Data Auditing:** Audit data access and modifications to detect potential security breaches.

### 4.5 Secure API Communication

-   **API Keys:** Use API keys for authenticating API requests.
-   **Rate Limiting:** Implement rate limiting to prevent abuse of APIs.
-   **Input Validation:** Validate all input data to prevent injection attacks.

## 5. Testing Approaches

### 5.1 Unit Testing

-   **Individual Components:** Test individual components (e.g., classes, functions) in isolation.
-   **Test Frameworks:** Use testing frameworks like Google Test (C++) or `unittest` (Python).
-   **Test Coverage:** Aim for high test coverage to ensure that all code paths are tested.

### 5.2 Integration Testing

-   **Multiple Components:** Test the interaction between multiple components or nodes.
-   **ROS Integration Tests:** Use ROS-specific testing tools (e.g., `rostest`) to test ROS nodes and their communication.

### 5.3 End-to-End Testing

-   **Complete System:** Test the entire ROS system from end to end.
-   **Simulation:** Use simulation environments (e.g., Gazebo) for testing in a realistic setting.

### 5.4 Test Organization

-   **Test Directory:** Create a dedicated `test` directory in the package.
-   **Test Naming:** Use descriptive names for test files and functions.
-   **Test Suites:** Group related tests into test suites.

### 5.5 Mocking and Stubbing

-   **Mock Objects:** Use mock objects to simulate the behavior of dependencies.
-   **Stub Functions:** Use stub functions to replace complex or external functions with simplified versions.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

-   **Incorrect Topic Names:** Double-check topic names for typos or inconsistencies.
-   **Missing Dependencies:** Ensure that all dependencies are declared in the package manifest.
-   **Incorrect Time Handling:** Be aware of the difference between real-world time and ROS time.
-   **Ignoring TF2:** Don't forget to use TF2 for managing coordinate transformations.

### 6.2 Edge Cases

-   **Network Latency:** Consider the impact of network latency on ROS communication.
-   **Message Loss:** Handle potential message loss in unreliable networks.
-   **Concurrency Issues:** Be aware of concurrency issues when using multiple threads.

### 6.3 Version-Specific Issues

-   **ROS 1 vs. ROS 2:** Be aware of the differences between ROS 1 and ROS 2.
-   **Package Compatibility:** Ensure that packages are compatible with the ROS distribution being used.

### 6.4 Compatibility Concerns

-   **Operating Systems:** Test ROS code on different operating systems (e.g., Linux, macOS, Windows).
-   **Hardware Platforms:** Ensure that ROS code is compatible with different hardware platforms.

### 6.5 Debugging Strategies

-   **ROS Logging:** Use `ROS_INFO`, `ROS_WARN`, `ROS_ERROR`, and `ROS_DEBUG` (C++) or `rospy.loginfo`, `rospy.logwarn`, `rospy.logerr`, and `rospy.logdebug` (Python) to log messages for debugging.
-   **RViz:** Use RViz for visualizing ROS data and debugging visual issues.
-   **Rosbag:** Use Rosbag for recording and playing back ROS messages.
-   **GDB (C++):** Use GDB for debugging C++ code.
-   **pdb (Python):** Use pdb for debugging Python code.

## 7. Tooling and Environment

### 7.1 Recommended Tools

-   **IDE:** VS Code with ROS extension, Eclipse, or other IDEs.
-   **Build System:** CMake (ROS 1 and ROS 2).
-   **Package Manager:** `apt` (Ubuntu), `pip` (Python).
-   **Version Control:** Git.
-   **Containerization:** Docker.

### 7.2 Build Configuration

-   **CMakeLists.txt:** Configure the build process using `CMakeLists.txt`.
-   **package.xml:** Declare package dependencies and metadata in `package.xml`.

### 7.3 Linting and Formatting

-   **C++:** Use `clang-format` for code formatting and `cpplint` for code style checking.
-   **Python:** Use `flake8` for code style checking and `black` for code formatting.
-   **ROS Lint:** Use `roslint` for checking ROS-specific code style issues.

### 7.4 Deployment

-   **Debian Packages:** Create Debian packages for easy deployment on Ubuntu systems.
-   **Docker Containers:** Package ROS applications into Docker containers for portability.

### 7.5 CI/CD Integration

-   **GitHub Actions:** Use GitHub Actions for continuous integration and continuous deployment.
-   **Jenkins:** Use Jenkins for automated testing and deployment.
-   **ROS Buildfarm:** Leverage the ROS buildfarm for automated builds and testing.

This document provides a comprehensive guide to ROS best practices. Adhering to these guidelines will improve the quality, maintainability, and interoperability of ROS projects.