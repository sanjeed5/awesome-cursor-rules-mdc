---
description: This rule outlines the best practices and coding standards for developing with FFmpeg, covering code organization, performance, security, testing, and common pitfalls.
globs: *.c
---
# FFmpeg Best Practices and Coding Standards

This document outlines the best practices and coding standards for developing with FFmpeg. Adhering to these guidelines will promote code quality, maintainability, performance, and security.

## 1. Code Organization and Structure

### 1.1 Directory Structure

The FFmpeg project itself has a specific, well-defined directory structure. While you don't necessarily need to replicate this exactly in your own projects, you can adapt principles from it.

*   `libavcodec/`: Contains all the codecs (encoders and decoders).  Organize your codec-related code in a similar manner if extending FFmpeg.
*   `libavformat/`: Contains the container format handling (muxers and demuxers).
*   `libavfilter/`: Contains the audio and video filters.
*   `libavutil/`: Contains utility functions used throughout the project (e.g., memory management, data structures).
*   `libswscale/`: Contains the video scaling and pixel format conversion functions.
*   `libswresample/`: Contains the audio resampling and format conversion functions.
*   `doc/`: Documentation.
*   `tools/`: Tools like `ffmpeg`, `ffprobe`, and `ffplay`.

For projects *using* FFmpeg, a good directory structure is crucial:


project/
├── src/
│   ├── ffmpeg_integration/
│   │   ├── audio_processing.c
│   │   ├── video_processing.c
│   │   ├── format_handling.c
│   │   └── ...
│   ├── core/
│   │   ├── data_structures.c
│   │   ├── utils.c
│   │   └── ...
│   ├── main.c
│   └── ...
├── include/
│   ├── ffmpeg_integration/
│   │   ├── audio_processing.h
│   │   ├── video_processing.h
│   │   ├── format_handling.h
│   │   └── ...
│   ├── core/
│   │   ├── data_structures.h
│   │   ├── utils.h
│   │   └── ...
│   └── ...
├── tests/
│   ├── unit/
│   │   ├── test_audio_processing.c
│   │   ├── test_video_processing.c
│   │   └── ...
│   ├── integration/
│   │   ├── test_end_to_end.c
│   │   └── ...
│   └── ...
├── build/
├── data/
├── docs/
└── ...


### 1.2 File Naming Conventions

*   Use descriptive file names. For example, `audio_encoder.c`, `video_decoder.c`, `format_muxer.c`.
*   Header files should have the same base name as their corresponding source files (e.g., `audio_encoder.c` and `audio_encoder.h`).
*   Test files can be prefixed with `test_` or postfixed with `_test` (e.g., `test_audio_encoder.c` or `audio_encoder_test.c`).

### 1.3 Module Organization

*   Group related functionalities into modules. Each module should have a clear responsibility (e.g., audio encoding, video decoding, format handling).
*   Use header files to define the public interface of each module.
*   Hide implementation details within the source files.
*   Minimize dependencies between modules.

### 1.4 Component Architecture

*   Adopt a layered architecture. Separate concerns into different layers (e.g., presentation layer, business logic layer, data access layer).
*   Use an abstraction layer to decouple your application from the FFmpeg API. This will make it easier to switch to a different multimedia library in the future.
*   Consider using a plugin architecture if you need to support different codecs or formats.

### 1.5 Code Splitting Strategies

*   Split large source files into smaller, more manageable files.
*   Group related functions and data structures into separate source files.
*   Use header files to define the interface between source files.
*   Consider splitting code based on functionality or layer.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

*   **Factory Pattern:** Use a factory pattern to create instances of codecs or formats based on their names or IDs.
*   **Strategy Pattern:** Use a strategy pattern to implement different encoding or decoding algorithms.
*   **Observer Pattern:**  Use an observer pattern for asynchronous events, like when a frame is decoded or an error occurs.
*   **Resource Acquisition Is Initialization (RAII):** Use RAII to ensure that resources (e.g., memory buffers, codec contexts) are properly released when they are no longer needed.  FFmpeg APIs often require explicit deallocation.

### 2.2 Recommended Approaches for Common Tasks

*   **Encoding:**
    *   Select appropriate codecs based on the desired quality, bitrate, and compatibility.
    *   Configure the encoder parameters properly (e.g., bitrate, frame rate, resolution).
    *   Handle the encoding process asynchronously to avoid blocking the main thread.
*   **Decoding:**
    *   Check for errors after each decoding operation.
    *   Allocate sufficient memory for the decoded frames.
    *   Convert the decoded frames to a suitable pixel format.
*   **Muxing/Demuxing:**
    *   Choose the appropriate container format based on the desired features and compatibility.
    *   Set the stream parameters correctly (e.g., codec, bitrate, frame rate).
    *   Handle metadata properly.
*   **Filtering:**
    *   Construct filter graphs carefully to achieve the desired effects.
    *   Optimize the filter graph for performance.
    *   Handle errors that may occur during filtering.

### 2.3 Anti-patterns and Code Smells

*   **Ignoring Error Codes:** Always check the return values of FFmpeg functions and handle errors appropriately.
*   **Leaking Resources:** Always free allocated memory and close opened files.
*   **Global State:** Avoid using global variables as much as possible.
*   **Magic Numbers:** Use named constants instead of hardcoded numerical values.
*   **Long Functions:** Break down long functions into smaller, more manageable functions.
*   **Unnecessary Transcoding:**  Avoid transcoding between lossy formats if possible. Transcode from lossless to lossy whenever feasible, as mentioned in the provided search results.
*   **Busy-waiting:** Avoid using tight loops that continuously poll for events. Use asynchronous mechanisms instead.

### 2.4 State Management

*   Use structs to encapsulate the state of FFmpeg operations (e.g., encoding, decoding, filtering).
*   Pass the state structs to functions that need to access the state.
*   Avoid using global variables to store state information.
*   Consider using a state machine to manage the different states of FFmpeg operations.

### 2.5 Error Handling

*   Always check the return values of FFmpeg functions.  Most functions return 0 on success and a negative value on failure.
*   Use `av_strerror()` to get a human-readable error message.
*   Log error messages to a file or console.
*   Handle errors gracefully. Don't just crash the application.
*   Use exception handling (if appropriate for the language) to catch unexpected errors.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Codec Selection:** Choose the most efficient codecs for your needs (e.g., use hardware acceleration if available).
*   **Bitrate Control:** Use appropriate bitrate control methods (e.g., constant bitrate (CBR), variable bitrate (VBR)).
*   **Multi-threading:** Use multi-threading to parallelize encoding, decoding, and filtering operations.
*   **SIMD Instructions:** Utilize SIMD (Single Instruction, Multiple Data) instructions (e.g., SSE, AVX) to accelerate processing.
*   **Caching:** Cache frequently accessed data to reduce memory access overhead.
*   **Reduce Memory Copies:** Minimize unnecessary memory copies.
*   **Hardware Acceleration:** Utilize hardware acceleration (e.g., VAAPI, NVENC, QSV) for encoding and decoding.  As the FFmpeg documentation and release notes indicate, there is constant work on expanding HW acceleration capabilities for various codecs.
*   **Filter Graph Optimization:** Optimize your filter graphs to reduce the number of operations performed.

### 3.2 Memory Management

*   Use `av_malloc()` and `av_free()` to allocate and free memory.
*   Always free allocated memory when it is no longer needed.
*   Avoid memory leaks.
*   Use memory pools to reduce memory allocation overhead.
*   Be aware of potential buffer overflows.

### 3.3 Rendering Optimization (If Applicable)

*   Use hardware acceleration for rendering (e.g., OpenGL, Direct3D).
*   Optimize your rendering pipeline to reduce the number of draw calls.
*   Use efficient data structures for storing video frames.

### 3.4 Bundle Size Optimization (If Applicable)

*   Only include the codecs and formats that you need in your application.
*   Use a linker to remove unused code.
*   Compress your application binary.

### 3.5 Lazy Loading (If Applicable)

*   Load codecs and formats on demand.
*   Use asynchronous loading to avoid blocking the main thread.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and Prevention

*   **Buffer Overflows:** Ensure that input buffers are large enough to hold the data being read into them. Use functions like `av_strlcpy()` or `av_strlcat()` to prevent overflows.
*   **Integer Overflows:** Check for potential integer overflows before performing arithmetic operations.
*   **Format String Vulnerabilities:** Avoid using user-supplied input directly in format strings.
*   **Denial of Service (DoS):** Limit the resources that can be consumed by FFmpeg operations (e.g., maximum bitrate, maximum frame size).
*   **Arbitrary Code Execution:** Be extremely careful when using user-supplied plugins or filters. Validate the code before executing it.
*   **Input Validation flaws:** As the Android Stagefright bug illustrated (mentioned in search results), carefully validate sizes and other parameters in subtitle parsing.

### 4.2 Input Validation

*   Validate all input parameters before passing them to FFmpeg functions.
*   Check the range of numerical values.
*   Sanitize strings to prevent format string vulnerabilities.
*   Use whitelists to restrict the allowed values for certain parameters.

### 4.3 Authentication and Authorization (If Applicable)

*   Use strong authentication mechanisms to protect your application from unauthorized access.
*   Implement proper authorization policies to restrict access to sensitive resources.
*   Use secure communication protocols (e.g., HTTPS) to protect data in transit.

### 4.4 Data Protection

*   Encrypt sensitive data at rest and in transit.
*   Use secure storage mechanisms to protect your data from unauthorized access.
*   Implement proper access control policies to restrict access to data.
*   Consider watermarking video and audio content to deter piracy.

### 4.5 Secure API Communication

*   Use secure protocols (e.g., HTTPS) to communicate with FFmpeg APIs.
*   Validate all data received from FFmpeg APIs.
*   Implement proper error handling to detect and respond to security threats.

## 5. Testing Approaches

### 5.1 Unit Testing

*   Test individual components in isolation.
*   Use mocking and stubbing to isolate the components being tested.
*   Write unit tests for all critical functionalities.
*   Use a test framework (e.g., Check, CUnit) to automate the testing process.
*   Aim for high code coverage.

### 5.2 Integration Testing

*   Test the interaction between different components.
*   Simulate real-world scenarios.
*   Use a test environment that is as close as possible to the production environment.

### 5.3 End-to-End Testing

*   Test the entire application from start to finish.
*   Verify that all components are working correctly together.
*   Use automated testing tools to automate the testing process.

### 5.4 Test Organization

*   Organize your tests into separate directories based on the component being tested.
*   Use descriptive names for test files and test functions.
*   Group related tests into test suites.

### 5.5 Mocking and Stubbing

*   Use mocking and stubbing to isolate the components being tested.
*   Create mock objects that mimic the behavior of real objects.
*   Use stub functions to replace the implementation of certain functions.
*   Use a mocking framework (e.g., CMock) to simplify the mocking process.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

*   Forgetting to initialize FFmpeg libraries (e.g., calling `avformat_network_init()` when using network protocols).
*   Using deprecated APIs.
*   Not handling codec context correctly.
*   Misunderstanding time base conversions.
*   Incorrectly setting pixel formats.
*   Not properly flushing encoders at the end of encoding.

### 6.2 Edge Cases

*   Handling corrupted or incomplete input files.
*   Dealing with variable frame rate videos.
*   Supporting different audio and video codecs.
*   Handling metadata correctly.
*   Properly managing threading and synchronization when using multiple threads.

### 6.3 Version-Specific Issues

*   Be aware of API changes between different FFmpeg versions.  Pay close attention to the `APIchanges` files released with each version (as seen in the provided search results).
*   Use conditional compilation to support different versions of FFmpeg.
*   Consult the FFmpeg documentation for the specific version you are using.

### 6.4 Compatibility Concerns

*   Ensure that your application is compatible with different operating systems and architectures.
*   Test your application on different devices.
*   Use conditional compilation to support different platforms.
*   Consider using a cross-platform build system (e.g., CMake).

### 6.5 Debugging

*   Use a debugger (e.g., GDB) to step through the code and inspect variables.
*   Use logging to track the execution of your application.
*   Use a memory checker (e.g., Valgrind) to detect memory leaks and other memory errors.
*   Use FFmpeg's built-in debugging tools (e.g., `-report` option).
*   Check the FFmpeg mailing lists and forums for known issues and solutions.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **IDE:** Visual Studio, CLion, Eclipse.
*   **Compiler:** GCC, Clang, MSVC.
*   **Debugger:** GDB, LLDB.
*   **Memory Checker:** Valgrind.
*   **Build System:** CMake, Make.
*   **Linting/Formatting:**  clang-format, cppcheck.

### 7.2 Build Configuration

*   Use a build system (e.g., CMake, Make) to automate the build process.
*   Configure the build system to include the necessary FFmpeg libraries and header files.
*   Use compiler flags to optimize the code for performance.
*   Create separate build configurations for debug and release builds.

### 7.3 Linting and Formatting

*   Use a linter (e.g., cppcheck) to check for coding errors and style violations.
*   Use a code formatter (e.g., clang-format) to automatically format the code according to a consistent style.
*   Enforce coding standards through automated checks during the build process.
*   Follow the FFmpeg coding style guidelines (K&R style, 4-space indentation, as mentioned in the search results).

### 7.4 Deployment

*   Package your application and its dependencies into a single distribution package.
*   Use a package manager (e.g., apt, yum, Homebrew) to install your application.
*   Consider using a containerization technology (e.g., Docker) to deploy your application.

### 7.5 CI/CD Integration

*   Integrate your tests into a Continuous Integration/Continuous Deployment (CI/CD) pipeline.
*   Run the tests automatically whenever code is committed to the repository.
*   Use a CI/CD tool (e.g., Jenkins, Travis CI, GitHub Actions) to automate the CI/CD process.
*   Automate the deployment process.