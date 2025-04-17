---
description: This rule enforces LLVM's coding standards and promotes best practices for writing efficient, maintainable, and robust code within the LLVM ecosystem. It covers style, language usage, optimization strategies, and more.
globs: *.h,*.hpp,*.c,*.cc,*.cpp,*.ll,*.mlir
---
- Adhere to LLVM's coding standards to ensure code consistency, readability, and maintainability.
- Always follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) and [LLVM Developer Policy](https://llvm.org/docs/DeveloperPolicy.html).

## Languages, Libraries, and Standards

- Use modern, standard-conforming, and portable C++ code (C++17 or later, as supported by major toolchains).
- Prefer LLVM's own libraries (e.g., `llvm::DenseMap`, `llvm::SmallVector`) over the C++ standard library when appropriate for performance or integration benefits.  Consult the LLVM Programmer's Manual for details.
- Use Python (with the minimum version specified in the Getting Started documentation) for automation, build systems, and utility scripts.
- Format Python code with `black` and `darker` to adhere to PEP 8 standards.

## Mechanical Source Issues

- Write clear diagnostic messages using succinct and correct English.
- Follow the recommended `#include` style: Main module header first, followed by LLVM project headers (most specific to least specific), then system headers.
- Limit source code width to 80 columns.
- Prefer spaces to tabs in source files.
- Avoid trailing whitespace.
- Format multi-line lambdas like blocks of code.
- Use braced initializer lists as if the braces were parentheses in a function call.

## Language and Compiler Issues

- Treat compiler warnings as errors.
- Write portable code, encapsulating non-portable code behind well-defined interfaces.
- Do not use RTTI or exceptions.
- Prefer C++-style casts (`static_cast`, `reinterpret_cast`, `const_cast`) over C-style casts, except when casting to `void` to suppress unused variable warnings or between integral types.
- Do not use static constructors (global variables with constructors or destructors).
- Use `struct` when all members are public, and `class` otherwise.
- Avoid using braced initializer lists to call constructors with non-trivial logic.
- Use `auto` type deduction to make code more readable, but be mindful of unnecessary copies.
- Beware of non-determinism due to the ordering of pointers in unordered containers.
- Beware of non-deterministic sorting order of equal elements; use `llvm::sort` instead of `std::sort`.

## Style Issues

- Layer libraries correctly to avoid circular dependencies. Use Unix linker principles to enforce this.
- `#include` as little as possible, especially in header files. Use forward declarations when possible.
- Use namespace qualifiers to implement previously declared functions in source files; avoid opening namespace blocks.
- Use early exits and `continue` statements to simplify code and reduce indentation.
- Don't use `else` or `else if` after control flow interrupting statements like `return`, `break`, `continue`, or `goto`.
- Turn predicate loops into predicate functions.

## The Low-Level Issues

- Name types, functions, variables, and enumerators properly, using descriptive and consistent names.
- Assert liberally to check preconditions and assumptions.  Include informative error messages in assertions. Use `llvm_unreachable` to mark code that should never be reached. Avoid `assert(false)`. When assertions produce “unused value” warnings, move the call into the assert or cast the value to void.
- Do not use `using namespace std`. Explicitly qualify identifiers from the standard namespace.
- Don't use default labels in fully covered switches over enumerations to catch new enumeration values. Use `llvm_unreachable` after the switch to suppress GCC warnings about reaching the end of a non-void function.
- Use range-based for loops wherever possible.
- Don’t evaluate `end()` every time through a loop; evaluate it once before the loop starts.
- `#include <iostream>` is forbidden. Use `raw_ostream` instead.
- Avoid `std::endl`; use `\n` instead.
- Don’t use `inline` when defining a function in a class definition; it's implicit.

## Microscopic Details

- Put a space before an open parenthesis only in control flow statements, but not in normal function call expressions and function-like macros.
- Prefer preincrement (`++X`) over postincrement (`X++`).
- Don't indent namespaces; add comments indicating which namespace is being closed when helpful.

## Code Organization and Structure

- **Directory Structure:** Follow the existing LLVM directory structure conventions.  New components should typically reside in a subdirectory within an existing high-level category (e.g., `lib/Transforms`, `include/llvm/Analysis`).  Consider the intended audience (end-user tool vs. internal library) when choosing a location.
- **File Naming:** Use descriptive names for files, matching the class or functionality they contain.  Header files should have a `.h` or `.hpp` extension, and implementation files should have a `.cpp`, `.cc`, or `.c` extension (depending on whether they are C or C++). MLIR files should have `.mlir` extension.
- **Module Organization:** Break down large components into smaller, more manageable modules. Each module should have a well-defined purpose and interface. Minimize dependencies between modules.
- **Component Architecture:** Design components with clear separation of concerns.  Use interfaces and abstract classes to promote loose coupling and enable polymorphism.  Follow the principle of least privilege.
- **Code Splitting:** Decompose large functions into smaller, well-named helper functions. Use namespaces to group related functions and classes.

## Common Patterns and Anti-patterns

- **Visitor Pattern:** LLVM frequently uses the Visitor pattern for traversing and manipulating data structures (e.g., the LLVM IR).  Implement visitor classes for custom analysis or transformations.
- **Pass Infrastructure:** Leverage LLVM's pass infrastructure for implementing optimization and analysis passes.  Create new pass types (e.g., analysis pass, transformation pass) as needed.
- **Factory Pattern:** Use the Factory pattern to create instances of classes based on runtime parameters or configuration settings.
- **Singleton Pattern (Avoid):** Minimize the use of the Singleton pattern, as it can lead to tight coupling and make testing difficult.  Consider dependency injection instead.
- **Global Variables (Avoid):** Avoid global variables whenever possible, as they can introduce unexpected side effects and make code harder to reason about. Use configuration objects or dependency injection instead.

## Performance Considerations

- **Inline Hints:** Use `inline` judiciously to guide the compiler to inline frequently called functions.  Avoid inlining large or complex functions.
- **Profile-Guided Optimization (PGO):** Use PGO to optimize code based on runtime profiles.  This can significantly improve performance for frequently executed code paths.
- **LTO (Link-Time Optimization):** Enable LTO to allow the compiler to optimize code across module boundaries.
- **Memory Management:** Use LLVM's memory management facilities (e.g., `BumpPtrAllocator`, `ScopedHashTable`) for efficient memory allocation.  Avoid excessive memory allocation and deallocation.
- **Avoid Unnecessary Copies:** Pass objects by reference or pointer to avoid unnecessary copying.
- **Use `SmallVector`:** Use `llvm::SmallVector` for small, fixed-size vectors to avoid dynamic memory allocation.
- **Cache Locality:** Consider cache locality when designing data structures and algorithms.

## Security Best Practices

- **Input Validation:** Validate all external inputs to prevent vulnerabilities such as buffer overflows and code injection. Use LLVM's API for parsing various file formats.
- **Sanitizers:** Use AddressSanitizer (ASan), MemorySanitizer (MSan), and UndefinedBehaviorSanitizer (UBSan) during development and testing to detect memory errors and undefined behavior.
- **Avoid Code Injection:** Do not construct code by concatenating strings or using other techniques that could lead to code injection vulnerabilities.
- **Data Validation:** Make sure to validate any data being retrieved from memory or other storage locations. Use `llvm::isSafeToLoadUnconditionally` to check if data is safe to load.

## Testing Approaches

- **Unit Tests:** Write unit tests for individual components and functions. Use the LLVM testing framework (e.g., `lit`) to automate test execution.
- **Integration Tests:** Write integration tests to verify the interaction between different components. Create realistic test cases that exercise common use scenarios.
- **Regression Tests:** Add regression tests for any bug fixes to prevent regressions in future versions.
- **Fuzzing:** Use fuzzing techniques to identify corner cases and potential vulnerabilities.  Integrate fuzzing into the CI/CD pipeline.
- **Code Coverage:** Measure code coverage to ensure that tests exercise all important code paths. Use `llvm-cov` to generate code coverage reports.

## Common Pitfalls and Gotchas

- **Incorrect Use of APIs:** Carefully read the documentation for LLVM APIs and ensure they are used correctly.  Pay attention to preconditions and postconditions.
- **Memory Leaks:** Ensure that all allocated memory is properly deallocated. Use memory leak detection tools to identify memory leaks.
- **Dangling Pointers:** Avoid dangling pointers by ensuring that objects are not deleted while they are still being referenced.
- **Undefined Behavior:** Avoid undefined behavior, as it can lead to unpredictable results. Use sanitizers to detect undefined behavior.
- **Version Compatibility:** Be aware of version compatibility issues when using LLVM with other libraries or tools.  Test your code with different versions of LLVM.

## Tooling and Environment

- **Clang:** Use Clang as the primary compiler for LLVM projects. Clang provides excellent support for C++ standards and LLVM extensions.
- **CMake:** Use CMake to manage the build process for LLVM projects.  CMake is a cross-platform build system generator that is well-supported by LLVM.
- **LLVM Tools:** Utilize the LLVM tools (e.g., `llvm-as`, `llvm-dis`, `opt`, `lli`) for various tasks such as assembling and disassembling LLVM IR, optimizing code, and executing LLVM bitcode.
- **Git:** Use Git for version control.  Follow the LLVM Git commit message conventions.
- **CI/CD:** Integrate CI/CD pipelines into LLVM development. Use test suites as part of your development cycle.

## Additional Notes

- When extending or modifying existing code, adhere to the style already in use.
- Document your code clearly and concisely.  Provide comments explaining the purpose of functions, classes, and variables.
- Keep pull requests small and focused.  This makes it easier for reviewers to understand and approve your changes.
- Be responsive to feedback from reviewers and address any issues promptly.
- Engage with the LLVM community to ask questions, share knowledge, and contribute to the project.
- Refer to existing documentation and examples in the LLVM codebase for guidance on how to implement new features and functionality.

By following these guidelines, you can write high-quality LLVM code that is efficient, maintainable, and robust. This ensures consistent code quality and facilitates collaboration within the LLVM community.