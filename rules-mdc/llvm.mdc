---
description: This guide provides definitive, actionable best practices for developing with LLVM, focusing on code quality, performance, and robust testing using modern tools like LIT and TableGen.
globs: **/*
---
# llvm Best Practices

Developing for LLVM requires adherence to specific patterns and standards to maintain consistency, performance, and testability across a massive codebase. This guide outlines critical best practices for our team.

## 1. Code Organization and Structure

### 1.1 Adhere to LLVM Coding Standards
Consistency is paramount. Follow LLVM's established naming conventions, whitespace rules, and C++17 usage. Deviations require strong justification.

*   **Naming**: `CamelCase` for types, `lower_case` for functions/variables.
*   **Whitespace**: Use `clang-format` with LLVM style.
*   **C++ Standard**: Stick to C++17 features supported by major toolchains.

❌ BAD: Inconsistent naming and formatting
```cpp
class my_custom_pass { // Class name should be CamelCase
  void RunPass() { // Function name should be lower_case
    int temp_var = 0;
  }
};
```

✅ GOOD: Consistent LLVM style
```cpp
class MyCustomPass {
  void runOnFunction(llvm::Function &F) { // LLVM passes typically run on a Function
    int tempVar = 0; // Local variables can be camelCase or snake_case, consistency is key.
  }
};
```

### 1.2 Prefer LLVM ADT and Utilities
Leverage LLVM's specialized data structures and utilities over standard library equivalents for performance and consistency.

*   **Containers**: Use `llvm::SmallVector`, `llvm::DenseMap`, `llvm::StringRef` etc.
*   **I/O**: Use `llvm::raw_ostream` for all output.

❌ BAD: Using standard library containers and I/O
```cpp
#include <vector>
#include <map>
#include <iostream>

std::vector<int> myVec;
std::map<std::string, int> myMap;
std::cout << "Debug info: " << val << std::endl;
```

✅ GOOD: Using LLVM ADT and `raw_ostream`
```cpp
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

llvm::SmallVector<int, 8> myVec; // SmallVector avoids heap allocation for small sizes
llvm::DenseMap<llvm::StringRef, int> myMap; // DenseMap is faster for many use cases
llvm::outs() << "Debug info: " << val << "\n"; // Use llvm::outs() for stdout
```

### 1.3 Utilize TableGen for Declarative Definitions
For instruction definitions, target descriptions, or pass registration, use TableGen. It reduces boilerplate and ensures consistency.

```td
// In a .td file
def MyInst : Instruction<"my_inst", ...>;
```

## 2. Common Patterns and Anti-patterns

### 2.1 Use LLVM_DEBUG for Debugging
Never use `printf` or `std::cout` for debug output. Use `LLVM_DEBUG` macros, which are compiled out in release builds.

❌ BAD: Debugging with `printf` or `std::cerr`
```cpp
#include <iostream>
void MyPass::runOnFunction(llvm::Function &F) {
  std::cerr << "Processing function: " << F.getName().str() << "\n";
  // ...
}
```

✅ GOOD: Debugging with `LLVM_DEBUG`
```cpp
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "my-pass" // Define a debug type for your pass

void MyPass::runOnFunction(llvm::Function &F) {
  LLVM_DEBUG(llvm::dbgs() << "Processing function: " << F.getName() << "\n");
  // ...
}
```

### 2.2 Assertions are Your Friend
Use `assert` liberally to catch invariants and impossible states early. Always build with assertions enabled during development (`-DLLVM_ENABLE_ASSERTIONS=On`).

❌ BAD: Relying on runtime crashes for logic errors
```cpp
if (!Val->hasOneUse()) {
  // This should never happen, but no assertion.
}
```

✅ GOOD: Asserting expected conditions
```cpp
assert(Val->hasOneUse() && "Expected value to have a single use!");
```

## 3. Performance Considerations

### 3.1 Optimize Data Structure Choices
`llvm::SmallVector` and `llvm::DenseMap` are specifically designed for compiler performance. Understand their tradeoffs and use them appropriately.

*   `SmallVector`: Avoids heap allocations for small collections.
*   `DenseMap`/`DenseSet`: Faster than `std::map`/`std::set` for many common cases due to hash-based implementation.

### 3.2 Incremental Builds
Use CMake effectively for incremental builds. Avoid full rebuilds.

```bash
# Configure for debug with assertions
cmake -G Ninja -DLLVM_ENABLE_ASSERTIONS=On -DCMAKE_BUILD_TYPE=Debug ../llvm-project/llvm
# Build only the necessary targets
ninja MyPassName
```

## 4. Common Pitfalls and Gotchas

### 4.1 Forgetting Pre-Commit Tests
Always add a baseline test *before* your change, then a second commit with the actual change and test diffs. This is crucial for review and regression safety.

### 4.2 Manual `CHECK` Line Editing
Never manually edit `CHECK` lines in LIT tests. Use `llvm/utils/update_test_checks.py`.

❌ BAD: Manually updating `CHECK` lines
```llvm
; CHECK: define i32 @foo() {
; CHECK-NEXT: ret i32 42
; Manually updated this
```

✅ GOOD: Using the script
```bash
llvm/utils/update_test_checks.py --opt-binary build/bin/opt llvm/test/Transforms/InstCombine/my_test.ll
```

## 5. Testing Approaches

### 5.1 Embrace the Test-First Workflow
For new optimizations or bug fixes, always write the test first.

1.  **Commit 1**: Add a new LIT test case that demonstrates the *original* behavior (e.g., the bug or the pre-optimization IR). Use `update_test_checks.py` to generate baseline `CHECK` lines.
2.  **Commit 2**: Implement your change. Run `update_test_checks.py` again to update the `CHECK` lines to reflect the *new, desired* behavior.

### 5.2 Comprehensive LIT Testing
LLVM's LIT framework is the definitive testing tool. Ensure your tests are:

*   **Minimal**: Test only the specific pattern being transformed.
*   **Meaningful Names**: Use descriptive test names (e.g., `@@add_of_selects_multi_use`).
*   **Negative Tests**: Include cases where your transform *should not* apply.
*   **Multi-Use Tests**: Verify your transform doesn't increase instruction count if results are used multiple times.
*   **Commuted Tests**: For commutative operations, test swapped operands (unless one is a constant, which is canonicalized).
*   **Vector Tests**: Include splat, splat-with-poison, and non-splat vector cases.
*   **Flag Tests**: Test interactions with flags like `nuw`, `nsw`, and specific fast-math-flags (`nnan`, `nsz`, `reassoc`).

Example of a multi-use test:
```llvm
; RUN: opt -S -instcombine < %s | FileCheck %s

; CHECK-LABEL: @add_mul_const_multi_use
; CHECK: %[[ADD:.*]] = add i8 %x, 1
; CHECK: call void @use(i8 %[[ADD]])
; CHECK: %[[MUL:.*]] = mul i8 %[[ADD]], 3
; CHECK: ret i8 %[[MUL]]
define i8 @add_mul_const_multi_use(i8 %x) {
  %add = add i8 %x, 1
  call void @use(i8 %add) ; Extra use
  %mul = mul i8 %add, 3
  ret i8 %mul
}
declare void @use(i8)
```

### 5.3 Provide Alive2 Proofs
For complex transformations, especially in InstCombine, include Alive2 proofs in your pull request description to demonstrate correctness. This significantly speeds up review.
```
// Example Alive2 proof snippet in PR description
// %0 = add i32 %a, 1
// %1 = mul i32 %0, 2
// =>
// %1 = add i32 (mul i32 %a, 2), 2
```