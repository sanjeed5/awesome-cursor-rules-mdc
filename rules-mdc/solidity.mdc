---
description: Definitive guidelines for writing secure, maintainable, and gas-efficient Solidity smart contracts, emphasizing modern best practices and security-first development.
globs: **/*
---
# Solidity Best Practices

This guide outlines the definitive best practices for writing Solidity smart contracts, ensuring security, readability, and efficiency. Adhere to these rules for all new and refactored code.

## 1. Code Organization & Structure

### 1.1 File Header
Every Solidity file MUST start with an SPDX license identifier and a precise pragma statement. Pin to a specific, audited compiler version.

❌ BAD
```solidity
pragma solidity ^0.8.0; // Vague pragma, allows minor version bumps
// No license
contract MyContract {}
```

✅ GOOD
```solidity
// SPDX-License-Identifier: MIT
pragma solidity 0.8.29; // Pin to specific, audited compiler version
contract MyContract {}
```

### 1.2 Indentation & Whitespace
Use 4 spaces for indentation. Never use tabs. Maintain consistent spacing around operators and within control structures.

❌ BAD
```solidity
function myFunction (uint256  _value) public {
if(_value > 0){ emit Event(_value);}
}
```

✅ GOOD
```solidity
function myFunction(uint256 _value) public {
    if (_value > 0) {
        emit Event(_value);
    }
}
```

### 1.3 Blank Lines
Use two blank lines between top-level declarations (contracts, interfaces, libraries). Use one blank line between function declarations within a contract.

❌ BAD
```solidity
contract A {}
contract B {}
contract C {
    function foo() public {}
    function bar() public {}
}
```

✅ GOOD
```solidity
contract A {}


contract B {}


contract C {
    function foo() public {}

    function bar() public {}
}
```

### 1.4 Maximum Line Length
Keep lines under 120 characters. Wrap long lines with a single indent for arguments, placing the closing parenthesis/semicolon on its own line.

❌ BAD
```solidity
function reallyLongFunctionName(address _arg1, uint256 _arg2, bytes32 _arg3, bool _arg4, string memory _arg5) public pure returns (bool) { return true; }
```

✅ GOOD
```solidity
function reallyLongFunctionName(
    address _arg1,
    uint256 _arg2,
    bytes32 _arg3,
    bool _arg4,
    string memory _arg5
)
    public
    pure
    returns (bool)
{
    return true;
}
```

### 1.5 Order of Elements
Organize contract elements for clarity and consistency.

1.  `type` declarations (structs, enums)
2.  State variables (constants, immutables, then others by visibility: public, internal, private)
3.  Events
4.  Errors
5.  Constructor
6.  `receive()` function (if exists)
7.  `fallback()` function (if exists)
8.  External functions (view/pure last)
9.  Public functions (view/pure last)
10. Internal functions (view/pure last)
11. Private functions (view/pure last)

### 1.6 Naming Conventions
Follow the Solidity style guide: `PascalCase` for Contracts, Structs, Enums, Custom Errors; `camelCase` for Functions, Events, Modifiers, Local Variables, State Variables; `SCREAMING_SNAKE_CASE` for `constant` and `immutable` variables. Prefix `_` for function parameters.

❌ BAD
```solidity
contract my_contract {
    uint256 PUBLIC_VAR;
    function get_value() internal view returns (uint256) {
        uint256 local_var = 1;
        return local_var;
    }
}
```

✅ GOOD
```solidity
contract MyContract {
    uint256 public publicVar;
    uint256 internal immutable MY_IMMUTABLE_VALUE;

    function getValue(uint256 _input) internal view returns (uint256) {
        uint256 localVariable = _input;
        return localVariable;
    }
}
```

### 1.7 Visibility Declarations
Explicitly declare visibility (`public`, `external`, `internal`, `private`) for all state variables and functions. Never rely on defaults.

❌ BAD
```solidity
uint256 myVar; // Defaults to internal
function calculate() returns (uint256) { // Defaults to public
    return 1;
}
```

✅ GOOD
```solidity
uint256 private myVar;
function calculate() external view returns (uint256) {
    return 1;
}
```

### 1.8 Natspec Documentation
Document all public-facing functions, events, and custom errors using Natspec. This is critical for clarity and security audits.

❌ BAD
```solidity
function transfer(address to, uint256 amount) public returns (bool) {
    // ...
}
```

✅ GOOD
```solidity
/// @notice Transfers `amount` tokens from the caller to `to`.
/// @param to The address to transfer tokens to.
/// @param amount The amount of tokens to transfer.
/// @return A boolean indicating whether the transfer was successful.
/// @dev Emits a {Transfer} event.
function transfer(address to, uint256 amount) public returns (bool) {
    // ...
}
```

## 2. Common Patterns & Anti-patterns

### 2.1 Custom Errors for Reverts
Always use custom errors instead of `require` strings. This saves gas and provides clearer, structured error handling.

❌ BAD
```solidity
function withdraw(uint256 amount) public {
    require(balance[msg.sender] >= amount, "Insufficient balance");
    // ...
}
```

✅ GOOD
```solidity
error InsufficientBalance(uint256 requested, uint256 available);

function withdraw(uint256 amount) public {
    if (balance[msg.sender] < amount) {
        revert InsufficientBalance(amount, balance[msg.sender]);
    }
    // ...
}
```

### 2.2 Immutable and Constant Variables
Use `immutable` for state variables initialized in the constructor and never changed. Use `constant` for compile-time fixed values. This saves gas by storing values in code, not storage.

❌ BAD
```solidity
uint256 public owner; // Stored in storage, mutable
uint256 public MAX_SUPPLY = 1000; // Stored in storage, mutable
constructor() {
    owner = msg.sender;
}
```

✅ GOOD
```solidity
address public immutable OWNER; // Stored in code, immutable
uint256 public constant MAX_SUPPLY = 1000; // Stored in code, constant

constructor() {
    OWNER = msg.sender;
}
```

### 2.3 `unchecked` Blocks
Only use `unchecked` blocks when you have *provably* ensured that arithmetic operations will not overflow or underflow. This is a micro-optimization and should be used sparingly, as it bypasses default 0.8.x safety.

❌ BAD
```solidity
function add(uint256 a, uint256 b) public pure returns (uint256) {
    return a + b; // Relying on default 0.8.x overflow checks
}
```

✅ GOOD
```solidity
function add(uint256 a, uint256 b) public pure returns (uint256) {
    // If a+b is guaranteed not to overflow due to external constraints
    unchecked {
        return a + b;
    }
}
```

### 2.4 Minimal On-chain Logic
Keep on-chain logic as minimal as possible. Perform heavy computation off-chain and verify results on-chain to reduce attack surface and gas costs.

## 3. Security & Pitfalls

### 3.1 Reentrancy Protection
Always use the Checks-Effects-Interactions pattern and apply reentrancy guards to functions that interact with external contracts and modify state. Leverage audited libraries like OpenZeppelin.

❌ BAD
```solidity
function withdrawFunds(uint256 amount) public {
    (bool success, ) = msg.sender.call{value: amount}(""); // External call first
    require(success, "Transfer failed");
    balance[msg.sender] -= amount; // State update after external call
}
```

✅ GOOD
```solidity
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract MyContract is ReentrancyGuard {
    mapping(address => uint256) public balance;

    function withdrawFunds(uint256 amount) public nonReentrant {
        // Checks
        require(balance[msg.sender] >= amount, "Insufficient balance");

        // Effects
        balance[msg.sender] -= amount;

        // Interactions
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
```

### 3.2 Compiler Version
Always use the latest **stable and audited** Solidity compiler version for deployment. Use the absolute latest version for static analysis to catch new warnings.

### 3.3 Avoid `delegatecall` Proxies (Unless Expert)
`delegatecall` proxy patterns are highly complex and error-prone. Prefer contract migration or simpler upgradeability patterns (e.g., data separation) unless your team has deep EVM expertise and dedicated audit resources.

### 3.4 No Inline Assembly (Unless Expert)
Avoid inline assembly unless absolutely necessary and you have mastered the EVM Yellow Paper. It bypasses compiler safety features and is a common source of critical bugs.

## 4. Testing & Verification

### 4.1 Comprehensive Unit Testing
Write thorough unit tests covering all functions, edge cases, and failure conditions. Use modern frameworks like Foundry or Hardhat.

### 4.2 Static Analysis & Fuzzing
Integrate static analysis tools (Slither, Crytic) and fuzzing/property testing tools (Echidna, Manticore) into your CI pipeline. Address all reported issues.

### 4.3 Security Audits
Prioritize external security audits for all production-ready contracts. No amount of internal testing replaces a professional audit.

### 4.4 Post-Deployment Monitoring
Implement robust monitoring for deployed contracts, watching logs and being prepared to react to incidents. Maintain a clear incident response plan.