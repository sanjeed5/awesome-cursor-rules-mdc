---
description: Ensure Hardhat projects follow modern best practices for configuration, testing, deployment, and security using TypeScript and Hardhat 3's Rust runtime.
globs: **/*.{js,jsx}
---
# hardhat Best Practices

Hardhat 3 is the definitive standard for Ethereum development. These rules enforce a modern, secure, and efficient workflow leveraging its Rust runtime, TypeScript support, and robust testing tools.

## 1. Adopt Hardhat 3 Immediately

Migrate all projects to Hardhat 3. Its Rust-powered runtime offers superior performance, instant stack traces, and enhanced debugging. Hardhat 2 is deprecated for new development.

## 2. Standardized Project Configuration (`hardhat.config.ts`)

Always use TypeScript for your Hardhat configuration. Pin Solidity compiler versions, enable Solidity test conventions, and activate essential plugins.

**✅ GOOD: `hardhat.config.ts`**

```typescript
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox"; // Includes Ethers, Chai, Verify, etc.
import "@nomicfoundation/hardhat-ignition";
import "hardhat-gas-reporter";
import "solidity-coverage";
import "@nomicfoundation/hardhat-solhint"; // For Solhint

const config: HardhatUserConfig = {
  solidity: {
    compilers: [
      { version: "0.8.20" }, // Pin specific, stable Solidity versions
      { version: "0.8.21" },
    ],
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    hardhat: {
      // Hardhat Network for local development and testing
      chainId: 31337,
    },
    // Example for a testnet:
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "",
      accounts: process.env.PRIVATE_KEY !== undefined ? [process.env.PRIVATE_KEY] : [],
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
    // Enable Solidity test file convention (e.g., `contracts/MyContract.t.sol`)
    // This is the default, but explicitly setting it is good practice.
    // tests: { solidity: "./contracts" } // If you want .t.sol files in contracts/
  },
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: "USD",
    coinmarketcap: process.env.COINMARKETCAP_API_KEY,
  },
  typechain: {
    outDir: "typechain-types",
    target: "ethers-v6", // Use ethers-v6 for modern projects
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY,
  },
};

export default config;
```

## 3. Opinionated Code Organization

Maintain a clean, predictable project structure. Use TypeScript for all non-Solidity code (scripts, tasks, tests).

**✅ GOOD: Project Structure**

```
.
├── contracts/
│   ├── MyContract.sol
│   └── MyContract.t.sol  // Solidity unit tests
├── scripts/
│   └── deploy.ts         // TypeScript deployment scripts
├── test/
│   └── MyContract.test.ts // TypeScript integration tests
├── hardhat.config.ts
├── package.json
├── tsconfig.json
└── README.md
```

**❌ BAD: Mixing JavaScript and TypeScript, inconsistent test locations**

```
.
├── contracts/
│   └── MyContract.sol
├── scripts/
│   └── deploy.js         // JavaScript deployment script
├── test/
│   ├── MyContract.test.js // JavaScript tests
│   └── MyContract.sol.js  // Solidity tests in JS? No.
├── hardhat.config.js     // JavaScript config
└── ...
```

## 4. Comprehensive Testing Strategy

Combine fast Solidity unit tests for core logic with expressive TypeScript integration tests for full system validation.

-   **Solidity Unit Tests (`.t.sol`)**: For fast, isolated checks of contract logic.
    -   Place these directly in `contracts/` with a `.t.sol` extension.
    -   Avoid `testFail` prefix; use `vm.expectRevert` for clarity.
    -   Hardhat's Solidity tests are Foundry-compatible but have differences (no inline NatSpec, no scripting cheatcodes like `startBroadcast`).
-   **TypeScript Integration Tests (`.test.ts`)**: For end-to-end scenarios, contract interactions, and network state changes.
    -   Use `ethers.js` (via `@nomicfoundation/hardhat-ethers`) and `chai-matchers`.
    -   Leverage Hardhat Network Helpers for advanced scenarios.

**✅ GOOD: Solidity Unit Test (`contracts/MyContract.t.sol`)**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol"; // Use forge-std for cheatcodes

contract MyContractTest is Test {
    MyContract public myContract;

    function setUp() public {
        myContract = new MyContract();
    }

    function test_InitialGreeting() public {
        assertEq(myContract.greeting(), "Hello, Hardhat!");
    }

    function test_SetGreeting() public {
        myContract.setGreeting("Hola!");
        assertEq(myContract.greeting(), "Hola!");
    }

    function testRevert_SetGreetingEmpty() public {
        vm.expectRevert("Greeting cannot be empty");
        myContract.setGreeting("");
    }
}

contract MyContract {
    string public greeting;

    constructor() {
        greeting = "Hello, Hardhat!";
    }

    function setGreeting(string memory _greeting) public {
        require(bytes(_greeting).length > 0, "Greeting cannot be empty");
        greeting = _greeting;
    }
}
```

**✅ GOOD: TypeScript Integration Test (`test/MyContract.test.ts`)**

```typescript
import { expect } from "chai";
import { ethers } from "hardhat";
import { MyContract } from "../typechain-types"; // Typechain for type safety

describe("MyContract (Integration)", function () {
  let myContract: MyContract;

  beforeEach(async function () {
    const MyContractFactory = await ethers.getContractFactory("MyContract");
    myContract = await MyContractFactory.deploy();
    await myContract.waitForDeployment();
  });

  it("should return the initial greeting", async function () {
    expect(await myContract.greeting()).to.equal("Hello, Hardhat!");
  });

  it("should allow setting a new greeting", async function () {
    const newGreeting = "Bonjour!";
    await myContract.setGreeting(newGreeting);
    expect(await myContract.greeting()).to.equal(newGreeting);
  });

  it("should revert if setting an empty greeting", async function () {
    await expect(myContract.setGreeting("")).to.be.revertedWith("Greeting cannot be empty");
  });
});
```

## 5. Secure Deployment with Hardhat Ignition

Use Hardhat Ignition for declarative, reliable, and secure deployments. It handles complex dependencies and ensures atomic deployments.

**✅ GOOD: Ignition Deployment (`ignition/modules/MyContract.ts`)**

```typescript
import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

const MyContractModule = buildModule("MyContractModule", (m) => {
  const myContract = m.contract("MyContract");
  return { myContract };
});

export default MyContractModule;
```

**❌ BAD: Manual, imperative deployment scripts prone to errors**

```typescript
// scripts/deploy.ts (without Ignition)
import { ethers } from "hardhat";

async function main() {
  const MyContract = await ethers.getContractFactory("MyContract");
  const myContract = await MyContract.deploy();
  await myContract.waitForDeployment();
  console.log(`MyContract deployed to ${myContract.target}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

## 6. Smart Contract Security Fundamentals

Adhere to core security principles throughout the development lifecycle.

-   **Documentation**: Use Natspec for all Solidity contracts and functions.
-   **Minimal On-chain Logic**: Pre-process data off-chain to keep on-chain verification simple.
-   **Upgradeability**: Make a deliberate choice (e.g., UUPS proxy, contract migration). Document the procedure rigorously.
-   **Core Principles**: Apply "Defense in Depth," "Least Privilege," and "Secure Defaults."
-   **Multi-chain Simulation**: Test on target L2s (Optimism OP Stack, Base) to validate behavior before mainnet deployment.

## 7. Avoid Common Hardhat Solidity Test Pitfalls

Hardhat's Solidity testing framework is powerful but has specific conventions.

-   **`testFail` Anti-pattern**: Do not use `testFail` as a prefix. It's less explicit and not supported by default.
    **❌ BAD**
    ```solidity
    function testFail_SetGreetingEmpty() public {
        myContract.setGreeting(""); // Hardhat won't treat this as a passing test on revert
    }
    ```
    **✅ GOOD**
    ```solidity
    function testRevert_SetGreetingEmpty() public {
        vm.expectRevert("Greeting cannot be empty");
        myContract.setGreeting("");
    }
    ```
-   **Foundry Differences**: Hardhat does not support:
    -   Inline configuration via NatSpec.
    -   Scripting cheatcodes (`startBroadcast`, `stopBroadcast`).
    -   Foundry's `fixture` cheatcodes (rely on programmatic setup in `setUp`).
    -   `getCode` and `getDeployedCode` cheatcodes.

## 8. Performance and Maintainability

-   **TypeScript Everywhere**: Use TypeScript for all Hardhat scripts, tasks, and tests for static type safety, better IDE support, and fewer runtime errors.
-   **CI Integration**: Integrate your test suite into CI pipelines to automatically vet every pull request.
-   **Gas Reporting & Coverage**: Always enable `hardhat-gas-reporter` and `solidity-coverage` to monitor contract efficiency and test thoroughness.