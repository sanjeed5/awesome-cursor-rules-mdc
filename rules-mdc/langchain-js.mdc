---
description: This guide provides opinionated, actionable best practices for building robust and maintainable LLM applications with langchain-js, emphasizing modularity, type safety, observability, and testing.
globs: **/*.{js,jsx}
---
# langchain-js Best Practices

Building LLM-powered applications with `langchain-js` requires a disciplined approach to ensure maintainability, performance, and reliability. This guide outlines our team's definitive best practices.

## 1. Code Organization and Structure

Always prioritize modularity and explicit typing. Break down complex logic into small, reusable components.

### 1.1. Modular Components & Single Responsibility

Each LangChain component (Agent, Tool, Chain, Model) should reside in its own file or a dedicated module, adhering to the Single Responsibility Principle.

❌ **BAD: Monolithic file**
```javascript
// src/agent.js
import { ChatOpenAI } from "@langchain/openai";
import { createAgent, tool } from "langchain";
import { z } from "zod";

const getStockPrice = tool(async ({ ticker }) => { /* ... */ }, { /* ... */ });
const getNews = tool(async ({ query }) => { /* ... */ }, { /* ... */ });

const model = new ChatOpenAI({ temperature: 0.7 });
const agent = createAgent({
  model,
  tools: [getStockPrice, getNews],
});

export async function runFinancialAgent(input) {
  return agent.invoke(input);
}
```

✅ **GOOD: Modular, reusable components**
```typescript
// src/tools/getStockPrice.ts
import { tool } from "langchain";
import { z } from "zod";

export const getStockPriceTool = tool(
  async ({ ticker }: { ticker: string }) => {
    // Simulate API call
    if (ticker === "AAPL") return "$170.00";
    return "Price not found.";
  },
  {
    name: "get_stock_price",
    description: "Get the current stock price for a given ticker symbol.",
    schema: z.object({
      ticker: z.string().describe("The stock ticker symbol (e.g., AAPL)"),
    }),
  }
);

// src/tools/getNews.ts
import { tool } from "langchain";
import { z } from "zod";

export const getNewsTool = tool(
  async ({ query }: { query: string }) => {
    // Simulate API call
    return `Latest news for ${query}: Market is up!`;
  },
  {
    name: "get_news",
    description: "Get the latest news for a given query.",
    schema: z.object({
      query: z.string().describe("The news query"),
    }),
  }
);

// src/agents/financialAgent.ts
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import { getStockPriceTool } from "../tools/getStockPrice";
import { getNewsTool } from "../tools/getNews";

const model = new ChatOpenAI({ temperature: 0.7 });

export const financialAgent = createAgent({
  model,
  tools: [getStockPriceTool, getNewsTool],
});

// src/index.ts (or wherever the agent is invoked)
import { financialAgent } from "./agents/financialAgent";

async function main() {
  const result = await financialAgent.invoke({
    messages: [{ role: "user", content: "What's the price of AAPL and the latest market news?" }],
  });
  console.log(result);
}

main();
```

### 1.2. Explicit TypeScript Types

Always use explicit types for prompts, models, and tool interfaces. Avoid `any`.

❌ **BAD: Implicit types, no schema validation**
```javascript
const myTool = tool((input) => { /* ... */ }, { name: "my_tool" });
```

✅ **GOOD: Explicit types with Zod schema**
```typescript
import { tool } from "langchain";
import { z } from "zod";

const MyToolInputSchema = z.object({
  param1: z.string().describe("Description for param1"),
  param2: z.number().optional().describe("Description for param2"),
});

export const myTypedTool = tool(
  async (input: z.infer<typeof MyToolInputSchema>) => {
    console.log(`Received: ${input.param1}, ${input.param2}`);
    return "Processed successfully";
  },
  {
    name: "my_typed_tool",
    description: "A tool that takes typed input.",
    schema: MyToolInputSchema,
  }
);
```

### 1.3. Configuration Management

Store API keys, model parameters, and external service URLs in environment variables. Never hardcode secrets.

❌ **BAD: Hardcoded API key**
```javascript
const model = new ChatOpenAI({
  openAIApiKey: "sk-YOUR_HARDCODED_KEY",
});
```

✅ **GOOD: Environment variables**
```typescript
// .env
// OPENAI_API_KEY=sk-YOUR_OPENAI_KEY
// LANGSMITH_API_KEY=ls__YOUR_LANGSMITH_KEY
// MONGODB_URI=mongodb+srv://...

// src/config.ts
import "dotenv/config"; // Ensure dotenv is loaded early

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;
export const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY!;
export const MONGODB_URI = process.env.MONGODB_URI!;

// src/models/chatModel.ts
import { ChatOpenAI } from "@langchain/openai";
import { OPENAI_API_KEY } from "../config";

export const chatModel = new ChatOpenAI({
  openAIApiKey: OPENAI_API_KEY,
  temperature: 0.7,
  modelName: "gpt-4o",
});
```

## 2. Common Patterns and Anti-patterns

Leverage LangChain's core abstractions effectively.

### 2.1. Robust Tool Definitions

Always define tools with `zod` schemas for clear input validation and better agent reasoning.

❌ **BAD: Undefined tool schema**
```javascript
const simpleTool = tool((input) => `Processed ${input}`, { name: "simple_tool" });
```

✅ **GOOD: Tool with Zod schema**
```typescript
import { tool } from "langchain";
import { z } from "zod";

export const calculateSumTool = tool(
  async ({ a, b }: { a: number; b: number }) => (a + b).toString(),
  {
    name: "calculate_sum",
    description: "Calculates the sum of two numbers.",
    schema: z.object({
      a: z.number().describe("The first number"),
      b: z.number().describe("The second number"),
    }),
  }
);
```

### 2.2. Retrieval-Augmented Generation (RAG)

For knowledge retrieval, use `VectorStoreRetriever` within a `RunnableSequence`.

```typescript
// src/retrievers/documentRetriever.ts
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
import { MongoClient } from "mongodb";
import { MONGODB_URI } from "../config";

const client = new MongoClient(MONGODB_URI);
const collection = client.db("langchain_db").collection("documents");

export async function getVectorStoreRetriever() {
  const embeddings = new VoyageEmbeddings();
  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection,
    indexName: "vector_index", // Your Atlas Vector Search index name
    textKey: "content", // Field containing the text
    embeddingKey: "embedding", // Field containing the vector embedding
  });
  return vectorStore.asRetriever();
}

// src/chains/ragChain.ts
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";
import { chatModel } from "../models/chatModel";
import { getVectorStoreRetriever } from "../retrievers/documentRetriever";

const RAG_PROMPT_TEMPLATE = `
You are an AI assistant for answering questions about documents.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Context: {context}
Question: {question}
`;

export async function createRagChain() {
  const retriever = await getVectorStoreRetriever();
  const ragPrompt = PromptTemplate.fromTemplate(RAG_PROMPT_TEMPLATE);

  return RunnableSequence.from([
    RunnablePassthrough.assign({
      context: async (input: { question: string }) => {
        const docs = await retriever.invoke(input.question);
        return formatDocumentsAsString(docs);
      },
    }),
    ragPrompt,
    chatModel,
    new StringOutputParser(),
  ]);
}

// src/index.ts
import { createRagChain } from "./chains/ragChain";

async function runRagExample() {
  const ragChain = await createRagChain();
  const question = "What are the best practices for MongoDB Atlas?";
  const answer = await ragChain.invoke({ question });
  console.log(answer);
}

runRagExample();
```

### 2.3. Streaming for Responsiveness

Implement streaming for LLM responses to improve user experience, especially for long generations.

```typescript
// src/models/streamingChatModel.ts
import { ChatOpenAI } from "@langchain/openai";
import { OPENAI_API_KEY } from "../config";

export const streamingChatModel = new ChatOpenAI({
  openAIApiKey: OPENAI_API_KEY,
  temperature: 0.7,
  modelName: "gpt-4o",
  streaming: true, // Enable streaming
});

// src/index.ts
import { streamingChatModel } from "./models/streamingChatModel";

async function streamResponse() {
  const stream = await streamingChatModel.stream("Tell me a long story about a space cat.");
  for await (const chunk of stream) {
    process.stdout.write(chunk.content);
  }
  console.log("\n--- Stream Finished ---");
}

streamResponse();
```

### 2.4. Error Handling and Retries

Wrap external LLM and API calls with robust retry logic to handle transient failures. Use a library like `p-retry`.

❌ **BAD: No retry logic**
```typescript
import { ChatOpenAI } from "@langchain/openai";
const model = new ChatOpenAI();
try {
  await model.invoke("...");
} catch (error) {
  console.error("LLM call failed:", error); // Fails on first error
}
```

✅ **GOOD: Retry logic with exponential backoff**
```typescript
import { ChatOpenAI } from "@langchain/openai";
import pRetry from "p-retry";
import { OPENAI_API_KEY } from "../config";

const model = new ChatOpenAI({ openAIApiKey: OPENAI_API_KEY });

async function reliableLLMCall(prompt: string) {
  return pRetry(
    async () => {
      console.log("Attempting LLM call...");
      return await model.invoke(prompt);
    },
    {
      retries: 5,
      minTimeout: 1000, // 1 second
      maxTimeout: 10000, // 10 seconds
      onFailedAttempt: (error) => {
        console.warn(`Attempt ${error.attemptNumber} failed. Retrying...`);
      },
    }
  );
}

// Usage
reliableLLMCall("Generate a creative story about a robot chef.")
  .then((res) => console.log(res))
  .catch((err) => console.error("Final LLM call failed after retries:", err));
```

## 3. Performance Considerations

Optimize for speed and resource usage.

### 3.1. Asynchronous Callbacks

For non-serverless environments, set `LANGCHAIN_CALLBACKS_BACKGROUND=true` to process LangSmith traces asynchronously, reducing latency. In serverless, ensure it's `false` or unset for traces to complete before function exit.

```bash
# In your development/production environment (non-serverless)
export LANGCHAIN_CALLBACKS_BACKGROUND=true

# In your serverless environment (e.g., AWS Lambda)
# Ensure this is NOT set to true, or explicitly set to false if needed.
# export LANGCHAIN_CALLBACKS_BACKGROUND=false
```

### 3.2. Efficient Vector Search

When using vector stores, leverage metadata filtering to narrow down search space *before* vector similarity search, if applicable.

```typescript
// Example: Filtering by document source
const retriever = vectorStore.asRetriever({
  filter: {
    source: "internal_docs",
  },
});
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes to prevent headaches.

### 4.1. Ignoring LangSmith for Observability

Not integrating with LangSmith means flying blind. Every chain, agent step, and retrieval call should be traced.

❌ **BAD: No LangSmith setup**
```bash
# Missing environment variables
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=...
```

✅ **GOOD: LangSmith enabled**
```bash
# Set these in your .env or deployment environment
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
export LANGSMITH_PROJECT=<your-project-name> # Optional, but recommended
export LANGSMITH_WORKSPACE_ID=<your-workspace-id> # If using multiple workspaces
```
With these set, LangChain automatically instruments your runs.

### 4.2. Over-privileged Agents

Agents should operate with the principle of least privilege. Limit tool access and permissions.

❌ **BAD: Agent with access to sensitive file system operations**
```typescript
import { tool } from "langchain";
import * as fs from "fs/promises";
import { z } from "zod";

// This tool can delete any file! DANGEROUS!
const deleteFileTool = tool(
  async ({ path }: { path: string }) => {
    await fs.unlink(path);
    return `Deleted ${path}`;
  },
  {
    name: "delete_file",
    description: "Deletes a file at the given path.",
    schema: z.object({ path: z.string() }),
  }
);
// ... agent given deleteFileTool
```

✅ **GOOD: Sandboxed or read-only tools**
```typescript
import { tool } from "langchain";
import * as fs from "fs/promises";
import { z } from "zod";
import path from "path";

const SAFE_DIR = "/app/data/safe_uploads"; // Define a safe, restricted directory

const readFileSafeTool = tool(
  async ({ filename }: { filename: string }) => {
    const filePath = path.join(SAFE_DIR, filename);
    if (!filePath.startsWith(SAFE_DIR)) { // Prevent path traversal
      throw new Error("Access denied: Cannot read outside safe directory.");
    }
    return await fs.readFile(filePath, "utf-8");
  },
  {
    name: "read_safe_file",
    description: "Reads a file from the designated safe directory.",
    schema: z.object({ filename: z.string() }),
  }
);
// ... agent given readFileSafeTool, NOT deleteFileTool
```

## 5. Testing Approaches

Treat testing as a first-class concern.

### 5.1. Unit Testing with `FakeLLM` / `MockChatModel`

Isolate and test individual components by mocking LLM responses.

```typescript
// src/chains/greetingChain.ts
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

export function createGreetingChain(model: BaseChatModel) {
  const prompt = PromptTemplate.fromTemplate("Say hello to {name}.");
  return RunnableSequence.from([prompt, model, new StringOutputParser()]);
}

// src/chains/greetingChain.test.ts
import { test, expect, vi } from "vitest";
import { FakeChatModel } from "@langchain/core/utils/testing";
import { createGreetingChain } from "./greetingChain";

test("createGreetingChain generates correct greeting", async () => {
  const mockModel = new FakeChatModel({});
  mockModel.predict = vi.fn((input) => Promise.resolve(`Hello, ${input.split(" ")[3]}!`)); // Mock response based on prompt

  const chain = createGreetingChain(mockModel);
  const result = await chain.invoke({ name: "Alice" });

  expect(mockModel.predict).toHaveBeenCalledWith("Human: Say hello to Alice.");
  expect(result).toBe("Hello, Alice!");
});
```

### 5.2. Integration Testing with LangSmith

Run end-to-end tests against a dedicated sandbox LangSmith project to verify real-world behavior and catch regressions.

```typescript
// .env.test
// LANGSMITH_TRACING=true
// LANGSMITH_API_KEY=ls__YOUR_TEST_API_KEY
// LANGSMITH_PROJECT=my-app-integration-tests

// src/integration.test.ts
import { test, expect } from "vitest";
import { createRagChain } from "./chains/ragChain"; // Your actual RAG chain
import { Client } from "langsmith"; // For querying test results

// Ensure LangSmith is configured for this test run via .env.test
// or by setting process.env variables directly in test setup.

test("RAG chain answers question correctly with real LLM and retriever", async () => {
  const ragChain = await createRagChain(); // Uses real LLM and vector store
  const question = "What is the capital of France?";
  const answer = await ragChain.invoke({ question });

  expect(answer).toContain("Paris"); // Basic assertion
  // For more advanced validation, inspect the trace in LangSmith manually
  // or programmatically using the LangSmith SDK if needed.
});
```