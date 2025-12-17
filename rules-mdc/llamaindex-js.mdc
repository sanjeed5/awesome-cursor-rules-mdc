---
description: Definitive guidelines for writing robust, performant, and maintainable `llamaindex-js` applications using modern TypeScript best practices.
globs: **/*.{js,jsx,ts,tsx}
---
# llamaindex-js Best Practices

This document outlines the definitive best practices for developing with `llamaindex-js` (LlamaIndex.TS). Adhering to these guidelines ensures your applications are type-safe, performant, scalable, and easy to maintain.

## 1. Embrace TypeScript End-to-End

`llamaindex-js` is built in TypeScript for a reason. Leverage its type system to prevent runtime errors, improve code clarity, and enable robust refactoring.

**✅ GOOD:** Explicitly type all `llamaindex` objects and function parameters.

```typescript
import { Document, VectorStoreIndex, QueryEngine } from "llamaindex";
import { OpenAIEmbedding } from "llamaindex/embeddings/OpenAIEmbedding";
import { Settings } from "llamaindex/Settings";

// Centralize LLM and embedding model configuration
Settings.llm = new OpenAI({ model: "gpt-4o-mini" });
Settings.embedModel = new OpenAIEmbedding({ model: "text-embedding-3-small" });

async function buildAndQueryIndex(documents: Document[]): Promise<string> {
  const index: VectorStoreIndex = await VectorStoreIndex.fromDocuments(documents);
  const queryEngine: QueryEngine = index.asQueryEngine();
  const response = await queryEngine.query({ query: "Summarize the key points." });
  return response.response;
}
```

**❌ BAD:** Using `any` or omitting types where `llamaindex` types are available.

```typescript
// Avoid 'any' - it defeats the purpose of TypeScript
async function processData(docs: any[]): Promise<any> {
  const index = await VectorStoreIndex.fromDocuments(docs);
  const engine = index.asQueryEngine();
  const res = await engine.query({ query: "What's up?" });
  return res.response;
}
```

## 2. Structure Code in Three Distinct Layers

Organize your `llamaindex` applications into logical layers: Data Ingestion, Index Construction, and Query/Agent Execution. This promotes modularity, testability, and separation of concerns.

### 2.1. Data Ingestion (Readers/Connectors)

Responsible for loading and transforming raw data into `Document` objects.

**✅ GOOD:** Use `SimpleDirectoryReader` or specific `Reader` implementations.

```typescript
// src/data/documentLoader.ts
import { Document, SimpleDirectoryReader } from "llamaindex";

export async function loadDocumentsFromDirectory(path: string): Promise<Document[]> {
  const reader = new SimpleDirectoryReader();
  const documents = await reader.loadData(path);
  console.log(`Loaded ${documents.length} documents from ${path}`);
  return documents;
}
```

**❌ BAD:** Manually creating `Document` objects from raw strings without leveraging readers for common formats.

```typescript
// Avoid manual parsing for complex data types
function createDocumentFromText(text: string): Document {
  return new Document({ text: text }); // Okay for simple strings, but not for files/APIs
}
```

### 2.2. Index Construction

Responsible for taking `Document` objects, splitting them into `Node`s, generating embeddings, and storing them in an `Index`.

**✅ GOOD:** Use `VectorStoreIndex.fromDocuments` for simplicity and `IngestionPipeline` for advanced control.

```typescript
// src/indexing/indexBuilder.ts
import { Document, VectorStoreIndex, IngestionPipeline, SentenceSplitter } from "llamaindex";
import { OpenAIEmbedding } from "llamaindex/embeddings/OpenAIEmbedding";
import { Settings } from "llamaindex/Settings";

export async function buildVectorIndex(documents: Document[]): Promise<VectorStoreIndex> {
  // For basic cases, fromDocuments is sufficient
  // const index = await VectorStoreIndex.fromDocuments(documents);

  // For advanced control (e.g., custom transformations, caching)
  const pipeline = new IngestionPipeline({
    transformations: [
      new SentenceSplitter({ chunkSize: 512, chunkOverlap: 20 }),
      Settings.embedModel, // Use the configured embed model
    ],
    // Consider adding a DocumentStore and VectorStore for persistence
    // docstore: new SimpleDocumentStore(),
    // vectorStore: new SimpleVectorStore(),
  });

  const nodes = await pipeline.run({ documents });
  const index = await VectorStoreIndex.init({ nodes });
  console.log("Vector index built successfully.");
  return index;
}
```

**❌ BAD:** Manually creating `Node`s and embeddings. This is error-prone and bypasses `llamaindex`'s optimized pipeline.

```typescript
// Avoid this complexity unless absolutely necessary for custom scenarios
async function manualIndexCreation(documents: Document[]): Promise<VectorStoreIndex> {
  const nodes = documents.map(doc => new NodeWith  ({ text: doc.text })); // Missing splitting, metadata
  // Manually generate embeddings for each node...
  // Store in a vector store...
  return new VectorStoreIndex({ nodes: nodes }); // Incomplete
}
```

### 2.3. Query/Agent Execution

Responsible for interacting with the constructed index or agents to answer user queries or perform tasks.

**✅ GOOD:** Use `index.asQueryEngine()` for RAG or `agent()` for agentic workflows.

```typescript
// src/querying/queryService.ts
import { VectorStoreIndex, QueryEngine } from "llamaindex";
import { agent } from "@llamaindex/workflow";
import { OpenAI } from "llamaindex/llm/OpenAI";
import { tool } from "llamaindex";
import { z } from "zod";

export async function queryIndex(index: VectorStoreIndex, query: string): Promise<string> {
  const queryEngine: QueryEngine = index.asQueryEngine();
  const response = await queryEngine.query({ query });
  return response.response;
}

// src/agents/chatAgent.ts
const sumNumbersTool = tool({
  name: "sumNumbers",
  description: "Use this function to sum two numbers",
  parameters: z.object({ a: z.number(), b: z.number() }),
  execute: ({ a, b }) => `${a + b}`,
});

export async function runChatAgent(message: string): Promise<string> {
  const chatAgent = agent({
    llm: new OpenAI({ model: "gpt-4o-mini" }),
    verbose: true,
    systemPrompt: "You are a helpful assistant.",
    tools: [sumNumbersTool],
  });
  const result = await chatAgent.run(message);
  return result.data.result;
}
```

**❌ BAD:** Directly calling the LLM for RAG without a `QueryEngine`, or building agents from scratch.

```typescript
// Avoid direct LLM calls when RAG is needed
import { OpenAI } from "llamaindex/llm/OpenAI";

async function directLLMQuery(query: string): Promise<string> {
  const llm = new OpenAI({ model: "gpt-4o-mini" });
  const response = await llm.chat({ messages: [{ role: "user", content: query }] });
  return response.message.content || ""; // Lacks context from your data
}
```

## 3. Prefer `async/await` for All I/O Operations

`llamaindex-js` is inherently asynchronous. Always use `async/await` for clarity and error handling, especially for file reads, API calls, and LLM interactions.

**✅ GOOD:** Structured `async/await` with proper error handling.

```typescript
async function executePipeline(): Promise<void> {
  try {
    const documents = await loadDocumentsFromDirectory("./data");
    const index = await buildVectorIndex(documents);
    const response = await queryIndex(index, "What is the main theme?");
    console.log("Query Response:", response);
  } catch (error) {
    console.error("Pipeline failed:", error);
  }
}
```

**❌ BAD:** Chaining `.then()` and `.catch()` without `async/await` for complex flows.

```javascript
// Harder to read and debug for multi-step async operations
loadDocumentsFromDirectory("./data")
  .then(documents => buildVectorIndex(documents))
  .then(index => queryIndex(index, "What is the main theme?"))
  .then(response => console.log("Query Response:", response))
  .catch(error => console.error("Pipeline failed:", error));
```

## 4. Leverage LlamaCloud for Production-Grade Pipelines

For heavy-weight document parsing (e.g., PDFs, complex layouts) and managed retrieval, integrate with LlamaCloud services like LlamaParse. This offloads computational burden and improves accuracy.

**✅ GOOD:** Using LlamaParse for robust document processing.

```typescript
import { Document, VectorStoreIndex } from "llamaindex";
import { LlamaParseReader } from "llamaindex/readers/LlamaParseReader"; // Assuming LlamaParseReader is available

async function parseAndIndexWithLlamaCloud(filePath: string): Promise<VectorStoreIndex> {
  // Ensure LLAMA_CLOUD_API_KEY is set in your environment
  const parser = new LlamaParseReader({
    resultType: "markdown", // or "text"
    // Add other LlamaParse options as needed
  });
  const documents: Document[] = await parser.loadData(filePath);
  const index = await VectorStoreIndex.fromDocuments(documents);
  console.log("Indexed documents parsed by LlamaCloud.");
  return index;
}
```

**❌ BAD:** Attempting to implement complex document parsing logic locally for production use cases.

```typescript
// Avoid custom, brittle parsing for complex documents
import { Document } from "llamaindex";
import fs from "fs/promises";

async function localPdfParse(filePath: string): Promise<Document[]> {
  const pdfBuffer = await fs.readFile(filePath);
  // This would involve a complex, error-prone local PDF parsing library
  // which is unlikely to match LlamaParse's quality.
  console.warn("Using local PDF parsing, consider LlamaParse for robustness.");
  return [new Document({ text: "Simulated PDF content" })];
}
```

## 5. Performance: Batching, Caching, and Streaming

Optimize `llamaindex` applications for speed and responsiveness.

### 5.1. Batch Document Processing

When ingesting many documents, process them in batches to manage memory and network requests.

**✅ GOOD:** Processing documents in chunks.

```typescript
import { Document, VectorStoreIndex } from "llamaindex";

async function batchIndexDocuments(allDocuments: Document[], batchSize: number = 100): Promise<VectorStoreIndex> {
  const index = new VectorStoreIndex(); // Initialize an empty index
  for (let i = 0; i < allDocuments.length; i += batchSize) {
    const batch = allDocuments.slice(i, i + batchSize);
    console.log(`Processing batch ${i / batchSize + 1}/${Math.ceil(allDocuments.length / batchSize)}`);
    await index.insertAll(batch); // Or use pipeline.run({ documents: batch })
  }
  return index;
}
```

### 5.2. Utilize Ingestion Pipeline Caching

For repeated ingestion of the same data, use the `cache` option in `IngestionPipeline` to avoid re-embedding.

**✅ GOOD:** Configuring a cache for the ingestion pipeline.

```typescript
import { Document, IngestionPipeline, SentenceSplitter, SimpleNodeParser } from "llamaindex";
import { SimpleCache } from "llamaindex/ingestion/SimpleCache"; // Example cache
import { Settings } from "llamaindex/Settings";

async function buildCachedIndex(documents: Document[]): Promise<void> {
  const pipeline = new IngestionPipeline({
    transformations: [
      new SentenceSplitter(),
      new SimpleNodeParser(),
      Settings.embedModel,
    ],
    cache: new SimpleCache(), // Persists cached embeddings
  });
  await pipeline.run({ documents });
  console.log("Ingestion pipeline ran with caching.");
}
```

### 5.3. Implement Streaming for Chat Responses

For interactive chat applications, stream LLM responses to provide immediate feedback to the user.

**✅ GOOD:** Using `stream: true` with agents or LLMs.

```typescript
import { OpenAIAgent } from "llamaindex";
import { createStreamableUI } from "ai/rsc"; // For Next.js RSC

async function streamAgentChat(question: string): Promise<JSX.Element> {
  const agent = new OpenAIAgent({
    // ... agent configuration ...
  });

  const responseStream = await agent.chat({
    stream: true,
    message: question,
  });

  const uiStream = createStreamableUI(<div>Thinking...</div>);

  // Pipe the response stream to update the UI
  responseStream.pipeTo(
    new WritableStream({
      start: () => uiStream.update(""), // Clear initial "Thinking..."
      write: (chunk) => uiStream.append(chunk.response.delta),
      close: () => uiStream.done(),
      abort: (err) => console.error("Stream aborted:", err),
    }),
  ).catch(console.error);

  return uiStream.value;
}
```

## 6. Common Pitfalls and Gotchas

### 6.1. Avoid Browser Environments

`llamaindex-js` relies on `AsyncLocalStorage`-like APIs which are not fully supported in browser environments. Restrict `llamaindex` code to Node.js, Deno, Bun, Cloudflare Workers, or Next.js Server Components.

**✅ GOOD:** Deploy `llamaindex` logic on the server-side (e.g., API routes, serverless functions, RSC).

```typescript
// src/actions/chat.ts (Next.js Server Action)
"use server";
import { runChatAgent } from "@/agents/chatAgent";

export async function serverChatAction(message: string): Promise<string> {
  return runChatAgent(message);
}

// src/app/page.tsx (Client component calling server action)
"use client";
import { serverChatAction } from "@/actions/chat";
import { useState } from "react";

export default function HomePage() {
  const [response, setResponse] = useState<string>("");
  const handleChat = async (input: string) => {
    const result = await serverChatAction(input);
    setResponse(result);
  };
  // ... UI to call handleChat ...
}
```

**❌ BAD:** Directly importing and running `llamaindex` code in a client-side React component.

```javascript
// components/ChatInput.tsx (Client component)
"use client";
import { OpenAIAgent } from "llamaindex"; // ❌ This will likely fail in the browser

export default function ChatInput() {
  const agent = new OpenAIAgent({ /* ... */ });
  // ...
}
```

### 6.2. Securely Manage Environment Variables

Never hardcode API keys or sensitive credentials. Use environment variables and ensure they are loaded correctly for your runtime.

**✅ GOOD:** Using `process.env` (Node.js) or equivalent.

```typescript
// Ensure your .env file is loaded (e.g., with dotenv)
// .env:
// OPENAI_API_KEY=sk-...
// LLAMA_CLOUD_API_KEY=lcsk-...

import { Settings } from "llamaindex/Settings";
import { OpenAI } from "llamaindex/llm/OpenAI";

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY is not set.");
}
Settings.llm = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
```

**❌ BAD:** Hardcoding API keys directly in code.

```typescript
import { OpenAI } from "llamaindex/llm/OpenAI";
const llm = new OpenAI({ apiKey: "sk-YOUR_HARDCODED_KEY_HERE" }); // ❌ Security vulnerability
```

## 7. Robust Testing Approaches

Implement a comprehensive testing strategy to ensure the reliability and correctness of your `llamaindex` applications.

### 7.1. Unit Tests for Individual Components

Test data loaders, custom tools, and transformation functions in isolation. Mock external dependencies like LLM calls.

**✅ GOOD:** Mocking LLM responses for unit tests.

```typescript
// tests/unit/sumNumbersTool.test.ts
import { tool } from "llamaindex";
import { z } from "zod";
import { vi, describe, it, expect } from "vitest"; // or Jest

const sumNumbers = tool({
  name: "sumNumbers",
  description: "Use this function to sum two numbers",
  parameters: z.object({ a: z.number(), b: z.number() }),
  execute: ({ a, b }: { a: number; b: number }) => `${a + b}`,
});

describe("sumNumbers tool", () => {
  it("should correctly sum two numbers", async () => {
    const result = await sumNumbers.execute({ a: 5, b: 3 });
    expect(result).toBe("8");
  });
});
```

### 7.2. Integration Tests for the Full Pipeline

Test the entire RAG pipeline from data ingestion through querying. Use a small, controlled dataset.

**✅ GOOD:** End-to-end integration test.

```typescript
// tests/integration/ragPipeline.test.ts
import { Document, VectorStoreIndex, SimpleDirectoryReader } from "llamaindex";
import { Settings } from "llamaindex/Settings";
import { MockLLM } from "llamaindex/llm/mock"; // Use a mock LLM for predictable responses
import { vi, describe, it, expect, beforeAll } from "vitest";
import fs from "fs/promises";
import path from "path";

describe("RAG Pipeline Integration", () => {
  let index: VectorStoreIndex;
  const testDataDir = path.join(__dirname, "test_data");

  beforeAll(async () => {
    // Setup mock data
    await fs.mkdir(testDataDir, { recursive: true });
    await fs.writeFile(path.join(testDataDir, "test.txt"), "The capital of France is Paris.");

    // Configure a mock LLM for predictable test results
    Settings.llm = new MockLLM({
      mockFn: async ({ messages }) => ({
        message: {
          content: "Mock response: " + messages[messages.length - 1].content,
          role: "assistant",
        },
      }),
    });

    const reader = new SimpleDirectoryReader();
    const documents = await reader.loadData(testDataDir);
    index = await VectorStoreIndex.fromDocuments(documents);
  });

  it("should answer a query based on indexed data", async () => {
    const queryEngine = index.asQueryEngine();
    const response = await queryEngine.query({ query: "What is the capital of France?" });
    expect(response.response).toContain("Mock response: What is the capital of France?");
  });
});
```

### 7.3. RAG Evaluation

For production systems, use `llamaindex`'s evaluation modules to measure the quality (faithfulness, relevance, correctness) of your RAG system.

**✅ GOOD:** Setting up an evaluator.

```typescript
import { ResponseSynthesizer, ServiceContext } from "llamaindex";
import { FaithfulnessEvaluator } from "llamaindex/evaluation/FaithfulnessEvaluator";
import { MockLLM } from "llamaindex/llm/mock";

async function evaluateRAGResponse(query: string, response: string, sourceNodes: any[]): Promise<void> {
  const serviceContext = ServiceContext.fromDefaults({
    llm: new MockLLM({
      mockFn: async ({ messages }) => ({
        message: {
          content: "Mock evaluation: " + messages[messages.length - 1].content,
          role: "assistant",
        },
      }),
    }),
  });
  const evaluator = new FaithfulnessEvaluator({ serviceContext });
  const evaluationResult = await evaluator.evaluate({
    query,
    response,
    sourceNodes,
  });
  console.log("Faithfulness Evaluation:", evaluationResult.score, evaluationResult.feedback);
}
```