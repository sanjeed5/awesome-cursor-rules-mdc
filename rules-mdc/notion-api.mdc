---
description: Definitive guidelines for building secure, typed, and maintainable integrations with the Notion API using modern best practices and official SDKs.
globs: **/*
---
# notion-api Best Practices

This guide establishes the definitive best practices for interacting with the Notion API. Adhering to these rules ensures our integrations are secure, performant, type-safe, and easily maintainable. We prioritize official SDKs and structured data handling.

## 1. Code Organization and Structure

Organize your Notion API interactions into dedicated modules. Separate concerns like authentication, data fetching, and data transformation.

**✅ GOOD: Modular Structure**
```typescript
// src/notion/client.ts
import { Client } from '@notionhq/client';

export const notionClient = new Client({
  auth: process.env.NOTION_API_TOKEN,
});

// src/notion/pages.ts
import { notionClient } from './client';
import { CreatePageParameters, GetPageResponse } from '@notionhq/client/build/src/api-endpoints';

export async function createNotionPage(params: CreatePageParameters): Promise<GetPageResponse> {
  return notionClient.pages.create(params);
}

// src/notion/databases.ts
import { notionClient } from './client';
import { QueryDatabaseParameters, QueryDatabaseResponse } from '@notionhq/client/build/src/api-endpoints';

export async function queryNotionDatabase(databaseId: string, params?: QueryDatabaseParameters): Promise<QueryDatabaseResponse> {
  return notionClient.databases.query({ database_id: databaseId, ...params });
}
```

**❌ BAD: Monolithic or Scattered Logic**
```typescript
// src/index.ts (or any random file)
import { Client } from '@notionhq/client';

async function main() {
  const notion = new Client({ auth: 'secret_token_hardcoded' }); // Hardcoded token is a major NO
  const response = await notion.databases.query({ database_id: 'some_id' });
  // ... page creation, block updates, all in one place
}
```

## 2. Authentication and Token Management

Always use granular integration tokens, store them in environment variables, and implement rotation. Never hardcode tokens.

**✅ GOOD: Secure Token Handling**
```typescript
// .env
NOTION_API_TOKEN="secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

// src/notion/client.ts
import { Client } from '@notionhq/client';

if (!process.env.NOTION_API_TOKEN) {
  throw new Error('NOTION_API_TOKEN is not set in environment variables.');
}

export const notionClient = new Client({
  auth: process.env.NOTION_API_TOKEN,
});
```

**❌ BAD: Insecure Token Handling**
```typescript
// Directly in code
const notion = new Client({ auth: 'secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' });
```

## 3. Typed SDKs and Data Structures

Leverage the official Notion SDKs (e.g., `@notionhq/client` for TypeScript/JavaScript, `notion-sdk` for Python) for type safety and schema validation. Prefer page properties for structured data and blocks for free-form content.

**✅ GOOD: Using Typed SDK for Page Creation**
```typescript
import { notionClient } from './client';
import { CreatePageParameters } from '@notionhq/client/build/src/api-endpoints';

async function createProjectPage(databaseId: string, projectName: string, dueDate: string) {
  const params: CreatePageParameters = {
    parent: { database_id: databaseId },
    properties: {
      'Name': {
        title: [{ text: { content: projectName } }],
      },
      'Due Date': {
        date: { start: dueDate },
      },
      'Status': {
        select: { name: 'To Do' },
      },
    },
    children: [ // Use children for rich content, not properties
      {
        object: 'block',
        type: 'paragraph',
        paragraph: {
          rich_text: [{ type: 'text', text: { content: 'Initial project description.' } }],
        },
      },
    ],
  };
  return notionClient.pages.create(params);
}
```

**❌ BAD: Manual JSON Construction & Mixing Concerns**
```typescript
// Prone to typos, schema mismatches, and difficult to maintain
async function createUntypedPage(databaseId: string, projectName: string) {
  return fetch('https://api.notion.com/v1/pages', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.NOTION_API_TOKEN}`,
      'Notion-Version': '2022-06-28', // Old version, bad practice
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      parent: { database_id: databaseId },
      properties: {
        'Name': {
          title: [{ text: { content: projectName } }],
        },
        'Description': { // Attempting to put rich text in a property that expects a different type
          rich_text: [{ text: { content: 'This is a description.' } }],
        },
      },
    }),
  });
}
```

## 4. Performance Considerations

Optimize API calls by fetching only necessary data, using pagination, and batching operations where supported.

**✅ GOOD: Paginated Database Query with Filters**
```typescript
import { notionClient } from './client';
import { QueryDatabaseResponse } from '@notionhq/client/build/src/api-endpoints';

async function getAllActiveTasks(databaseId: string): Promise<QueryDatabaseResponse['results']> {
  let allResults: QueryDatabaseResponse['results'] = [];
  let cursor: string | undefined = undefined;

  while (true) {
    const response = await notionClient.databases.query({
      database_id: databaseId,
      filter: {
        property: 'Status',
        select: {
          does_not_equal: 'Done',
        },
      },
      page_size: 100, // Fetch in reasonable chunks
      start_cursor: cursor,
    });

    allResults = allResults.concat(response.results);
    if (!response.has_more) {
      break;
    }
    cursor = response.next_cursor || undefined;
  }
  return allResults;
}
```

**❌ BAD: Fetching All Data Without Pagination**
```typescript
// Will fail for large databases, or be very slow
async function getTooManyTasks(databaseId: string) {
  return notionClient.databases.query({
    database_id: databaseId,
    // No pagination, no filters
  });
}
```

## 5. Error Handling and Rate Limiting

Implement robust error handling, including retries with exponential back-off for rate limits and transient errors. Always check `X-RateLimit-Remaining`.

**✅ GOOD: Robust Error Handling with Exponential Back-off**
```typescript
import { notionClient } from './client';
import { APIResponseError } from '@notionhq/client';

async function safeNotionCall<T>(
  fn: () => Promise<T>,
  retries = 3,
  delay = 1000 // 1 second
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (error instanceof APIResponseError) {
      if (error.status === 429 && retries > 0) { // Rate limit error
        console.warn(`Rate limit hit. Retrying in ${delay / 1000}s...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return safeNotionCall(fn, retries - 1, delay * 2); // Exponential back-off
      }
      console.error(`Notion API Error (${error.status}): ${error.message}`);
      throw error;
    }
    console.error('Unexpected error during Notion API call:', error);
    throw error;
  }
}

// Usage
async function updatePageSafely(pageId: string, properties: any) {
  await safeNotionCall(() =>
    notionClient.pages.update({ page_id: pageId, properties })
  );
}
```

**❌ BAD: Ignoring Errors or Blind Retries**
```typescript
async function unsafeNotionCall(pageId: string, properties: any) {
  try {
    await notionClient.pages.update({ page_id: pageId, properties });
  } catch (error) {
    // Just logs, no retry, no specific handling
    console.error('Failed to update page:', error);
  }
}
```

## 6. Request/Response Patterns

Understand and leverage the `object` and `type` fields in Notion API responses for dynamic content handling. Always assume nested blocks and rich text.

**✅ GOOD: Processing Block Children**
```typescript
import { notionClient } from './client';
import { BlockObjectResponse } from '@notionhq/client/build/src/api-endpoints';

async function processBlockChildren(blockId: string) {
  const { results } = await notionClient.blocks.children.list({ block_id: blockId });

  for (const block of results) {
    if ('type' in block) { // Type guard for BlockObjectResponse
      console.log(`Block Type: ${block.type}`);
      if (block.type === 'paragraph' && block.paragraph.rich_text) {
        console.log('Paragraph content:', block.paragraph.rich_text.map(rt => rt.plain_text).join(''));
      }
      if (block.has_children) {
        console.log(`Block ${block.id} has children. Recursively processing...`);
        await processBlockChildren(block.id); // Recursive call for nested blocks
      }
    }
  }
}
```

**❌ BAD: Assuming Flat Structure or Ignoring `has_children`**
```typescript
async function incompleteBlockProcessing(blockId: string) {
  const { results } = await notionClient.blocks.children.list({ block_id: blockId });
  // Only processes top-level blocks, misses all nested content
  for (const block of results) {
    if ('type' in block && block.type === 'paragraph') {
      console.log(block.paragraph.rich_text[0]?.plain_text);
    }
  }
}
```

## 7. Testing Approaches

Implement unit tests for utility functions and integration tests for API interactions. Use mock clients for unit tests and a dedicated, isolated Notion workspace for integration tests.

**✅ GOOD: Mocking Notion Client for Unit Tests**
```typescript
// src/utils/notionHelpers.ts
export function extractPageTitle(page: any): string {
  const titleProperty = page.properties.Name?.title;
  return titleProperty ? titleProperty.map((t: any) => t.plain_text).join('') : 'Untitled';
}

// tests/notionHelpers.test.ts
import { extractPageTitle } from '../src/utils/notionHelpers';

describe('extractPageTitle', () => {
  it('should extract the title from a Notion page object', () => {
    const mockPage = {
      properties: {
        Name: {
          title: [{ type: 'text', text: { content: 'My Test Page' }, plain_text: 'My Test Page' }],
        },
      },
    };
    expect(extractPageTitle(mockPage)).toBe('My Test Page');
  });

  it('should return "Untitled" if title property is missing', () => {
    const mockPage = { properties: {} };
    expect(extractPageTitle(mockPage)).toBe('Untitled');
  });
});
```

**✅ GOOD: Integration Tests with Dedicated Workspace**
*   Set up a separate Notion workspace or database specifically for testing.
*   Use a dedicated integration token with minimal permissions for this test workspace.
*   Clean up test data after each test run.

```typescript
// tests/integration/notion.test.ts
import { notionClient } from '../../src/notion/client'; // Your actual client
import { v4 as uuidv4 } from 'uuid';

const TEST_DATABASE_ID = process.env.NOTION_TEST_DATABASE_ID!; // Ensure this is set

describe('Notion API Integration', () => {
  let createdPageId: string;

  beforeAll(() => {
    if (!TEST_DATABASE_ID) {
      throw new Error('NOTION_TEST_DATABASE_ID must be set for integration tests.');
    }
  });

  afterEach(async () => {
    // Clean up created pages
    if (createdPageId) {
      await notionClient.pages.update({
        page_id: createdPageId,
        archived: true, // Archive instead of delete to preserve history if needed
      });
      createdPageId = ''; // Reset for next test
    }
  });

  it('should create and retrieve a page in the test database', async () => {
    const pageTitle = `Test Page ${uuidv4()}`;
    const newPage = await notionClient.pages.create({
      parent: { database_id: TEST_DATABASE_ID },
      properties: {
        'Name': {
          title: [{ text: { content: pageTitle } }],
        },
      },
    });
    createdPageId = newPage.id;
    expect(newPage.properties.Name.type).toBe('title');

    const retrievedPage = await notionClient.pages.retrieve({ page_id: createdPageId });
    expect(extractPageTitle(retrievedPage)).toBe(pageTitle);
  });
});
```

**❌ BAD: No Testing or Manual Testing Only**
*   Relying solely on manual checks after deployment.
*   No automated verification of API interactions.