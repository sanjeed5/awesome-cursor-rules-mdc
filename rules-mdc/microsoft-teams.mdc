---
description: This rule file provides opinionated best practices for developing and managing applications within the Microsoft Teams ecosystem, ensuring high-quality, performant, and maintainable solutions.
globs: **/*
---
# microsoft-teams Best Practices

Microsoft Teams is the backbone of enterprise collaboration. To build high-quality, maintainable, and performant applications, adhere to these definitive guidelines.

## 1. Code Organization and Structure

Always start with a standardized project structure and leverage modern tooling.

### ✅ Use Microsoft 365 Agents Toolkit for Project Scaffolding

The `Microsoft 365 Agents Toolkit` (formerly Teams Toolkit) provides the official project structure, linting rules, and CI/CD pipelines. It's the fastest way to a compliant and maintainable app.

❌ BAD: Manual project setup, inconsistent file organization.
```typescript
// Don't manually create files and folders.
// This leads to inconsistent project structures.
// src/components/MyComponent.tsx
// src/api/teamsApi.ts
// ...
```

✅ GOOD: Generate with the toolkit.
```bash
# Use the VS Code extension or CLI
# This ensures a consistent, maintainable structure
# and pre-configured build/deploy scripts.
npm install -g @microsoft/teamsfx-cli # Or use the VS Code extension
teamsfx new --app-name my-teams-app --capabilities tab,bot
```

### ✅ Prefer TypeScript for All Development

TypeScript is mandatory for type safety, better tooling, and improved code quality.

❌ BAD: JavaScript for Teams app logic.
```javascript
// myBotLogic.js
function handleMessage(context, message) {
    // No type checking, prone to runtime errors
    if (message.text.startsWith('/hello')) {
        context.sendActivity(`Hello ${context.activity.from.name}!`);
    }
}
```

✅ GOOD: TypeScript with explicit types.
```typescript
// myBotLogic.ts
import { TurnContext, Activity } from 'botbuilder';

export async function handleMessage(context: TurnContext, activity: Activity): Promise<void> {
    if (activity.text?.startsWith('/hello')) {
        await context.sendActivity(`Hello ${activity.from?.name || 'there'}!`);
    }
}
```

## 2. Common Patterns and Anti-patterns

Adopt established patterns for UI, interaction, and data handling.

### ✅ Use Fluent UI React Components for Tabs

Ensure a native look and feel. Fluent UI is the official design system for Microsoft 365.

❌ BAD: Custom CSS frameworks or generic UI libraries.
```tsx
// MyTab.tsx
import React from 'react';
import { Button } from 'some-generic-ui-lib'; // Inconsistent look and feel

const MyTab: React.FC = () => {
  return <Button className="my-custom-button">Click Me</Button>;
};
```

✅ GOOD: Fluent UI React components.
```tsx
// MyTab.tsx
import React from 'react';
import { Button, Field } from '@fluentui/react-components'; // Native Teams experience

const MyTab: React.FC = () => {
  return (
    <Field label="Your Name">
      <Button appearance="primary">Submit</Button>
    </Field>
  );
};
```

### ✅ Leverage Adaptive Cards for Bots and Message Extensions

Adaptive Cards provide rich, platform-agnostic UI for conversational experiences.

❌ BAD: Sending plain text or basic HTML from bots.
```typescript
// bot.ts
await context.sendActivity('Here is your task: Buy milk. Due: Tomorrow.');
```

✅ GOOD: Adaptive Cards for structured information.
```typescript
// bot.ts
import { CardFactory } from 'botbuilder';

const taskCard = CardFactory.adaptiveCard({
  "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
  "version": "1.3",
  "body": [
    { "type": "TextBlock", "text": "New Task", "weight": "bolder", "size": "medium" },
    { "type": "TextBlock", "text": "Buy Milk", "wrap": true },
    { "type": "FactSet", "facts": [{ "title": "Due Date:", "value": "Tomorrow" }] }
  ]
});
await context.sendActivity({ attachments: [taskCard] });
```

### ✅ Integrate with Microsoft Graph for Data Access

Always use Microsoft Graph for accessing Microsoft 365 data (user profiles, files, calendars, etc.).

❌ BAD: Directly accessing SharePoint APIs or other M365 services.
```typescript
// teamsApi.ts
// Avoid direct SharePoint REST calls; use Graph API instead.
fetch(`https://{tenant}.sharepoint.com/_api/web/lists/GetByTitle('Documents')/items`);
```

✅ GOOD: Use Microsoft Graph SDK.
```typescript
// teamsGraphClient.ts
import { Client } from '@microsoft/microsoft-graph-client';

async function getUserProfile(accessToken: string) {
  const graphClient = Client.init({
    authProvider: (done) => { done(null, accessToken); }
  });
  const profile = await graphClient.api('/me').get();
  return profile;
}
```

## 3. Performance Considerations

Optimize for speed and responsiveness, especially for tabs.

### ✅ Optimize Tab Bundle Size and Lazy Load Components

Large JavaScript bundles impact load times. Minimize dependencies and use dynamic imports.

❌ BAD: Single large bundle, loading all components upfront.
```tsx
// App.tsx
import HeavyComponent from './HeavyComponent'; // Always loaded

const App: React.FC = () => <HeavyComponent />;
```

✅ GOOD: Lazy load non-critical components.
```tsx
// App.tsx
import React, { Suspense, lazy } from 'react';

const LazyHeavyComponent = lazy(() => import('./HeavyComponent'));

const App: React.FC = () => (
  <Suspense fallback={<div>Loading...</div>}>
    <LazyHeavyComponent />
  </Suspense>
);
```

### ✅ Implement Efficient Data Fetching with Caching and Pagination

Minimize network requests and handle large datasets gracefully.

❌ BAD: Fetching all data at once.
```typescript
// api.ts
async function getAllItems() {
  // Fetches potentially thousands of items, slow and memory intensive
  const response = await fetch('/api/items?limit=99999');
  return response.json();
}
```

✅ GOOD: Paginate and cache data.
```typescript
// api.ts
import { useQuery } from '@tanstack/react-query'; // Example with React Query

async function getItems(page: number, pageSize: number) {
  const response = await fetch(`/api/items?page=${page}&pageSize=${pageSize}`);
  return response.json();
}

// In a component:
// const { data } = useQuery(['items', page, pageSize], () => getItems(page, pageSize));
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes to prevent issues in development and deployment.

### ❌ Ignoring Teams App Store Validation Checklist

Failing to meet validation requirements leads to rejection. Always review the latest checklist. This includes ensuring your app adheres to naming conventions, governance policies, and accessibility standards.

✅ GOOD: Proactively test against all validation criteria (e.g., accessibility, branding, functionality, security).

### ❌ Poor Authentication and Authorization Handling

Incorrect SSO or permission scopes lead to security vulnerabilities and broken experiences.

✅ GOOD: Implement Azure AD SSO using the Teams SDK and request only necessary Microsoft Graph permissions.
```typescript
// auth.ts
import * as microsoftTeams from "@microsoft/teams-js";

async function getTeamsSsoToken(): Promise<string> {
  try {
    await microsoftTeams.app.initialize();
    const token = await microsoftTeams.authentication.getAuthToken();
    return token;
  } catch (error) {
    console.error("Failed to get SSO token:", error);
    throw error;
  }
}
```

### ❌ Not Transitioning from TeamsFx SDK to Teams SDK / Microsoft 365 Agents SDK

The `TeamsFx SDK` is deprecated as of September 2025. Migrate your AI-powered bots and agents immediately.

✅ GOOD: Use the `Teams SDK` for agents within Teams, and `Microsoft 365 Agents SDK` for broader M365 platform integration.
```typescript
// bot.ts (using new Teams SDK for conversational AI)
import { TeamsActivityHandler, TurnContext } from '@microsoft/teams-ai';

class MyTeamsBot extends TeamsActivityHandler {
  constructor() {
    super();
    this.onMessage(async (context, next) => {
      // Use Teams SDK specific methods for AI/agent interactions
      await context.sendActivity(`You said: ${context.activity.text}`);
      await next();
    });
  }
}
```

## 5. Testing Approaches

Comprehensive testing ensures reliability and stability.

### ✅ Implement Unit, Integration, and End-to-End Testing

A robust testing strategy is non-negotiable.

*   **Unit Tests**: For individual functions and components (e.g., Jest, Vitest).
*   **Integration Tests**: For interactions between components and APIs.
*   **End-to-End Tests**: Simulate user flows with the `Teams App Test Framework`.

❌ BAD: Manual testing only.
```typescript
// No automated tests, relying solely on manual clicks and checks.
```

✅ GOOD: Automated testing suite.
```typescript
// example.test.ts (Unit Test with Jest)
import { add } from './utils';

describe('Utils', () => {
  it('should add two numbers', () => {
    expect(add(1, 2)).toBe(3);
  });
});

// e2e.test.ts (Conceptual End-to-End Test with Teams App Test Framework)
// This framework simulates user interactions within Teams.
// import { TeamsAppTestFramework } from '@microsoft/teams-app-test-framework';
// const framework = new TeamsAppTestFramework();
// await framework.login('user@example.com');
// await framework.openTab('My Teams App');
// await framework.clickButton('Submit');
// expect(await framework.getMessageText()).toContain('Success');
```

### ✅ Test on a Dedicated Developer Tenant

Always test your app in a realistic Teams environment before production deployment. This catches environment-specific issues and permission problems early.

❌ BAD: Only testing locally or on a generic dev environment.
```bash
# Only npm start and local browser testing
```

✅ GOOD: Deploy to a dev tenant for full integration testing.
```bash
# Use Teams Toolkit to provision and deploy to a dev tenant
teamsfx deploy --env dev
```