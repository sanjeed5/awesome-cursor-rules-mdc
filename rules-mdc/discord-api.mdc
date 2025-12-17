---
description: Definitive guidelines for building robust, scalable, and compliant Discord applications using the discord-api, emphasizing REST over Gateway, strong typing, and modern interaction patterns.
globs: **/*
---
# discord-api Best Practices

Building Discord applications requires adherence to specific patterns for reliability and compliance. This guide outlines the essential practices for our team.

## 1. API Interaction: REST First, Gateway for Events

Always prefer the HTTP REST API for creating, updating, or deleting resources. The Gateway is exclusively for receiving real-time events. This minimizes rate-limit issues and simplifies scaling.

❌ **BAD: Using Gateway for resource modification**
```typescript
// Don't send messages or modify roles via Gateway opcodes directly
// This is an oversimplification, but illustrates the anti-pattern
gateway.send({ op: GatewayOpcodes.MessageCreate, d: { channel_id: '...', content: '...' } });
```

✅ **GOOD: Using REST for resource modification**
```typescript
import { Routes, REST } from 'discord-api-types/v10';

const rest = new REST().setToken(process.env.DISCORD_BOT_TOKEN!);

// Always specify the API version (v10 as of 2025)
await rest.post(Routes.channelMessages('12345'), {
  body: { content: 'Hello via REST!' },
});
```

## 2. Strong Typing with `discord-api-types`

Ensure all API payloads conform to Discord's official types. This prevents runtime errors and future-proofs against API changes.

❌ **BAD: Untyped payloads**
```typescript
// Prone to typos and API changes breaking at runtime
const messagePayload = {
  channel_id: '123',
  content: 'Untyped message',
};
```

✅ **GOOD: Typed payloads**
```typescript
import { APIChannel, APIMessage } from 'discord-api-types/v10';

const channel: APIChannel = await rest.get(Routes.channel('123'));
const messagePayload: APIMessage = {
  channel_id: channel.id,
  content: 'Typed message!',
  // ... other required fields
};
```

## 3. Rate Limit Handling: Exponential Back-off

Discord's API is heavily rate-limited. Implement robust exponential back-off with jitter to handle `429 Too Many Requests` responses gracefully.

❌ **BAD: Ignoring rate limits or simple retries**
```typescript
try {
  await rest.post(Routes.channelMessages('...'), { body: { content: '...' } });
} catch (error: any) {
  if (error.status === 429) {
    console.warn('Rate limited! Retrying immediately...');
    await rest.post(Routes.channelMessages('...'), { body: { content: '...' } }); // Will likely fail again
  }
}
```

✅ **GOOD: Exponential back-off with `Retry-After`**
```typescript
import { setTimeout } from 'node:timers/promises';

async function makeRequestWithRetry(requestFn: () => Promise<any>, retries = 5, delay = 1000): Promise<any> {
  try {
    return await requestFn();
  } catch (error: any) {
    if (error.status === 429 && retries > 0) {
      const retryAfter = (error.headers?.get('Retry-After') ? parseInt(error.headers.get('Retry-After')!) * 1000 : delay) + Math.random() * 500;
      console.warn(`Rate limited. Retrying in ${retryAfter}ms. Retries left: ${retries}`);
      await setTimeout(retryAfter);
      return makeRequestWithRetry(requestFn, retries - 1, delay * 2);
    }
    throw error;
  }
}

await makeRequestWithRetry(() => rest.post(Routes.channelMessages('...'), { body: { content: '...' } }));
```

## 4. Secure Token Management

Never hardcode or commit bot tokens. Use environment variables or a secure secret management system.

❌ **BAD: Hardcoded token**
```typescript
const rest = new REST().setToken('YOUR_SUPER_SECRET_TOKEN_HERE');
```

✅ **GOOD: Environment variable**
```typescript
// Ensure DISCORD_BOT_TOKEN is set in your environment
const rest = new REST().setToken(process.env.DISCORD_BOT_TOKEN!);
```

## 5. Gateway Intents: Request Only What's Needed

Specify only the Gateway Intents your bot genuinely requires. Over-requesting intents can lead to verification denial and unnecessary resource consumption.

❌ **BAD: Requesting all intents or privileged intents without justification**
```typescript
// Don't request all intents if you only need MESSAGE_CREATE
const client = new Client({ intents: [
  GatewayIntentBits.Guilds,
  GatewayIntentBits.GuildMembers,
  GatewayIntentBits.GuildMessages,
  GatewayIntentBits.MessageContent, // Privileged, requires justification
  // ... many more
]});
```

✅ **GOOD: Minimal, justified intents**
```typescript
import { Client, GatewayIntentBits } from 'discord.js'; // Or your chosen library

const client = new Client({ intents: [
  GatewayIntentBits.Guilds, // For guild-related events
  GatewayIntentBits.GuildMessages, // For message events in guilds
  GatewayIntentBits.MessageContent, // ONLY if you process message content (privileged)
]});
```

## 6. Modern UI: Slash Commands & Message Components

Prioritize Application Commands (slash commands) and Message Components (buttons, select menus) for user interaction. Text commands are largely deprecated for new features.

❌ **BAD: Relying solely on prefix commands**
```typescript
// This approach is outdated for new features
client.on('messageCreate', message => {
  if (message.content.startsWith('!mycommand')) {
    // ...
  }
});
```

✅ **GOOD: Registering and handling slash commands**
```typescript
import { ApplicationCommandType, Routes, REST } from 'discord-api-types/v10';

const commands = [{
  name: 'ping',
  description: 'Replies with Pong!',
  type: ApplicationCommandType.ChatInput,
}];

const rest = new REST().setToken(process.env.DISCORD_BOT_TOKEN!);

// Register commands globally or per guild
await rest.put(Routes.applicationCommands(process.env.DISCORD_CLIENT_ID!), { body: commands });

// Handle interactionCreate event for slash commands
client.on('interactionCreate', async interaction => {
  if (!interaction.isChatInputCommand()) return;
  if (interaction.commandName === 'ping') {
    await interaction.reply('Pong!');
  }
});
```

## 7. Code Organization: Modular and Testable

Structure your bot into modular, testable components (e.g., separate files for commands, events, services). This improves maintainability and allows for easier testing.

❌ **BAD: Monolithic bot file**
```typescript
// bot.ts
client.on('ready', () => { /* ... */ });
client.on('messageCreate', () => { /* ... */ });
// ... all logic in one file
```

✅ **GOOD: Modular structure**
```typescript
// src/bot.ts
import { Client } from 'discord.js';
import { registerCommands } from './commands';
import { registerEvents } from './events';

const client = new Client({ intents: [...] });
registerCommands(client);
registerEvents(client);
client.login(process.env.DISCORD_BOT_TOKEN);

// src/commands/index.ts
export function registerCommands(client: Client) {
  // Load and register command handlers
}

// src/events/index.ts
export function registerEvents(client: Client) {
  // Load and register event handlers
}
```

## 8. Data Storage: Use Databases

For persistent data, always use a proper database (SQL or NoSQL). Avoid JSON files for anything beyond simple configuration. JSON files are inefficient for frequent writes and concurrent access.

❌ **BAD: Storing user data in JSON files**
```typescript
// user_data.json
// { "userId": { "points": 100, "level": 5 } }
// This rewrites the entire file on every change, inefficient and error-prone
```

✅ **GOOD: Using a database**
```typescript
// Example with a simple ORM/ODM (e.g., Prisma, Mongoose, SQLAlchemy)
import { PrismaClient } from '@prisma/client';
const prisma = new PrismaClient();

async function updateUserPoints(userId: string, points: number) {
  await prisma.user.upsert({
    where: { id: userId },
    update: { points: { increment: points } },
    create: { id: userId, points: points, level: 1 },
  });
}
```

## 9. Game SDK & Rich Presence: Modern Approach

The original Game SDK is archived. For new game integrations, prioritize the Discord Social SDK. If using the legacy Game SDK, rely on its "stub" libraries to ensure compatibility. For Rich Presence, keep strings concise, actionable, and utilize all available fields for maximum impact.

❌ **BAD: Long, uninformative Rich Presence strings**
```typescript
// "Playing a very long game of my favorite game with my friends and having a great time"
```

✅ **GOOD: Concise and actionable Rich Presence**
```typescript
// "Playing [Game Name]"
// "In a party with 3 others"
// "Elapsed: 00:15:30"
// "Join Game" button
```