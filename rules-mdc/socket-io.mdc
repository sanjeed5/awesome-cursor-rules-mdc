---
description: Definitive guidelines for building robust, performant, and maintainable real-time applications using Socket.IO v4.x, emphasizing modern JavaScript practices and common pitfalls.
globs: **/*.{js,jsx,ts,tsx}
---
# Socket.IO Best Practices

Socket.IO v4.x is the bedrock for real-time communication in our applications. This guide establishes the definitive standards for its usage, focusing on reliability, performance, and maintainability.

## 1. Core Initialization & Security

Always initialize Socket.IO clients with explicit, secure, and efficient options.

### ✅ GOOD: Secure & Explicit Client Initialization
Use `wss://` in production, enable CORS on the server for cross-origin clients, and avoid `forceNew` unless strictly necessary.

```javascript
// client.js
import { io } from "socket.io-client";

// For production, always use wss://
const socket = io("wss://api.yourdomain.com", {
  path: "/socket.io/", // Explicitly define path if customized on server
  transports: ["websocket", "polling"], // Prefer WebSocket, fallback to polling
  autoConnect: true, // Connect automatically
  auth: {
    token: "YOUR_AUTH_TOKEN", // Pass authentication data
  },
  query: {
    // Additional query parameters
    clientId: "unique-client-id",
  },
});

// Server-side (Node.js with Express)
// Ensure CORS is correctly configured for your client origin
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const ioServer = new Server(server, {
  cors: {
    origin: "https://yourclientdomain.com", // Your client's domain
    methods: ["GET", "POST"],
    credentials: true,
  },
  path: "/socket.io/", // Match client path
});

ioServer.on("connection", (socket) => {
  // Access auth token: socket.handshake.auth.token
  // Access query params: socket.handshake.query.clientId
  console.log(`Client connected: ${socket.id}`);
});
```

### ❌ BAD: Insecure or Inefficient Initialization
Avoid `http://` in production, implicit options, and unnecessary `forceNew`.

```javascript
// client.js
// ❌ Insecure in production, relies on implicit defaults
const socket = io("http://localhost:3000");

// ❌ Creates a new underlying connection even if one exists for the same URL,
//    wasting resources unless specifically required for isolation.
const anotherSocket = io("https://api.yourdomain.com", { forceNew: true });
```

## 2. Code Organization: Namespaces & Rooms

Structure your real-time logic using namespaces and rooms to isolate concerns and manage communication efficiently.

### ✅ GOOD: Logical Separation with Namespaces and Rooms
Use namespaces for distinct application features (e.g., `/chat`, `/admin`) and rooms for grouping clients within a namespace (e.g., `roomId: 'general'`, `userId: 'user123'`).

```javascript
// Server-side
const chatNamespace = ioServer.of("/chat");
chatNamespace.use((socket, next) => {
  // Namespace-specific authentication/authorization
  if (socket.handshake.auth.token === "valid-chat-token") {
    next();
  } else {
    next(new Error("Authentication error for chat namespace"));
  }
});

chatNamespace.on("connection", (socket) => {
  console.log(`User ${socket.id} connected to /chat`);

  socket.on("joinRoom", (roomName) => {
    socket.join(roomName);
    socket.emit("roomJoined", `You joined ${roomName}`);
    chatNamespace.to(roomName).emit("userJoined", `${socket.id} joined ${roomName}`);
  });

  socket.on("sendMessage", ({ room, message }) => {
    chatNamespace.to(room).emit("newMessage", { user: socket.id, message });
  });
});

// Client-side
const chatSocket = io("wss://api.yourdomain.com/chat", {
  auth: { token: "valid-chat-token" },
});
chatSocket.on("connect", () => {
  chatSocket.emit("joinRoom", "general");
});
chatSocket.on("newMessage", (data) => console.log("New chat message:", data));
```

### ❌ BAD: Monolithic Event Handling
Avoid dumping all events into the default namespace without logical grouping.

```javascript
// Server-side
// ❌ All events mixed in the root namespace, hard to manage access and scale
ioServer.on("connection", (socket) => {
  socket.on("chatMessage", () => {
    /* ... */
  });
  socket.on("adminAction", () => {
    /* ... */
  });
});
```

## 3. Robust Error Handling & Reconnection

Socket.IO provides automatic reconnection, but you must handle connection errors and customize retry strategies for production.

### ✅ GOOD: Comprehensive Error & Reconnection Handling
Listen for `connect_error` to diagnose issues and `reconnect_attempt` to customize back-off strategies.

```javascript
// Client-side
socket.on("connect_error", (err) => {
  console.error("Connection error:", err.message);
  // Implement custom exponential back-off if needed, though Socket.IO has defaults
  if (err.message === "Authentication error") {
    console.error("Authentication failed, stopping reconnection attempts.");
    socket.disconnect(); // Stop trying to reconnect if auth permanently fails
  }
});

socket.on("reconnect_attempt", (attemptNumber) => {
  console.log(`Reconnection attempt ${attemptNumber}`);
  // Dynamically adjust query parameters or auth tokens if necessary
  // socket.io.opts.query = { newAuthToken: "..." };
});

socket.on("reconnect", (attemptNumber) => {
  console.log(`Reconnected after ${attemptNumber} attempts.`);
});

socket.on("disconnect", (reason) => {
  console.log("Disconnected:", reason);
  if (reason === "io server disconnect") {
    // The server has forcefully disconnected the client,
    // do not attempt to reconnect automatically.
    // Instead, prompt user or log for investigation.
  }
});
```

### ❌ BAD: Ignoring Connection State
Neglecting error events leads to silent failures and poor user experience.

```javascript
// Client-side
// ❌ No error handling, user won't know if connection fails
socket.on("connect", () => console.log("Connected"));
// ... no other connection-related event listeners
```

## 4. Payload Hygiene & Acknowledgements

Keep payloads lean, avoid circular references, and use acknowledgements for critical data delivery guarantees.

### ✅ GOOD: Efficient Payloads & Guaranteed Delivery
Send only necessary data. For critical operations, use acknowledgements.

```javascript
// Client-side
socket.emit("createOrder", { productId: "P123", quantity: 2 }, (response) => {
  if (response.success) {
    console.log("Order created successfully:", response.orderId);
  } else {
    console.error("Failed to create order:", response.error);
  }
});

// Server-side
ioServer.on("connection", (socket) => {
  socket.on("createOrder", (orderData, callback) => {
    try {
      // Process orderData, ensure it's clean and valid
      const newOrder = processOrder(orderData); // Avoid circular references here
      callback({ success: true, orderId: newOrder.id }); // Acknowledge with result
    } catch (error) {
      callback({ success: false, error: error.message });
    }
  });
});
```

### ❌ BAD: Bloated Payloads & Unreliable Delivery
Sending large, unoptimized data or assuming delivery for critical actions.

```javascript
// Client-side
// ❌ Sending an entire complex object graph, potentially with circular references
const userSession = { /* ... large object ... */ };
socket.emit("updateSession", userSession); // No acknowledgement, delivery not guaranteed

// Server-side
ioServer.on("connection", (socket) => {
  socket.on("updateSession", (sessionData) => {
    // ❌ If sessionData has circular references, this will crash
    JSON.stringify(sessionData);
    // ... no callback for client to confirm update
  });
});
```

## 5. Performance & Scalability

Optimize for network efficiency and design for horizontal scaling.

### ✅ GOOD: Minimal Data & Scalable Design
Send only deltas, use binary for large data, and consider a Redis adapter for multi-server deployments.

```javascript
// Client-side (sending binary data)
const imageBuffer = new ArrayBuffer(1024); // Example binary data
socket.emit("uploadImage", imageBuffer);

// Server-side (using Redis adapter for scalability)
const { createAdapter } = require("@socket.io/redis-adapter");
const { createClient } = require("redis");

const pubClient = createClient({ url: "redis://localhost:6379" });
const subClient = pubClient.duplicate();

Promise.all([pubClient.connect(), subClient.connect()]).then(() => {
  ioServer.adapter(createAdapter(pubClient, subClient));
  // Now events can be broadcast across multiple Socket.IO servers
  ioServer.emit("globalEvent", "This reaches all clients across all servers");
});
```

### ❌ BAD: Inefficient Data Transfer & Monolithic Servers
Sending full state updates repeatedly or running a single server without an adapter for high-traffic scenarios.

```javascript
// Client-side
// ❌ Sending entire game state every frame instead of just changes
const gameState = { /* ... huge object ... */ };
setInterval(() => {
  socket.emit("gameStateUpdate", gameState);
}, 16); // 60 FPS

// Server-side
// ❌ No adapter for scaling, only works for a single server instance
// ioServer.emit("globalEvent", "This only reaches clients on THIS server");
```

## 6. Type Safety with TypeScript

Always use TypeScript to define clear contracts for events and payloads, preventing runtime errors.

### ✅ GOOD: Strongly Typed Event Definitions
Define interfaces for event names and their corresponding payloads.

```typescript
// types/socket.ts
export interface ServerToClientEvents {
  noArg: () => void;
  basicEmit: (a: number, b: string, c: Buffer) => void;
  withAck: (d: string, callback: (e: number) => void) => void;
  roomJoined: (roomName: string) => void;
  newMessage: (data: { user: string; message: string }) => void;
}

export interface ClientToServerEvents {
  hello: () => void;
  createOrder: (orderData: { productId: string; quantity: number }, callback: (response: { success: boolean; orderId?: string; error?: string }) => void) => void;
  joinRoom: (roomName: string) => void;
  sendMessage: (data: { room: string; message: string }) => void;
}

export interface InterServerEvents {
  ping: () => void;
}

export interface SocketData {
  userId: string;
  foo: string;
}
```

```typescript
// server.ts
import { Server } from "socket.io";
import { ClientToServerEvents, ServerToClientEvents, InterServerEvents, SocketData } from "./types/socket";

const ioServer = new Server<ClientToServerEvents, ServerToClientEvents, InterServerEvents, SocketData>(3000);

ioServer.on("connection", (socket) => {
  socket.data.userId = "abc"; // Type-safe socket data
  socket.on("createOrder", (orderData, callback) => {
    console.log(orderData.productId); // Autocomplete and type checking
    callback({ success: true, orderId: "ORD123" });
  });
});

// client.ts
import { io, Socket } from "socket.io-client";
import { ClientToServerEvents, ServerToClientEvents } from "./types/socket";

const socket: Socket<ServerToClientEvents, ClientToServerEvents> = io("wss://api.yourdomain.com");

socket.on("roomJoined", (roomName) => {
  console.log(`Joined room: ${roomName}`); // Type-safe event listener
});

socket.emit("createOrder", { productId: "P456", quantity: 1 }, (response) => {
  if (response.success) {
    console.log(response.orderId); // Type-safe response
  }
});
```

### ❌ BAD: Untyped Events
Using plain JavaScript or `any` types for events, leading to potential runtime errors and difficult debugging.

```javascript