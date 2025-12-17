---
description: This guide defines definitive best practices for building robust, scalable, and secure Express.js applications, emphasizing modern patterns and common pitfalls.
globs: **/*.{js,ts}
---
# express Best Practices

Express.js is the backbone of many Node.js backends. To build scalable, maintainable, and secure APIs in 2025, you *must* adhere to strict coding standards. This guide provides the definitive rules for our team.

## 1. Code Organization and Structure

Adopt a modular, layered architecture. This is non-negotiable for maintainability.

### 1.1. Enforce a Strict Modular Folder Structure

Separate concerns into dedicated directories.

❌ **BAD: Monolithic `server.js`**
```javascript
// server.js
const express = require('express');
const app = express();
const mongoose = require('mongoose');
// ... all models, routes, controllers, middleware here
app.get('/users', async (req, res) => { /* ... */ });
mongoose.connect('...');
app.listen(3000);
```

✅ **GOOD: Layered Structure**
```
📁 src/
 ├── config/          # Environment variables, DB connection
 ├── controllers/     # Request handling, orchestrates services
 ├── models/          # Mongoose schemas, data access
 ├── routes/          # API endpoint definitions
 ├── middlewares/     # Reusable Express middleware
 ├── services/        # Core business logic
 ├── utils/           # Helper functions
 ├── app.js           # Express app setup, middleware, route registration
 └── server.js        # Server start, DB connection, graceful shutdown
📁 .env               # Environment variables
```

### 1.2. Separate `app.js` from `server.js`

`app.js` configures the Express application. `server.js` starts the HTTP server and handles infrastructure (DB connection, graceful shutdown). This enables easier testing and deployment.

**`src/app.js`**
```javascript
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const userRoutes = require('./routes/userRoutes');
const { notFound, errorHandler } = require('./middlewares/errorMiddleware');

const app = express();

app.use(helmet()); // Essential security headers
app.use(cors());   // Enable CORS
app.use(express.json()); // Body parser

app.use('/api/v1/users', userRoutes);

app.use(notFound);
app.use(errorHandler);

module.exports = app; // Export app for server.js and testing
```

**`src/server.js`**
```javascript
require('dotenv').config(); // Load .env first
const app = require('./app');
const connectDB = require('./config/db');
const { port } = require('./config/config'); // Centralized config

connectDB(); // Connect to MongoDB

const server = app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
  console.error(`Error: ${err.message}`);
  server.close(() => process.exit(1));
});
```

## 2. Environment Management

Never hardcode sensitive information.

### 2.1. Use `.env` for Configuration and Secrets

Store all configuration and secrets in `.env` files and load them with `dotenv`.

❌ **BAD: Hardcoded secrets**
```javascript
// config.js
const DB_URI = 'mongodb://user:password@localhost:27017/mydb';
const JWT_SECRET = 'supersecretkey';
```

✅ **GOOD: `.env` and `dotenv`**
**`.env`**
```
PORT=5000
MONGO_URI=mongodb://user:password@localhost:27017/mydb
JWT_SECRET=a_very_secure_random_string_for_jwt
```
**`src/config/config.js`**
```javascript
require('dotenv').config();

module.exports = {
  port: process.env.PORT || 3000,
  mongoURI: process.env.MONGO_URI,
  jwtSecret: process.env.JWT_SECRET,
  nodeEnv: process.env.NODE_ENV || 'development',
};
```

## 3. Common Patterns and Anti-patterns

Embrace modern JavaScript and robust patterns.

### 3.1. Always Use `async/await` for Asynchronous Operations

Avoid callback hell and `.then()` chains.

❌ **BAD: Callback hell or `.then()` chains**
```javascript
// userController.js
exports.getUsers = (req, res) => {
  User.find().then(users => {
    res.json(users);
  }).catch(err => {
    res.status(500).json({ message: err.message });
  });
};
```

✅ **GOOD: `async/await` with `express-async-handler`**
Install `npm i express-async-handler`. This eliminates repetitive `try/catch` blocks in controllers.

**`src/controllers/userController.js`**
```javascript
const asyncHandler = require('express-async-handler');
const userService = require('../services/userService');

exports.getUsers = asyncHandler(async (req, res) => {
  const users = await userService.getAllUsers();
  res.json(users);
});

exports.createUser = asyncHandler(async (req, res) => {
  const newUser = await userService.createUser(req.body);
  res.status(201).json(newUser);
});
```

## 4. Security Best Practices

Security is paramount. Implement these from day one.

### 4.1. Validate and Sanitize All User Input

Never trust client-side data. Use libraries like `Joi` or `zod`.

❌ **BAD: No input validation**
```javascript
// userController.js
exports.createUser = asyncHandler(async (req, res) => {
  const newUser = new User(req.body); // Directly use req.body
  await newUser.save();
  res.status(201).json(newUser);
});
```

✅ **GOOD: Input Validation with `Joi`**
Install `npm i joi`.
**`src/validation/userValidation.js`**
```javascript
const Joi = require('joi');

const userSchema = Joi.object({
  username: Joi.string().alphanum().min(3).max(30).required(),
  email: Joi.string().email().required(),
  password: Joi.string().pattern(new RegExp('^[a-zA-Z0-9]{3,30}$')).required(),
});

exports.validateUser = (req, res, next) => {
  const { error } = userSchema.validate(req.body);
  if (error) {
    return res.status(400).json({ message: error.details[0].message });
  }
  next();
};
```
**`src/routes/userRoutes.js`**
```javascript
const router = require('express').Router();
const { createUser } = require('../controllers/userController');
const { validateUser } = require('../validation/userValidation');

router.post('/', validateUser, createUser);
```

### 4.2. Use Security Middleware (`helmet`, `cors`, `express-rate-limit`)

Protect your API from common web vulnerabilities.

**`src/app.js` (already shown, but reiterating)**
```javascript
const express = require('express');
const helmet = require('helmet'); // Sets various HTTP headers for security
const cors = require('cors');     // Enables CORS with configurable options
const rateLimit = require('express-rate-limit'); // Basic rate limiting

const app = express();

app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN || '*', credentials: true })); // Configure origin
app.use(rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again after 15 minutes',
}));
// ... other middleware and routes
```

### 4.3. Secure Cookies

Always set `httpOnly`, `secure`, and `SameSite` attributes for session cookies.

❌ **BAD: Insecure cookies**
```javascript
res.cookie('token', token);
```

✅ **GOOD: Secure cookies**
```javascript
res.cookie('token', token, {
  httpOnly: true, // Prevents client-side JS access
  secure: process.env.NODE_ENV === 'production', // Only send over HTTPS
  sameSite: 'Strict', // Protects against CSRF
  maxAge: 3600000, // 1 hour
});
```

## 5. Error Handling

Centralize and standardize error responses.

### 5.1. Implement Centralized Error Handling Middleware

Catch all errors and return consistent JSON responses.

**`src/middlewares/errorMiddleware.js`**
```javascript
const { nodeEnv } = require('../config/config');

const notFound = (req, res, next) => {
  const error = new Error(`Not Found - ${req.originalUrl}`);
  res.status(404);
  next(error);
};

const errorHandler = (err, req, res, next) => {
  const statusCode = res.statusCode === 200 ? 500 : res.statusCode;
  res.status(statusCode);
  res.json({
    message: err.message,
    stack: nodeEnv === 'production' ? '🥞' : err.stack, // Hide stack in prod
  });
};

module.exports = { notFound, errorHandler };
```
**`src/app.js` (already shown, but reiterating)**
```javascript
// ...
app.use(notFound);
app.use(errorHandler);
// ...
```

## 6. API Design

Design APIs for clarity, consistency, and future growth.

### 6.1. Follow RESTful Principles and Version Your API

Use clear resource-based URLs and HTTP methods. Versioning prevents breaking changes.

❌ **BAD: Inconsistent, unversioned endpoints**
```javascript
app.get('/getAllUsers', /* ... */);
app.post('/addUser', /* ... */);
```

✅ **GOOD: RESTful and Versioned**
```javascript
// src/routes/userRoutes.js
router.get('/', getUsers);     // GET /api/v1/users
router.post('/', createUser);  // POST /api/v1/users
router.get('/:id', getUserById); // GET /api/v1/users/:id
router.put('/:id', updateUser); // PUT /api/v1/users/:id
router.delete('/:id', deleteUser); // DELETE /api/v1/users/:id
```
**`src/app.js`**
```javascript
app.use('/api/v1/users', userRoutes);
```

## 7. Performance Considerations

Optimize for speed and efficiency.

### 7.1. Enable Gzip/Brotli Compression

Reduce response payload size.

Install `npm i compression`.
**`src/app.js`**
```javascript
const express = require('express');
const compression = require('compression'); // Import compression

const app = express();

app.use(compression()); // Use compression middleware early
// ... other middleware and routes
```

## 8. Testing Approaches

Ensure reliability and prevent regressions.

### 8.1. Implement Unit and Integration Tests

Use `Jest` and `Supertest` for comprehensive testing.

Install `npm i --save-dev jest supertest`.
**`package.json`**
```json
{
  "scripts": {
    "test": "jest --detectOpenHandles"
  }
}
```
**`src/tests/user.test.js`**
```javascript
const request = require('supertest');
const app = require('../app'); // Test your app.js directly
const mongoose = require('mongoose');
const User = require('../models/User');
const { mongoURI } = require('../config/config');

beforeAll(async () => {
  await mongoose.connect(mongoURI);
});

afterEach(async () => {
  await User.deleteMany({}); // Clean up after each test
});

afterAll(async () => {
  await mongoose.connection.close();
});

describe('User API', () => {
  it('should create a new user', async () => {
    const res = await request(app)
      .post('/api/v1/users')
      .send({
        username: 'testuser',
        email: 'test@example.com',
        password: 'password123'
      });
    expect(res.statusCode).toEqual(201);
    expect(res.body).toHaveProperty('username', 'testuser');
  });

  it('should fetch all users', async () => {
    await request(app).post('/api/v1/users').send({ username: 'u1', email: 'u1@e.com', password: 'p1' });
    await request(app).post('/api/v1/users').send({ username: 'u2', email: 'u2@e.com', password: 'p2' });

    const res = await request(app).get('/api/v1/users');
    expect(res.statusCode).toEqual(200);
    expect(res.body.length).toEqual(2);
  });
});
```