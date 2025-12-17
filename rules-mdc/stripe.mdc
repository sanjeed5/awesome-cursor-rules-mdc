---
description: Definitive guidelines for building robust, secure, and efficient Stripe integrations using modern API practices (v2, Checkout Sessions, Payment Intents, Elements) and essential safeguards like idempotency and webhook verification.
globs: **/*
---
# Stripe Best Practices

This guide outlines the definitive best practices for integrating Stripe into our applications. Adhere to these rules to ensure secure, performant, and maintainable payment flows. Always target the latest Stripe API version (`2025-12-15.clover` as of this writing) and keep SDKs up-to-date.

## 1. Code Organization & Structure

### 1.1. Secure API Key Management
Never hardcode API keys. Store them securely in environment variables.

❌ BAD:
```javascript
const stripe = require('stripe')('sk_test_YOUR_SECRET_KEY');
```

✅ GOOD:
```javascript
// .env
// STRIPE_SECRET_KEY=sk_test_YOUR_SECRET_KEY
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
```

### 1.2. Stripe Client Initialization
Initialize the Stripe client once per process or module. Avoid re-initializing on every request to prevent resource overhead.

✅ GOOD (Node.js example):
```javascript
// lib/stripeClient.js
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY, {
  apiVersion: '2025-12-15', // Pin to a specific API version
});
module.exports = stripe;

// In your service/route handler:
const stripe = require('../lib/stripeClient');
// ... use stripe for all API calls
```

## 2. Common Patterns & Anti-patterns

### 2.1. Modern Checkout Flows
Always use Stripe Checkout Sessions or Payment Intents with Stripe Elements for UI. These handle complex authentication, multi-currency, and compliance automatically, significantly reducing your PCI scope.

❌ BAD: Building custom card input fields and handling raw card data directly on your server.
```javascript
// Avoid handling raw card data on your server
const charge = await stripe.charges.create({ /* ... */ });
```

✅ GOOD: Use Checkout Sessions for pre-built, hosted pages, or Payment Intents + Elements for custom UI.
```javascript
// Server-side (Checkout Session example)
const session = await stripe.checkout.sessions.create({
  line_items: [{ price: 'price_123', quantity: 1 }],
  mode: 'payment',
  success_url: 'https://example.com/success',
  cancel_url: 'https://example.com/cancel',
});
// Redirect user to session.url
```
```javascript
// Server-side (Payment Intent example for custom UI)
const paymentIntent = await stripe.paymentIntents.create({
  amount: 1000,
  currency: 'usd',
  automatic_payment_methods: { enabled: true },
});
// Send paymentIntent.client_secret to frontend to use with Stripe Elements
```

### 2.2. Idempotency for Write Operations
Always include a unique `Idempotency-Key` header for all write requests (e.g., creating charges, customers, refunds). This prevents duplicate operations during network retries.

❌ BAD:
```javascript
await stripe.paymentIntents.create({ amount: 1000, currency: 'usd' });
```

✅ GOOD: Generate a stable, unique key per request (e.g., UUID, or a hash of request parameters + user ID).
```javascript
const { v4: uuidv4 } = require('uuid'); // npm install uuid
// ...
await stripe.paymentIntents.create(
  { amount: 1000, currency: 'usd' },
  { idempotencyKey: uuidv4() } // Use a stable key for retries
);
```

### 2.3. Metadata for Traceability
Attach relevant metadata to Stripe objects (charges, customers, Payment Intents) for easier debugging, reconciliation, and integrating with internal business logic.

✅ GOOD:
```javascript
await stripe.paymentIntents.create({
  amount: 2000,
  currency: 'usd',
  metadata: {
    order_id: 'ORDER-XYZ-123',
    user_id: 'user_abc',
    product_sku: 'PROD-001',
  },
});
```

### 2.4. API Versioning
Pin your API version in the Stripe client initialization and for webhook endpoints. Regularly upgrade to benefit from new features and security patches, testing thoroughly.

❌ BAD: Relying on Stripe's default API version, which might change unexpectedly.
```javascript
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY); // No apiVersion specified
```

✅ GOOD: Explicitly set the `apiVersion`.
```javascript
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY, {
  apiVersion: '2025-12-15', // Pin to the latest stable version
});
// Ensure webhook endpoints also use this version in the Dashboard
```

## 3. Error Handling

### 3.1. Catch Specific Stripe Errors
Handle different `StripeError` subclasses to provide precise feedback to users and implement appropriate retry logic. Always log the `requestId`.

✅ GOOD:
```javascript
try {
  await stripe.paymentIntents.confirm(paymentIntentId);
} catch (e) {
  if (e.type === 'StripeCardError') {
    console.error(`Card declined: ${e.message} (Code: ${e.code}, Req ID: ${e.requestId})`);
    // Inform user, prompt for new card
  } else if (e.type === 'StripeInvalidRequestError') {
    console.error(`Invalid request: ${e.message} (Param: ${e.param}, Req ID: ${e.requestId})`);
    // Log, alert dev team for code fix
  } else {
    console.error(`Generic Stripe error: ${e.message} (Req ID: ${e.requestId})`);
    // Log, alert dev team, potentially retry for transient errors
  }
}
```

### 3.2. Webhook-Driven Error Handling
For asynchronous events (e.g., payment failures after initial confirmation), rely on webhooks. Your server should be the source of truth for payment status updates.

✅ GOOD:
```javascript
// In your webhook handler for 'payment_intent.payment_failed'
app.post('/webhook', async (req, res) => { /* ... signature verification ... */
  if (event.type === 'payment_intent.payment_failed') {
    const paymentIntent = event.data.object;
    console.error(`Payment Intent ${paymentIntent.id} failed: ${paymentIntent.last_payment_error?.message}`);
    // Update order status in DB, notify customer, trigger