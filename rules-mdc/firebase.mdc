---
description: This guide provides definitive, opinionated best practices for building secure, performant, and maintainable applications with Firebase, focusing on modern patterns and avoiding common pitfalls.
globs: **/*
---
# firebase Best Practices

Firebase is a powerful Backend-as-a-Service (BaaS) that accelerates development. To leverage it effectively, you must adhere to strict best practices for security, performance, and maintainability. This guide outlines the definitive approach for our team.

## 1. Security Best Practices

Security is paramount. Assume all client-side code is compromised and enforce access control exclusively through Firebase Security Rules and App Check.

### 1.1 Enable Firebase App Check

Always enable App Check for all supported Firebase services. This verifies that requests originate from your authentic app and an untampered device, preventing unauthorized clients from accessing your backend.

**✅ GOOD: Enable App Check**
Configure App Check in the Firebase Console for Firestore, Realtime Database, Storage, and Functions. On the client, initialize it early.

```typescript
// client/src/firebase.ts
import { initializeApp } from 'firebase/app';
import { initializeAppCheck, ReCaptchaV3Provider } from 'firebase/app-check';
// For mobile, use platform-specific providers (e.g., ReactNativeFirebaseAppCheckProvider)

const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_