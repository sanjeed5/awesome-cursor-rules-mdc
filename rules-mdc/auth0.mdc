---
description: Definitive guidelines for integrating Auth0 securely and efficiently, preventing common identity attacks, and leveraging modern authentication patterns.
globs: **/*
---
# auth0 Best Practices

This guide outlines the definitive best practices for integrating Auth0 into our applications. Adhering to these standards is critical for maintaining a robust security posture against modern identity threats.

## 1. Prioritize Universal Login

Always centralize authentication flows using Auth0 Universal Login. This offloads complex security logic, ensures consistency, and leverages Auth0's built-in security features across all applications (web, mobile, AI agents).

❌ **BAD: Implementing custom login forms or embedded login**
This introduces significant security risks and maintenance overhead.

✅ **GOOD: Redirecting to Universal Login with Auth0 SDKs**

```javascript
// Example for a React/SPA using @auth0/auth0-react (built on auth0-spa-js)
import { useAuth0 } from '@auth0/auth0-react';

function LoginButton() {
  const { loginWithRedirect } = useAuth0();

  const handleLogin = () => {
    loginWithRedirect({
      appState: { targetUrl: window.location.pathname },
      authorizationParams: {
        redirect_uri: window.location.origin + '/callback',
        scope: 'openid profile email', // Request only necessary scopes
      },
    });
  };

  return <button onClick={handleLogin}>Log In</button>;
}
```

## 2. Enforce Authorization Code Flow with PKCE

For all public clients (SPAs, mobile apps, native desktop apps), the Authorization Code Flow with Proof Key for Code Exchange (PKCE) is the *only* acceptable grant type. This prevents authorization code injection attacks. **Never use Implicit Grant or Resource Owner Password Grant (ROPG).**

❌ **BAD: Using Implicit Grant or ROPG**
These flows are deprecated and highly vulnerable.

```javascript
// ❌ BAD: Implicit Grant (responseType: 'token id_token')
// Avoid configuring your Auth0 application to allow this.

// ❌ BAD: ROPG (sending username/password directly to /oauth/token)
// Never trust the client with user credentials.
```

✅ **GOOD: Authorization Code Flow with PKCE**
Ensure your Auth0 application is configured for PKCE, and your SDK handles the `code_challenge` and `code_verifier`.

```go
// Example for a Go backend (handling Auth0 callback)
package main

import (
	"context"
	"log"
	"net/http"
	"github.com/auth0/go-auth0/v2/authentication"
	"github.com/auth0/go-auth0/v2/authentication/oauth"
)

// authAPI and tokenValidator are initialized once at application startup (see below)
var authAPI *authentication.Authentication

// init function (or similar setup) for authAPI
func initAuthAPI() {
	var err error
	authAPI, err = authentication.New(
		context.Background(),
		"example.us.auth0.com", // Replace with your Auth0 tenant domain
		authentication.WithClientID("YOUR_CLIENT_ID"),
		authentication.WithClientSecret("YOUR_CLIENT_SECRET"), // For backend apps
	)
	if err != nil {
		log.Fatalf("Failed to initialize Auth0 authentication API client: %+v", err)
	}
}

func handleAuth0Callback(w http.ResponseWriter, r *http.Request) {
	code := r.URL.Query().Get("code")
	// In a real app, codeVerifier must be retrieved from a secure, short-lived storage
	// (e.g., session) associated with the user's browser session.
	codeVerifier := r.Context().Value("code_verifier").(string) // Placeholder

	tokenSet, err := authAPI.OAuth.LoginWithAuthCodeWithPKCE(r.Context(), oauth.LoginWithAuthCodeWithPKCERequest{
		Code:         code,
		CodeVerifier: codeVerifier,
		RedirectURI:  "https://your-app.com/callback", // Must match configured callback URL
	}, oauth.IDTokenValidationOptions{})
	if err != nil {
		log.Printf("Failed to exchange code for tokens: %v", err)
		http.Error(w, "Authentication failed", http.StatusInternalServerError)
		return
	}
	// ✅ GOOD: Store tokenSet securely (e.g., in an HTTP-only, Secure cookie for web apps)
	// Do NOT expose tokens to client-side JavaScript unnecessarily.
	log.Printf("Successfully authenticated user: %s", tokenSet.IDToken)
	http.Redirect(w, r, "/dashboard", http.StatusFound)