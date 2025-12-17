---
description: This guide defines the definitive NGINX configuration best practices for our team, focusing on modularity, security, performance, and maintainability.
globs: **/*
---
# nginx Best Practices

NGINX is our go-to for web serving, reverse proxying, and load balancing. This guide outlines the mandatory best practices for NGINX configurations, ensuring readability, security, and optimal performance across all our deployments.

## Critical Guidelines:

### 1. Code Organization and Structure

Always break down your NGINX configuration into small, focused, and reusable `include` files. This significantly improves maintainability and reduces cognitive load.

**✅ GOOD: Modular Configuration**

```nginx
# /etc/nginx/nginx.conf
user  nginx;
worker_processes  auto;

error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    tcp_nopush      on;
    tcp_nodelay     on;
    keepalive_timeout  65;

    # Global HTTP settings
    include /etc/nginx/conf.d/http_common.conf;
    # Upstream server definitions
    include /etc/nginx/conf.d/upstreams/*.conf;
    # SSL/TLS profiles
    include /etc/nginx/conf.d/ssl_profiles/*.conf;
    # Virtual hosts
    include /etc/nginx/sites-enabled/*.conf;
}
```

**❌ BAD: Monolithic Configuration**

```nginx
# /etc/nginx/nginx.conf (too long, hard to navigate)
# ... all server blocks, upstream definitions, SSL settings in one file ...
server {
    listen 80;
    server_name example.com;
    # ... hundreds of lines ...
}
upstream backend_app {
    # ... more lines ...
}
# ... SSL config ...
```

### 2. Consistent Formatting and Comments

Use two-space indentation. Add descriptive comments for every directive or block that isn't immediately obvious.

**✅ GOOD**

```nginx
# /etc/nginx/sites-enabled/api.conf
server {
  listen 80;
  listen 443 ssl http2; # Enforce HTTPS and HTTP/2
  server_name api.example.com;

  # Include standard SSL settings
  include /etc/nginx/conf.d/ssl_profiles/default_ssl.conf;

  location / {
    proxy_pass http://api_backend; # Proxy to the API upstream
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_redirect off;
  }
}
```

**❌ BAD**

```nginx
server {
listen 80;
listen 443 ssl http2;
server_name api.example.com;
include /etc/nginx/conf.d/ssl_profiles/default_ssl.conf;
location / {
proxy_pass http://api_backend;
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_redirect off;
}
} # No comments, inconsistent indentation
```

### 3. Security Hardening

Always enable modern TLS, HTTP/2, strong cipher suites, and essential security headers.

**✅ GOOD: Secure SSL Profile (`/etc/nginx/conf.d/ssl_profiles/default_ssl.conf`)**

```nginx
ssl_protocols TLSv1.2 TLSv1.3; # Only modern, secure protocols
ssl_prefer_server_ciphers on;
ssl_ciphers "ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384";
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 1h;
ssl_session_tickets off;
ssl_dhparam /etc/nginx/ssl/dhparam.pem; # Generate with `openssl dhparam -out dhparam.pem 2048`

# Essential security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
```

**❌ BAD: Weak SSL/Missing Headers**

```nginx
ssl_protocols TLSv1 TLSv1.1 TLSv1.2; # Outdated protocols
ssl_ciphers HIGH:!aNULL:!MD5; # Too broad, potentially weak
# No security headers
```

### 4. Load Balancing and Upstream Configuration

Define upstream groups with explicit `weight`, `max_fails`, and `fail_timeout`. Use `proxy_protocol` and `RealIP` when behind another proxy/load balancer.

**✅ GOOD: Robust Upstream (`/etc/nginx/conf.d/upstreams/api_backend.conf`)**

```nginx
upstream api_backend {
  # Round-robin load balancing by default
  server 10.0.0.1:8080 weight=5 max_fails=3 fail_timeout=10s;
  server 10.0.0.2:8080 weight=5 max_fails=3 fail_timeout=10s;
  # NGINX Plus: Enable health checks
  # health_check interval=5s passes=1 fails=3;
}

# In http block or server block that accepts PROXY protocol
http {
  # ...
  server {
    listen 80   proxy_protocol;
    listen 443  ssl proxy_protocol;
    # ...
    # Set the real client IP from the PROXY protocol header
    set_real_ip_from 192.168.1.0/24; # IP range of your upstream load balancer
    real_ip_header proxy_protocol;
    # ...
  }
}
```

**❌ BAD: Basic Upstream / Missing RealIP**

```nginx
upstream api_backend {
  server 10.0.0.1:8080; # No weight, fail_timeout, max_fails
  server 10.0.0.2:8080;
}
# No proxy_protocol or RealIP configuration, logs show proxy IP
```

### 5. Performance Optimization

Leverage caching, compression, and resource limits to ensure high-throughput and low-latency.

**✅ GOOD: Caching, Compression, and Limits**

```nginx
http {
  # ...
  # Caching for static assets or API responses
  proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m inactive=60m max_size=1g;

  server {
    # ...
    location /static/ {
      root /var/www/html;
      expires 30d;
      add_header Cache-Control "public, immutable";
    }

    location /api/ {
      proxy_cache my_cache;
      proxy_cache_valid 200 302 10m;
      proxy_cache_valid 404      1m;
      proxy_cache_revalidate on;
      proxy_cache_min_uses 1;
      proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
      proxy_cache_background_update on;
      proxy_cache_lock on;
      proxy_cache_lock_timeout 5s;
      proxy_cache_lock_age 5s;
      proxy_cache_bypass $http_pragma; # Bypass cache for specific requests
      add_header X-Cache-Status $upstream_cache_status;
      proxy_pass http://api_backend;
    }

    # Enable Gzip compression for text-based content
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_buffers 16 8k;
    gzip_http_version 1.1;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Rate limiting (per IP, 10 requests per second burstable by 20)
    limit_req_zone $binary_remote_addr zone=mylimit:10m rate=10r/s;
    limit_req zone=mylimit burst=20 nodelay;

    # Connection limits (10 connections per IP)
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
    limit_conn conn_limit 10;

    # Max request body size (10MB)
    client_max_body_size 10M;
  }
}
```

**❌ BAD: No Caching, No Compression, No Limits**

```nginx
# No proxy_cache_path defined
server {
  # ...
  location /static/ {
    root /var/www/html;
    # No expires, no Cache-Control
  }
  # No gzip directives
  # No rate limiting or connection limits
  # No client_max_body_size, vulnerable to large body attacks
}
```

### 6. Common Pitfalls and Gotchas

**a. `location` Block Order:** Specific (exact, regex) locations must come before general (prefix) ones.

**❌ BAD**

```nginx
location / { # This catches everything first
  proxy_pass http://default_backend;
}
location = /specific { # This will never be hit
  return 200 "Specific page";
}
```

**✅ GOOD**

```nginx
location = /specific { # Exact match first
  return 200 "Specific page";
}
location ~ \.php$ { # Regex matches next
  fastcgi_pass unix:/var/run/php/php-fpm.sock;
  # ...
}
location / { # General prefix match last
  proxy_pass http://default_backend;
}
```

**b. Missing `resolver` for Upstream Hostnames:** If your upstream servers are defined by hostnames (not IPs), NGINX needs a DNS resolver.

**❌ BAD**

```nginx
upstream dynamic_backend {
  server app.internal.svc.cluster.local:8080; # Will fail to resolve without resolver
}
```

**✅ GOOD**

```nginx
http {
  resolver 10.0.0.10 valid=30s; # Use your internal DNS server, cache for 30s
  resolver_timeout 5s;

  upstream dynamic_backend {
    server app.internal.svc.cluster.local:8080;
  }
}
```

### 7. Testing Approaches

Always validate your NGINX configuration before reloading or restarting the service.

**✅ GOOD: Configuration Syntax Check**

```bash
sudo nginx -t
```

This command checks the syntax of your configuration files and attempts to open files referenced by `include` directives. If successful, it will output:

```
nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

**❌ BAD: Deploying without `nginx -t`**

Never deploy or reload NGINX without first running `nginx -t`. A single syntax error can bring down your entire web service.

For more advanced testing, implement integration tests that hit your NGINX endpoints and verify expected behavior (e.g., correct routing, headers, content). Consider using NGINX Amplify for continuous configuration analysis and performance monitoring.