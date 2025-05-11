---
description: This rule provides a comprehensive guide to nginx configuration best practices, covering code organization, common patterns, performance, security, testing, pitfalls, and tooling.
globs: nginx.conf
---
# nginx Configuration Best Practices

This document outlines best practices for configuring nginx as a web server, reverse proxy, and load balancer. Adhering to these guidelines promotes maintainability, security, and performance.

## 1. Code Organization and Structure

### 1.1 Directory Structure

Maintain a well-organized directory structure to manage configuration files efficiently.


/etc/nginx/
├── nginx.conf        # Main nginx configuration file
├── conf.d/           # Site-specific configurations (symlinks preferred)
│   ├── default.conf    # Default configuration
│   ├── example.com.conf# Site-specific configuration
│   └── ...
├── snippets/         # Reusable configuration snippets
│   ├── ssl-params.conf # SSL parameters
│   ├── proxy-params.conf # Reverse proxy parameters
│   └── ...
└── modules-enabled/  # Enabled modules (usually managed by package manager)


**Explanation:**

*   `/etc/nginx/nginx.conf`:  The main configuration file that includes other configurations.
*   `/etc/nginx/conf.d/`: Contains site-specific configurations.  Symlinking from `/etc/nginx/sites-available/` to `/etc/nginx/sites-enabled/` is a common, more manageable pattern.
*   `/etc/nginx/snippets/`: Stores reusable configuration snippets, promoting modularity and reducing duplication.

### 1.2 File Naming Conventions

Adopt consistent file naming conventions for clarity.

*   Site-specific configurations: `domain.com.conf` or `application-name.conf`.
*   Snippets: `description.conf` (e.g., `ssl-params.conf`, `proxy-params.conf`).

### 1.3 Modular Configuration

Break down complex configurations into smaller, reusable modules (snippets).

**Example: `snippets/ssl-params.conf`**

nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers on;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_ecdh_curve prime256v1;
ssl_session_timeout  10m;
ssl_session_cache shared:SSL:10m;
ssl_session_tickets off;

add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options SAMEORIGIN;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";

ssl_stapling on;
ssl_stapling_verify on;
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;


**Usage in site configuration:**

nginx
server 
    listen 443 ssl;
    server_name example.com;

    include snippets/ssl-params.conf;

    ssl_certificate /etc/nginx/ssl/example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;

    location / {
        # ...
    
}


### 1.4 Component Architecture

Design your nginx configurations with a clear separation of concerns:

*   **Global Settings**: Core directives in `nginx.conf` affecting all virtual hosts.
*   **Virtual Host Configuration**:  Separate files for each site/application in `conf.d/`.
*   **Reusable Blocks**: Snippets for common settings like SSL, reverse proxy configurations, or access control.

### 1.5 Code Splitting

For large and complex configurations, split files by functionality (e.g., routing rules, security policies, caching configurations).  Use `include` directives to assemble the complete configuration.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

*   **Reverse Proxy**:  Forward requests to backend servers. Useful for load balancing, caching, and security.
*   **Load Balancer**: Distribute traffic across multiple backend servers for high availability and scalability. Use `upstream` blocks.
*   **Static Content Server**:  Efficiently serve static files like images, CSS, and JavaScript.
*   **Cache**: Reduce latency and server load by caching frequently accessed content.

### 2.2 Recommended Approaches for Common Tasks

*   **Serving Static Content:**

    nginx
    location /static/ {
        alias /var/www/example.com/static/;
        expires 30d;
        access_log off;
    }
    

*   **Reverse Proxy to a Backend Server:**

    nginx
    location / {
        proxy_pass http://backend_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    upstream backend_server {
        server backend1.example.com;
        server backend2.example.com;
    }
    

*   **Implementing Basic Authentication:**

    nginx
    location /protected/ {
        auth_basic "Restricted Area";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
    

### 2.3 Anti-patterns and Code Smells

*   **Large, Monolithic Configuration Files**: Difficult to manage and debug. Break them into smaller, modular files using `include`.
*   **Duplicated Configuration Blocks**:  Create snippets to reuse common configurations.
*   **Using `if` Statements Excessively**: `if` statements can have performance implications.  Prefer using `try_files` or other more efficient directives.
*   **Insecure Configuration**: Exposing sensitive information, using weak ciphers, or not implementing proper access controls.
*   **Ignoring Error Logs**: Neglecting to monitor and analyze error logs can lead to undetected issues.

### 2.4 State Management

Nginx itself is generally stateless. State management is typically handled by backend applications.  However, Nginx Plus offers features like session persistence (sticky sessions) for load balancing.

*   **Session Persistence (Nginx Plus):**

    nginx
    upstream backend {
        server backend1.example.com;
        server backend2.example.com;
        sticky cookie srv_id expires=1h;
    }
    

### 2.5 Error Handling

*   **Custom Error Pages**: Configure custom error pages for a better user experience.

    nginx
    error_page 404 /404.html;
    location = /404.html {
        root /var/www/example.com/html;
        internal;
    }
    

*   **Logging**:  Use comprehensive logging to capture errors and warnings for troubleshooting.
*   **Retry Logic**: Implement retry logic in your application or use Nginx's `proxy_next_upstream` directive to retry failed requests on different backend servers.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Enable Gzip Compression**: Compress text-based responses to reduce bandwidth usage.

    nginx
gzip on;
gzip_disable "msie6";
gzip_vary on;
gzip_proxied any;
gzip_comp_level 6;
gzip_buffers 16 8k;
gzip_http_version 1.1;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    

*   **Enable Caching**: Cache static content and dynamic responses to reduce server load and improve response times. Use `proxy_cache_path` and `proxy_cache` directives.

    nginx
proxy_cache_path /tmp/nginx_cache levels=1:2 keys_zone=my_cache:10m max_size=10g inactive=60m use_temp_path=off;

server {
    location / {
        proxy_pass http://backend_server;
        proxy_cache my_cache;
        proxy_cache_valid 200 302 1h;
        proxy_cache_valid 404 10m;
        proxy_cache_use_stale error timeout updating invalid_header http_500 http_502 http_503 http_504;
        add_header X-Cache-Status $upstream_cache_status;
    }
}
    

*   **Tune Worker Processes and Connections**: Adjust `worker_processes` and `worker_connections` based on your server's resources.

    nginx
worker_processes auto;
events {
    worker_connections 1024;
}
    

*   **HTTP/2 or HTTP/3**: Enable HTTP/2 or HTTP/3 for improved performance.
*   **Keep-Alive Connections**: Enable keep-alive connections to reduce latency by reusing existing connections.

    nginx
keepalive_timeout 65;
    

### 3.2 Memory Management

*   **Optimize Buffer Sizes**: Tune buffer sizes (e.g., `proxy_buffer_size`, `proxy_buffers`) to match your application's needs and avoid memory fragmentation.
*   **Monitor Memory Usage**: Use tools like `top` or `htop` to monitor nginx's memory usage and identify potential memory leaks.

### 3.3 Rendering Optimization (If applicable)

Nginx primarily serves content; rendering optimization is typically handled by the backend application.

*   **Content Delivery Network (CDN)**: Offload static content delivery to a CDN.

### 3.4 Bundle Size Optimization (If applicable)

This primarily applies to serving static assets.

*   **Minify and Compress**: Minify and compress CSS and JavaScript files before serving them.
*   **Use Brotli**: Consider using Brotli compression, which offers better compression ratios than Gzip.

### 3.5 Lazy Loading (If applicable)

This is typically implemented at the application level.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and Prevention

*   **SQL Injection**: Prevent SQL injection by validating and sanitizing user inputs at the application level.
*   **Cross-Site Scripting (XSS)**: Implement proper output encoding and use HTTP headers like `X-XSS-Protection` and `Content-Security-Policy`.

    nginx
add_header X-XSS-Protection "1; mode=block";
add_header Content-Security-Policy "default-src 'self'";
    

*   **Clickjacking**: Prevent clickjacking by setting the `X-Frame-Options` header.

    nginx
add_header X-Frame-Options "SAMEORIGIN";
    

*   **Buffer Overflow**: Keep Nginx up to date to patch vulnerabilities related to buffer overflows.
*   **Denial of Service (DoS)**: Implement rate limiting to prevent DoS attacks.

    nginx
limit_req_zone $binary_remote_addr zone=mylimit:10m rate=1r/s;

location / {
    limit_req zone=mylimit burst=5 nodelay;
    # ...
}
    

### 4.2 Input Validation

*   **Validate User Inputs**: Validate all user inputs at the application level to prevent malicious data from reaching backend servers.
*   **Limit Request Size**: Limit the size of incoming requests to prevent buffer overflows and DoS attacks.

    nginx
client_max_body_size 2m;
    

### 4.3 Authentication and Authorization

*   **Basic Authentication**: Use basic authentication for simple access control. However, it's not secure over HTTP; always use HTTPS.
*   **JWT (JSON Web Tokens)**: Implement JWT-based authentication for APIs.
*   **OAuth 2.0**: Use OAuth 2.0 for delegated authorization.
*   **Access Control Lists (ACLs)**: Use ACLs to control access to specific resources based on IP addresses or other criteria.

    nginx
allow 192.168.1.0/24;
deny all;
    

### 4.4 Data Protection

*   **HTTPS**: Use HTTPS to encrypt all traffic between clients and the server. Obtain and configure SSL/TLS certificates.
*   **Secure Storage**: Store sensitive data (e.g., passwords, API keys) securely using encryption and access controls.

### 4.5 Secure API Communication

*   **API Keys**: Require API keys for all API requests.
*   **Rate Limiting**: Implement rate limiting to prevent abuse and DoS attacks.
*   **Input Validation**: Validate all API request parameters.
*   **Output Encoding**: Encode API responses to prevent XSS attacks.

## 5. Testing Approaches

### 5.1 Unit Testing

Nginx configuration files are difficult to unit test directly.  Unit testing is more relevant for custom Nginx modules written in C.

### 5.2 Integration Testing

*   **Test Configuration Syntax**: Use `nginx -t` to test the syntax of your configuration files.
*   **Test with Docker**: Use Docker containers to create isolated test environments.
*   **Test with Real Traffic**: Use tools like `ab` (Apache Bench) or `wrk` to simulate real traffic and measure performance.

### 5.3 End-to-End Testing

Use tools like Selenium or Cypress to automate end-to-end tests of your application.

### 5.4 Test Organization

*   **Separate Test Environment**: Create a separate test environment that mirrors your production environment.
*   **Automated Testing**: Automate your testing process using CI/CD pipelines.

### 5.5 Mocking and Stubbing

*   **Mock Backend Services**: Mock backend services to isolate your Nginx configuration and test different scenarios.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

*   **Incorrect Syntax**: Nginx configuration syntax can be tricky. Always test your configuration files using `nginx -t`.
*   **Missing Semicolons**: Forgetting semicolons at the end of directives.
*   **Incorrect File Permissions**: Ensure that Nginx has the necessary permissions to read configuration files and access static content.
*   **Not Reloading Configuration**: Forgetting to reload Nginx after making changes to the configuration files (e.g., `nginx -s reload`).
*   **Using `if` Statements Inappropriately**: Overusing `if` can lead to unexpected behavior and performance issues.  Prefer `try_files` or other alternatives.

### 6.2 Edge Cases

*   **Handling Large Files**: Tune buffer sizes and timeouts when handling large file uploads or downloads.
*   **WebSockets**: Configure Nginx to properly proxy WebSocket connections.
*   **Long URIs**: Configure `client_header_buffer_size` and `large_client_header_buffers` to handle long URIs.

### 6.3 Version-Specific Issues

*   **Check Release Notes**: Review the release notes for each Nginx version to identify any breaking changes or known issues.
*   **Test Upgrades**: Test Nginx upgrades in a staging environment before deploying to production.

### 6.4 Compatibility Concerns

*   **Backend Application Compatibility**: Ensure that your Nginx configuration is compatible with the backend applications you are using.
*   **TLS Versions and Ciphers**: Choose TLS versions and ciphers that are compatible with your clients and backend servers.

### 6.5 Debugging Strategies

*   **Error Logs**: Check Nginx error logs (`/var/log/nginx/error.log`) for errors and warnings.
*   **Access Logs**: Analyze Nginx access logs (`/var/log/nginx/access.log`) to understand traffic patterns and identify potential issues.
*   **`nginx -t`**: Use `nginx -t` to test the syntax of your configuration files.
*   **Verbose Logging**: Enable verbose logging to capture more detailed information about requests and responses.
*   **tcpdump/Wireshark**: Use `tcpdump` or Wireshark to capture and analyze network traffic.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **Text Editor**: Use a text editor with syntax highlighting and autocompletion for Nginx configuration files (e.g., VS Code with Nginx configuration extension).
*   **Command-Line Tools**: Use command-line tools like `nginx`, `curl`, `ab`, and `tcpdump` for testing and debugging.
*   **Docker**: Use Docker for creating isolated test environments.

### 7.2 Build Configuration

*   **Version Control**: Store your Nginx configuration files in a version control system (e.g., Git).
*   **Configuration Management**: Use configuration management tools like Ansible or Chef to automate the deployment and management of your Nginx configuration.

### 7.3 Linting and Formatting

*   **Nginx Linter**: Use an Nginx linter to check your configuration files for syntax errors and best practices violations (e.g., `nginx -t`).
*   **Code Formatting**: Use a code formatter to ensure consistent formatting of your configuration files.

### 7.4 Deployment Best Practices

*   **Blue-Green Deployment**: Use blue-green deployment to minimize downtime during deployments.
*   **Canary Deployment**: Use canary deployment to test new Nginx configurations with a small subset of users before rolling them out to the entire user base.
*   **Rollback Strategy**: Have a rollback strategy in place in case of deployment failures.

### 7.5 CI/CD Integration

*   **Automated Testing**: Integrate automated testing into your CI/CD pipeline.
*   **Configuration Validation**: Validate Nginx configuration files as part of your CI/CD pipeline.
*   **Automated Deployment**: Automate the deployment of Nginx configuration files using CI/CD tools like Jenkins, GitLab CI, or CircleCI.

By following these best practices, you can create a robust, secure, and performant Nginx configuration that meets the needs of your web applications.