---
description: This guide defines definitive best practices for writing, organizing, and securing Kubernetes manifests and Operators, ensuring maintainable, performant, and reliable cloud-native deployments.
globs: **/*.{yaml,yml}
---
# kubernetes Best Practices

Kubernetes manifests are your application's blueprint. Treat them as code. These rules ensure your deployments are secure, stable, and easy to manage.

## Code Organization and Structure

Organize your Kubernetes manifests for clarity and maintainability. Avoid monolithic files.

### 1. One Resource Per File (or Logical Group)

Keep files focused. Group related resources (e.g., Deployment, Service, HPA for a single microservice) into a dedicated directory, but separate individual resource types into their own files within that directory.

❌ **BAD:** Monolithic file for an entire application.

```yaml
# app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
spec:
  # ...
---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  # ...
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
spec:
  # ...
```

✅ **GOOD:** Separate files for each resource type, grouped by application.

```bash
# Directory structure
my-app/
├── deployment.yaml
├── service.yaml
└── ingress.yaml
```

```yaml
# my-app/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
  labels:
    app: my-app
spec:
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        ports:
        - containerPort: 8080
```

### 2. Consistent Labeling Strategy

Labels are critical for selection, organization, and automation. Use a consistent set of labels across all resources.

✅ **GOOD:** Standardized labels for `app`, `tier`, `environment`, `version`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
  labels:
    app.kubernetes.io/name: auth-service
    app.kubernetes.io/instance: auth-service-prod
    app.kubernetes.io/version: "1.0.0"
    app.kubernetes.io/component: backend
    app.kubernetes.io/part-of: my-application
    app.kubernetes.io/managed-by: kustomize
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: auth-service
      app.kubernetes.io/instance: auth-service-prod
  template:
    metadata:
      labels:
        app.kubernetes.io/name: auth-service
        app.kubernetes.io/instance: auth-service-prod
    # ...
```

## Common Patterns and Anti-patterns

Adopt proven patterns and rigorously avoid anti-patterns.

### 3. Immutable Image Tags

Always use immutable, versioned image tags. Never use `latest` in production.

❌ **BAD:** Non-deterministic deployments.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:latest # ❌ Don't use 'latest'
```

✅ **GOOD:** Reproducible deployments.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3-abcd123 # ✅ Use specific, immutable tags
        imagePullPolicy: IfNotPresent # Or Always for dev environments
```

### 4. Explicit Resource Requests and Limits

Define CPU and memory requests and limits for all containers. This prevents resource starvation and ensures fair scheduling.

❌ **BAD:** Unbounded resource consumption, leading to instability.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        # ❌ No resource requests/limits
```

✅ **GOOD:** Stable, predictable resource usage.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### 5. Health Probes (Liveness and Readiness)

Implement Liveness and Readiness probes for all application containers.

❌ **BAD:** Unresponsive or unhealthy pods remain in service.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        ports:
        - containerPort: 8080
        # ❌ No probes defined
```

✅ **GOOD:** Automatic healing and graceful degradation.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Security Considerations

Security is paramount. Enforce the Pod Security Standards and principle of least privilege.

### 6. Pod Security Standards (PSS)

Apply the `restricted` Pod Security Standard to all application namespaces. For system components, use `baseline`. Avoid `privileged` unless absolutely critical and justified.

❌ **BAD:** Default, permissive `PodSpec` allows root and host access.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        securityContext: {} # ❌ Default, permissive
```

✅ **GOOD:** Enforce `restricted` PSS via Namespace labels and explicit `securityContext`.

```yaml
# Namespace definition (ensure this is applied to your namespace)
apiVersion: v1
kind: Namespace
metadata:
  name: my-app-namespace
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/audit: restricted
```

```yaml
# deployment.yaml
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000 # Use a non-root user ID
        fsGroup: 1000
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"] # Drop all capabilities
          readOnlyRootFilesystem: true # Make root filesystem read-only
          seccompProfile:
            type: RuntimeDefault # Use the default seccomp profile
```

### 7. Service Accounts and RBAC

Always define a dedicated `ServiceAccount` for your deployments and grant it only the necessary permissions via `RoleBinding` and `Role`. Never use the `default` service account.

❌ **BAD:** Using the `default` service account with elevated permissions.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      # ❌ No serviceAccountName specified, uses 'default'
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
```

✅ **GOOD:** Dedicated service account with minimal RBAC.

```yaml
# serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app-sa
  namespace: my-app-namespace
---
# role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: my-app-role
  namespace: my-app-namespace
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"] # Only necessary permissions
---
# rolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: my-app-rb
  namespace: my-app-namespace
subjects:
- kind: ServiceAccount
  name: my-app-sa
  namespace: my-app-namespace
roleRef:
  kind: Role
  name: my-app-role
  apiGroup: rbac.authorization.k8s.io
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-deployment
  namespace: my-app-namespace
spec:
  template:
    spec:
      serviceAccountName: my-app-sa # ✅ Use dedicated service account
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
```

## Configuration Management

Manage configuration external to your application images.

### 8. ConfigMaps for Non-Sensitive Data

Use `ConfigMap` for non-sensitive configuration data. Mount them as files or inject them as environment variables.

❌ **BAD:** Hardcoding configuration in the Deployment manifest.

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        env:
        - name: API_URL
          value: "http://my-api-service:8080" # ❌ Hardcoded
```

✅ **GOOD:** Externalized configuration via ConfigMap.

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-app-config
  namespace: my-app-namespace
data:
  API_URL: "http://my-api-service:8080"
  LOG_LEVEL: "INFO"
---
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        envFrom:
        - configMapRef:
            name: my-app-config # ✅ Inject all ConfigMap data as env vars
```

### 9. Secrets for Sensitive Data

Use `Secret` for sensitive data (API keys, database credentials). Mount them as files or inject as environment variables, but prefer file mounts for better security. Consider tools like Sealed Secrets or Vault for encrypting secrets at rest in Git.

❌ **BAD:** Storing sensitive data directly in ConfigMaps or plain text.

```yaml
# configmap.yaml
data:
  DB_PASSWORD: "mysecretpassword" # ❌ Never store secrets in ConfigMaps
```

✅ **GOOD:** Securely manage secrets.

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-app-secret
  namespace: my-app-namespace
type: Opaque
data:
  DB_PASSWORD: "bXlzZWNyZXRwYXNzd29yZA==" # ✅ Base64 encoded, but still plaintext in Git without encryption
---
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        volumeMounts:
        - name: secret-volume
          mountPath: "/etc/secrets"
          readOnly: true
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: my-app-secret
              key: DB_PASSWORD # ✅ Inject specific secret key as env var
      volumes:
      - name: secret-volume
        secret:
          secretName: my-app-secret # ✅ Mount secret as files
```

## Logging and Monitoring

Ensure applications are observable.

### 10. Log to `stdout`/`stderr`

Containers must log to `stdout` and `stderr`. Kubernetes handles log collection, forwarding them to your cluster's logging solution.

❌ **BAD:** Logging to files inside the container.

```dockerfile
# Dockerfile
CMD ["/app/start.sh"] # start.sh writes logs to /var/log/app.log
```

✅ **GOOD:** Standard output logging.

```dockerfile
# Dockerfile
CMD ["/app/start.sh"] # start.sh writes logs to stdout/stderr
```

```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
      - name: my-app
        image: myregistry/my-app:v1.2.3
        # ✅ Logs automatically collected from stdout/stderr
```

## Testing and Validation

Automate validation of your manifests.

### 11. Lint and Validate Manifests in CI/CD

Integrate tools like `kube-linter`, `kube-score`, `kube-val`, or OPA Gatekeeper/Kyverno into your CI/CD pipeline to validate manifests against best practices and policies before deployment.

❌ **BAD:** Deploying manifests without automated checks.

```bash
# CI/CD pipeline step
kubectl apply -f manifests/ # ❌ No validation, deploys potentially problematic YAML
```

✅ **GOOD:** Automated validation prevents common errors.

```bash
# CI/CD pipeline step
kube-linter lint manifests/
kube-score score manifests/
# For policy enforcement:
# conftest test -p policies/ manifests/
# ✅ Fails fast if manifests violate rules
kubectl apply -f manifests/
```