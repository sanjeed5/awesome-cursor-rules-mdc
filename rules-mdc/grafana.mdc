---
description: This guide defines definitive best practices for developing, structuring, and maintaining Grafana dashboards, alerts, and data sources as code, ensuring readability, performance, and scalability across teams.
globs: **/*
---
# Grafana Best Practices

Our team treats Grafana dashboards, alerts, and data sources as first-class code artifacts. This means applying software engineering principles: version control, peer review, automated testing, and CI/CD. Adhere strictly to these guidelines to ensure our observability platform remains robust, maintainable, and actionable.

## 1. Code Organization and Structure

Define all Grafana resources declaratively using Observability as Code (OaC). This enables Git-based workflows, automated provisioning, and consistent deployments.

### 1.1. Declarative Configuration (OaC)

**Always** define dashboards, alerts, and data sources in JSON/YAML files. Store these in Git and provision them via Grafana's API. Avoid manual UI configuration for anything beyond initial prototyping.

❌ **BAD**: Manual UI changes, no version history.
✅ **GOOD**: Git-versioned JSON/YAML files, automated provisioning.

```json
// my-service-dashboard.json
{
  "apiVersion": 1,
  "title": "My Service Overview",
  "uid": "my-service-overview",
  "panels": [
    // ... panel definitions ...
  ],
  "templating": {
    "list": [
      // ... variables ...
    ]
  }
}
```

### 1.2. Utilize Foundation SDKs

For complex or dynamically generated dashboards, **prefer** using Grafana Foundation SDKs (Go, Python, TypeScript) to programmatically define resources. This provides strong typing, validation, and better code reusability than raw JSON manipulation.

```typescript
// dashboard.ts (using TypeScript SDK)
import { Dashboard, Row, Panel, GraphPanel } from '@grafana/sdk';

const dashboard = new Dashboard({
  title: 'Service Health',
  uid: 'service-health',
  panels: [
    new Row({
      panels: [
        new GraphPanel({
          title: 'CPU Utilization',
          targets: [{ expr: 'sum(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)' }],
          yAxis: { min: 0 }
        })
      ]
    })
  ]
});

console.log(JSON.stringify(dashboard.toGrafanaJson(), null, 2));
```

### 1.3. Templated Variables

**Mandate** the use of templated variables for dynamic filtering and reusability across environments or instances. This prevents dashboard duplication.

❌ **BAD**: Hardcoded instance names or environment-specific queries.
```json
// Duplicated dashboard for 'prod' and 'dev'
{ "title": "Prod CPU Usage", "targets": [{ "expr": "node_cpu_seconds_total{instance='prod-app-01'}" }] }
{ "title": "Dev CPU Usage", "targets": [{ "expr": "node_cpu_seconds_total{instance='dev-app-01'}" }] }
```
✅ **GOOD**: Single dashboard with variables.
```json
{
  "title": "Service CPU Usage",
  "templating": {
    "list": [
      { "name": "instance", "type": "query", "query": "label_values(node_cpu_seconds_total, instance)" },
      { "name": "environment", "type": "custom", "options": ["prod", "dev"], "current": { "value": "prod" } }
    ]
  },
  "panels": [
    {
      "title": "CPU Utilization on $instance ($environment)",
      "targets": [{ "expr": "node_cpu_seconds_total{instance=\"$instance\", env=\"$environment\"}" }]
    }
  ]
}
```

## 2. Common Patterns and Anti-patterns

Design dashboards for immediate diagnostic value and clarity.

### 2.1. Diagnostic Frameworks (USE & Four Golden Signals)

**Structure dashboards** around the USE method (Utilization, Saturation, Errors) for resource-centric views and the Four Golden Signals (Rate, Errors, Duration, Latency) for service-centric views. Each row should focus on a single resource or signal.

*   **USE Method**: Left column for Utilization, right for Saturation/Errors.
*   **Four Golden Signals**: Panels ordered logically (e.g., Rate, Errors, Latency).

### 2.2. Visual Consistency

**Enforce** these visual standards for immediate comprehension:

*   **Y-axis**: **Always** zero-based.
*   **Graph Type**: Prefer thin line graphs (`fill: 0`) for time-series. Use bar charts for quantities per interval (e.g., rates, CPU usage).
*   **Metrics per Panel**: Limit to **max 4 lines** per graph panel. More than this indicates the panel is trying to answer too many questions.
*   **Titles & Legends**: Clear, concise panel titles. Informative legends using `alias` to describe metric transformations.

❌ **BAD**: Misleading Y-axis, cluttered panels.
```json
{
  "title": "Service Latency",
  "type": "graph",
  "yAxis": { "min": null }, // Y-axis auto-scales, can hide small changes
  "targets": [
    { "expr": "service_latency_p99", "legendFormat": "P99" },
    { "expr": "service_latency_p95", "legendFormat": "P95" },
    { "expr": "service_latency_p75", "legendFormat": "P75" },
    { "expr": "service_latency_mean", "legendFormat": "Mean" },
    { "expr": "service_latency_median", "legendFormat": "Median" } // Too many lines
  ]
}
```
✅ **GOOD**: Zero-based Y-axis, focused metrics, clear aliases.
```json
{
  "title": "Service Latency (P99 & P95)",
  "type": "graph",
  "yAxis": { "min": 0 }, // Always zero-based
  "options": { "legend": { "displayMode": "table", "calcs": ["mean", "max"] } },
  "fieldConfig": { "defaults": { "custom": { "drawStyle": "line", "lineInterpolation": "linear", "fillOpacity": 0 } } },
  "targets": [
    { "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))", "legendFormat": "P99 Latency" },
    { "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))", "legendFormat": "P95 Latency" }
  ]
}
```

### 2.3. Metric Selection

*   **Counters**: **Always** use `.rate()` for StatsD counters from Graphite or `rate()`/`irate()` for Prometheus counters. Never `count()` or `sum()`.
*   **Timing**: **Prioritize** `max`, `p99`, or `p75` for timing metrics. Avoid `mean` or `median` as they hide outliers.

❌ **BAD**: Using `sum` for a counter.
```json
{ "targets": [{ "expr": "sum(http_requests_total)" }] } // Shows cumulative total, not rate
```
✅ **GOOD**: Using `rate` for a counter.
```json
{ "targets": [{ "expr": "sum(rate(http_requests_total[5m]))" }] } // Shows requests per second
```

## 3. Performance Considerations

Efficient dashboards minimize load on data sources and Grafana itself.

### 3.1. Query Optimization

**Keep queries minimal and efficient**. Avoid excessive aggregations or long time ranges in individual panels unless explicitly needed for historical context. Use `group by` sparingly.

### 3.2. Auto-Refresh Interval

**Set auto-refresh to sensible defaults** (e.g., 5min or 15min). Avoid smaller intervals (<1min) for non-alerting dashboards to prevent unnecessary load on the metric database. For immediate notification, use alerts.

```json
{
  "refresh": "5m", // Recommended default
  "time": { "from": "now-6h", "to": "now" }
}
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes that lead to confusing or inefficient dashboards.

### 4.1. Misleading Y-axis

**Never** allow the Y-axis to auto-scale or start above zero for percentage or count metrics. This distorts trends and makes small changes appear significant.

### 4.2. Overloaded Panels

A panel with too many metrics or complex queries becomes unreadable and slow. If you can't interpret a panel at a glance, it's overloaded. Break it down.

### 4.3. Lack of Context

Every dashboard **must** include a "Legend" text panel at the top. This panel should describe:
*   The dashboard's purpose.
*   The data flow and any significant transformations.
*   Links to relevant documentation or source repositories.

```json
{
  "title": "Legend",
  "type": "text",
  "gridPos": { "x": 0, "y": 0, "w": 24, "h": 2 },
  "options": {
    "content": "This dashboard monitors the **Payment Service** health. Metrics are collected via Prometheus exporters, aggregated every 15s. [Service Docs](https://internal.wiki/payment-service) | [Source Code](https://github.com/org/payment-service)",
    "mode": "markdown"
  }
}
```

### 4.4. Inconsistent Annotations

**Standardize annotations** for deployments and critical events. Configure dashboards to query common tags (e.g., `deploy`, `incident`).

```json
{
  "annotations": {
    "list": [
      {
        "name": "Deployments",
        "datasource": "Grafana",
        "iconColor": "rgba(255, 96, 96, 1)",
        "enable": true,
        "query": { "tags": "deploy", "matchAny": true }
      }
    ]
  }
}
```

## 5. Testing Approaches

Treat Grafana configurations as production code, subject to automated testing and review.

### 5.1. Dashboard Linter Integration

**Integrate `grafana/dashboard-linter`** into your CI/CD pipeline. This tool automatically scans JSON definitions for common anti-patterns (e.g., missing titles, non-zero-based axes, excessive series).

```bash
# Example CI/CD step
- name: Lint Grafana Dashboards
  run: |
    go install github.com/grafana/dashboard-linter@latest
    dashboard-linter lint dashboards/*.json
```

### 5.2. Peer Review

**Require peer review** for all changes to Grafana configuration files (dashboards, alerts, data sources). This ensures adherence to standards and catches logical errors before deployment.

### 5.3. Automated Provisioning & Git Sync

**Leverage Grafana's provisioning API** for automated deployment. For Grafana 12+, **explore and adopt Git Sync** to manage dashboards directly from the UI with Git workflows (branches, PRs). This bridges the gap between UI-driven iteration and code-driven version control.