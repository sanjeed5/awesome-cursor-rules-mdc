---
description: Definitive guidelines for developing, deploying, and operating applications on Azure, focusing on Well-Architected Framework principles, secure coding, and efficient resource management.
globs: **/*
---
# Azure Best Practices

This guide provides opinionated, actionable best practices for developing and operating solutions on Azure. It emphasizes the **Azure Well-Architected Framework** (Cost Optimization, Security, Reliability, Performance Efficiency, Operational Excellence) and the **Microsoft Cloud Adoption Framework** (CAF) for governance. Adhere to these standards to build robust, secure, and cost-effective cloud-native applications.

## 1. Code Organization and Structure

**Always use Infrastructure-as-Code (IaC) for Azure resource deployment.** Prefer modular, reusable components.

### 1.1. Modular IaC with Bicep or Terraform
Break down infrastructure into logical, reusable modules. This improves maintainability, reusability, and enforces consistency.

❌ BAD: Monolithic ARM templates or single large Bicep/Terraform files.
```bicep
// main.bicep - too many resources, hard to manage
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  // ... all storage config
}
resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  // ... all app service plan config
}
// ... many more resources
```

✅ GOOD: Modular Bicep (or Terraform modules).
```bicep
// main.bicep
module storage '