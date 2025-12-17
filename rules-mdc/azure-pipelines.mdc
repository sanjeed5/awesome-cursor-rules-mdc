---
description: This guide provides opinionated, actionable best practices for writing secure, performant, and maintainable Azure Pipelines using YAML, focusing on modern CI/CD patterns and common pitfalls.
globs: **/*
---
# azure-pipelines Best Practices

Azure Pipelines are the backbone of modern CI/CD. This guide enforces a YAML-first, security-conscious, and highly automated approach to pipeline development.

## Code Organization and Structure

Always define pipelines using YAML for version control, reusability, and clarity. Structure your pipelines using multi-stage designs and leverage templates for common patterns.

### 1. Embrace Multi-Stage Pipelines

Separate distinct phases (Build, Test, Deploy) into stages. This provides clear logical separation, enables independent quality gates, and improves visibility.

❌ BAD: Monolithic pipeline with all steps in one job.

```yaml
# azure-pipelines.yml
jobs:
- job: BuildAndDeploy
  steps:
  - script: dotnet build
  - script: dotnet test
  - script: az webapp deploy
```

✅ GOOD: Clearly defined stages for build, test, and deployment.

```yaml
# azure-pipelines.yml
stages:
- stage: Build
  jobs:
  - job: BuildApp
    steps:
    - script: dotnet build
    - publish: $(Build.ArtifactStagingDirectory)
      artifact: drop

- stage: Test
  dependsOn: Build
  jobs:
  - job: RunTests
    steps:
    - download: current
      artifact: drop
    - script: dotnet test --logger "trx;LogFileName=test-results.trx"
    - task: PublishTestResults@2
      inputs:
        testResultsFormat: 'VSTest'
        testResultsFiles: '**/*.trx'

- stage: Deploy
  dependsOn: Test
  condition: succeeded('Test') # Only deploy if tests pass
  jobs:
  - deployment: DeployApp
    environment: 'Production' # Use environments for approvals and checks
    strategy:
      runOnce:
        deploy:
          steps:
          - download: current
            artifact: drop
          - script: az webapp deploy --resource-group MyResourceGroup --name MyApp --src-path $(Pipeline.Workspace)/drop
```

### 2. Leverage Templates for Reusability

Define common stages, jobs, or steps in reusable templates. This keeps your pipelines DRY, consistent, and easier to maintain.

❌ BAD: Copy-pasting build steps across multiple `azure-pipelines.yml` files.

```yaml
# service-a/azure-pipelines.yml
stages:
- stage: Build
  jobs:
  - job: BuildServiceA
    steps:
    - script: npm install
    - script: npm run build
    # ... more steps
```

```yaml
# service-b/azure-pipelines.yml
stages:
- stage: Build
  jobs:
  - job: BuildServiceB
    steps:
    - script: npm install
    - script: npm run build
    # ... identical steps
```

✅ GOOD: Centralize common build logic in a template.

```yaml
# templates/build-node-app.yml
parameters:
- name: appName
  type: string
  default: 'app'

jobs:
- job: BuildNodeApp
  displayName: Build ${{ parameters.appName }}
  steps:
  - task: NodeTool@0
    inputs:
      versionSpec: '18.x'
    displayName: 'Install Node.js'
  - script: npm install
    displayName: 'Install Dependencies'
  - script: npm run build
    displayName: 'Build App'
  - publish: $(Build.ArtifactStagingDirectory)
    artifact: drop-${{ parameters.appName }}
    displayName: 'Publish Build Artifact'
```

```yaml
# service-a/azure-pipelines.yml
stages:
- stage: Build
  jobs:
  - template: ../templates/build-node-app.yml
    parameters:
      appName: ServiceA
```

```yaml
# service-b/azure-pipelines.yml
stages:
- stage: Build
  jobs:
  - template: ../templates/build-node-app.yml
    parameters:
      appName: ServiceB
```

## Common Patterns and Anti-patterns

### 3. Secure Secrets with Azure Key Vault

Never hardcode secrets. Always use Azure Key Vault linked to Variable Groups for secure secret management.

❌ BAD: Storing secrets directly in Variable Groups or YAML.

```yaml
# Variable Group 'MySecrets' (in Azure DevOps UI)
#   MyApiKey: "SuperSecretKey123" (plaintext)
```

```yaml
# azure-pipelines.yml
variables:
  MyApiKey: 'SuperSecretKey123' # Directly in YAML
```

✅ GOOD: Link Variable Group to Azure Key Vault.

```yaml
# Azure Key Vault: Create a secret named 'MyApiKey'
# Azure DevOps Variable Group 'MySecrets': Link to Key Vault, select 'MyApiKey'
```

```yaml
# azure-pipelines.yml
variables:
- group: MySecrets # Link to the Variable Group
stages:
- stage: Deploy
  jobs:
  - job: DeployApp
    steps:
    - script: echo "Using API Key: $(MyApiKey)" # Access as a regular variable
```
*Note: `$(MyApiKey)` will be masked in logs if it's a secret variable.*

### 4. Use Workload Identity Federation for Service Connections

Prefer Workload Identity Federation (OIDC) over Service Principals for Azure Service Connections. It eliminates the need for manual secret management.

❌ BAD: Service Connection using a Service Principal with a client secret.

```yaml
# Azure DevOps Service Connection: Azure Resource Manager
#   Authentication Method: Service principal (manual)
#   Client secret: Manually managed and rotated
```

✅ GOOD: Service Connection using Workload Identity Federation.

```yaml
# Azure DevOps Service Connection: Azure Resource Manager
#   Authentication Method: Workload Identity Federation (automatic)
#   No client secret to manage
```
*When creating a new Azure Resource Manager service connection, choose "Workload Identity Federation (automatic)" if available.*

## Performance Considerations

### 5. Cache Dependencies

Speed up builds by caching frequently used dependencies (e.g., `node_modules`, `pip` packages, `dotnet` NuGet packages).

❌ BAD: Installing dependencies from scratch on every build.

```yaml
# azure-pipelines.yml
steps:
- script: npm install
  displayName: 'Install Node Modules'
- script: dotnet restore
  displayName: 'Restore NuGet Packages'
```

✅ GOOD: Use the `Cache@2` task.

```yaml
# azure-pipelines.yml
steps:
- task: Cache@2
  inputs:
    key: 'npm | "$(Agent.OS)" | **/package-lock.json'
    path: '$(npm_config_cache)'
    cacheHitVar: 'NPM_CACHE_RESTORED'
  displayName: 'Cache npm packages'
- script: npm install
  displayName: 'Install Node Modules'
  condition: ne(variables.NPM_CACHE_RESTORED, 'true')

- task: Cache@2
  inputs:
    key: 'nuget | "$(Agent.OS)" | $(Build.SourcesDirectory)/**/*.csproj'
    path: '$(NUGET_PACKAGES)'
    cacheHitVar: 'NUGET_CACHE_RESTORED'
  displayName: 'Cache NuGet packages'
- script: dotnet restore
  displayName: 'Restore NuGet Packages'
  condition: ne(variables.NUGET_CACHE_RESTORED, 'true')
```

### 6. Run Jobs in Parallel

For independent tasks, run jobs in parallel to reduce overall pipeline execution time.

❌ BAD: Sequential jobs when they could run concurrently.

```yaml
jobs:
- job: Build
  # ...
- job: Lint
  dependsOn: Build # Unnecessary dependency
  # ...
- job: UnitTests
  dependsOn: Build # Unnecessary dependency
  # ...
```

✅ GOOD: Parallelize independent jobs.

```yaml
jobs:
- job: Build
  # ...
- job: Lint
  dependsOn: [] # No dependency on Build
  # ...
- job: UnitTests
  dependsOn: Build # Unit tests might need build artifacts
  # ...
```

## Common Pitfalls and Gotchas

### 7. Avoid Exposing Secrets to Fork Builds

When working with public repositories, prevent secrets from being exposed to builds from forks.

❌ BAD: Enabling "Make secrets available to builds of forks" in pipeline settings.

✅ GOOD: Keep "Make secrets available to builds of forks" **disabled**. Manually trigger fork builds after review if necessary.

### 8. Use Project-Scoped Build Identities

Restrict pipeline permissions to the project level to minimize lateral exposure.

❌ BAD: Using a collection-level build identity (`Project Collection Build Service (YourCollectionName)`).

✅ GOOD: Use the default project-level build identity (`YourProjectName Build Service (YourOrganizationName)`).
*This is the default for new YAML pipelines. Verify `Job authorization scope` is set to `Project Collection` or `Current project` as appropriate for your needs, but generally `Current project` is more secure.*

## Configuration Management

### 9. Define Infrastructure as Code (IaC)

Manage your infrastructure (Azure resources, environments) as code (e.g., Bicep, ARM templates, Terraform) and deploy it via pipelines.

❌ BAD: Manually provisioning environments or using UI-driven deployments only.

✅ GOOD: Deploy infrastructure using a dedicated IaC stage.

```yaml
# azure-pipelines.yml
stages:
- stage: DeployInfrastructure
  jobs:
  - deployment: DeployBicep
    environment: 'DevEnvironment'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureCLI@2
            inputs:
              azureSubscription: 'MyAzureSubscription'
              scriptType: 'bash'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az group create --name $(ResourceGroupName) --location $(Location)
                az deployment group create \
                  --resource-group $(ResourceGroupName) \
                  --template-file ./infra/main.bicep \
                  --parameters environmentType=dev
```

## Environment Variables

### 10. Use Predefined Variables Correctly

Leverage Azure Pipelines' predefined variables for consistent and reliable access to build, release, and agent information.

❌ BAD: Hardcoding paths or trying to guess build numbers.

```yaml
steps:
- script: echo "Build number: 1.0.0.$(date +%s)" # Inconsistent
- script: cp ./dist/* /home/vsts/work/1/a/ # Fragile path
```

✅ GOOD: Use predefined variables.

```yaml
steps:
- script: echo "Build number: $(Build.BuildNumber)"
- script: cp $(Build.SourcesDirectory)/dist/* $(Build.ArtifactStagingDirectory)/
```

## Logging

### 11. Mask Sensitive Information in Logs

Ensure secrets are never printed to logs. Azure Pipelines automatically masks variables marked as secret.

❌ BAD: Printing a secret directly.

```yaml
steps:
- script: echo "My secret key is $(MyApiKey)" # Key will be visible
```

✅ GOOD: Access secret variables, which are automatically masked.

```yaml
steps:
- script: echo "Attempting to use API Key..."
- script: az login --service-principal -u $(ServicePrincipalId) -p $(ServicePrincipalKey) --tenant $(TenantId) # ServicePrincipalKey is a secret variable
```

## Testing Approaches

### 12. Integrate Comprehensive Testing Early

Run unit tests and static analysis (SAST) as early as possible in the CI stage. Dedicate separate stages for integration and end-to-end tests.

❌ BAD: Only running manual tests or basic compilation.

✅ GOOD: Multi-layered testing strategy.

```yaml
stages:
- stage: Build
  jobs:
  - job: BuildAndScan
    steps:
    - script: dotnet build
    - task: SonarQubePrepare@5 # SAST integration
    - task: SonarQubeAnalyze@5
    - task: SonarQubePublish@5

- stage: Test
  dependsOn: Build
  jobs:
  - job: UnitAndIntegrationTests
    steps:
    - script: dotnet test --filter "Category=Unit"
    - script: dotnet test --filter "Category=Integration"
    - task: PublishTestResults@2
      inputs:
        testResultsFormat: 'VSTest'
        testResultsFiles: '**/*.trx'

- stage: E2ETest
  dependsOn: Test
  condition: succeeded('Test')
  jobs:
  - deployment: DeployToTestEnv
    environment: 'Test'
    strategy:
      runOnce:
        deploy:
          steps:
          - script: npm run e2e-tests # Run Cypress/Selenium tests
```