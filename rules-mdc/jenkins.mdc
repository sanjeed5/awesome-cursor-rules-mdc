---
description: Definitive guidelines for writing robust, maintainable, and secure Jenkins Pipelines using modern best practices.
globs: **/*
---
# jenkins Best Practices

This guide outlines essential best practices for developing Jenkins Pipelines. Adhere to these rules to ensure your CI/CD processes are efficient, secure, and maintainable.

## 1. Code Organization and Structure

### 1.1. Always use "Pipeline as Code"
Store your `Jenkinsfile` in the root of your project's SCM repository. This enables version control, peer review, and a traceable history for your pipeline definitions.

❌ BAD: Defining pipelines directly in the Jenkins UI.
```groovy
// No Jenkinsfile in SCM, pipeline defined in UI.
// Hard to track changes, no version history.
```

✅ GOOD: Store `Jenkinsfile` in SCM.
```groovy
// Jenkinsfile at project root (e.g., my-app/Jenkinsfile)
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'mvn clean package'
            }
        }
    }
}
```

### 1.2. Prefer Declarative Pipeline Syntax
Declarative syntax is more readable, structured, and offers built-in validation. Use it for all new pipelines unless a specific, advanced scenario *absolutely* requires Scripted syntax.

❌ BAD: Using Scripted Pipeline for standard CI/CD flows.
```groovy
// Scripted Pipeline (less readable, harder to validate)
node('agent-label') {
    stage('Checkout') {
        checkout scm
    }
    stage('Build') {
        sh 'npm install'
        sh 'npm run build'
    }
}
```

✅ GOOD: Use Declarative Pipeline Syntax.
```groovy
// Declarative Pipeline (structured, readable, validated)
pipeline {
    agent { label 'agent-label' }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
    }
}
```

### 1.3. Leverage Shared Libraries for Reusability
Extract common, complex, or security-sensitive logic into Jenkins Shared Libraries. This promotes code reuse, centralizes maintenance, and keeps `Jenkinsfile`s clean and focused on orchestration. Avoid `script` tags in Declarative Pipelines; they are a strong indicator that a shared library is needed.

❌ BAD: Duplicating complex logic or using `script` tags in `Jenkinsfile`.
```groovy
// Jenkinsfile with inline script for complex logic
pipeline {
    agent any
    stages {
        stage('Deploy') {
            steps {
                script { // Avoid script tags in Declarative
                    if (env.BRANCH_NAME == 'main') {
                        sh "deploy-prod.sh --version ${env.BUILD_ID}"
                    } else {
                        sh "deploy-staging.sh --branch ${env.BRANCH_NAME}"
                    }
                }
            }
        }
    }
}
```

✅ GOOD: Use Shared Libraries for common logic.
```groovy
// Jenkinsfile using a shared library step
// (Assuming 'myOrg.deployApp' is defined in a shared library)
pipeline {
    agent any
    stages {
        stage('Deploy Staging') {
            when { expression { env.BRANCH_NAME != 'main' } }
            steps {
                myOrg.deployApp('staging') // Reusable, tested logic
            }
        }
        stage('Deploy Production') {
            when { expression { env.BRANCH_NAME == 'main' } }
            steps {
                myOrg.deployApp('prod') // Reusable, tested logic
            }
        }
    }
}
```

### 1.4. Use Organization Folders or Multibranch Pipelines
Automate job creation and management. Organization Folders (for GitHub, GitLab, Bitbucket organizations) are preferred as they automatically discover repositories and create Multibranch Pipelines. If not possible, use Multibranch Pipelines directly. This ensures every branch gets its own pipeline run.

❌ BAD: Manually creating Freestyle jobs or single Pipeline jobs per branch.
```groovy
// Manual job creation: time-consuming, error-prone, not scalable.
// This is a UI-driven anti-pattern, no code example.
```

✅ GOOD: Configure Organization Folders or Multibranch Pipelines.
```groovy
// Jenkins is configured to scan a GitHub Organization,
// automatically creating Multibranch Pipelines for each repository.
// The Jenkinsfile within each repository defines the pipeline.
```

## 2. Common Patterns and Anti-patterns

### 2.1. Execute All Material Work on Agents
Never run heavy tasks directly on the Jenkins controller (master). The controller is for orchestration; agents perform the actual build, test, and deployment work. This prevents controller overload and ensures isolated, reproducible environments.

❌ BAD: Running `sh` or `bat` commands outside an `agent` block.
```groovy
pipeline {
    // agent none is implicit here, or explicitly set
    stages {
        stage('Bad Stage') {
            steps {
                // This runs on the Jenkins controller!
                sh 'echo "This is bad, running on master!"'
                sh 'mvn clean install' // Very bad!
            }
        }
    }
}
```

✅ GOOD: Always specify an `agent` for stages performing work.
```groovy
pipeline {
    agent none // Controller only orchestrates
    stages {
        stage('Good Stage') {
            agent { label 'build-agent' } // Work runs on a dedicated agent
            steps {
                sh 'echo "This is good, running on an agent!"'
                sh 'mvn clean install'
            }
        }
    }
}
```

### 2.2. Place `input` Steps Outside `agent` Blocks and Wrap with `timeout`
`input` steps pause pipeline execution, holding an expensive agent resource unnecessarily. Always place `input` steps in a stage with `agent none` or outside any agent-bound stage. Furthermore, wrap `input` steps with a `timeout` to prevent pipelines from hanging indefinitely, ensuring proper cleanup.

❌ BAD: `input` inside an `agent` or without a `timeout`.
```groovy
pipeline {
    agent none
    stages {
        stage('Build') {
            agent { label 'build-agent' }
            steps {
                sh 'build-app.sh'
            }
        }
        stage('Manual Approval (Bad)') {
            agent { label 'build-agent' } // Agent held unnecessarily
            steps {
                input message: 'Approve deployment?' // No timeout
            }
        }
    }
}
```

✅ GOOD: `input` outside `agent` and with `timeout`.
```groovy
pipeline {
    agent none
    stages {
        stage('Build') {
            agent { label 'build-agent' }
            steps {
                sh 'build-app.sh'
            }
        }
        stage('Manual Approval (Good)') {
            // No agent specified, runs on controller (lightweight)
            options {
                timeout(time: 10, unit: 'MINUTES') // Fails if no approval in 10 min
            }
            steps {
                input message: 'Approve deployment to Production?'
            }
        }
        stage('Deploy') {
            agent { label 'deploy-agent' }
            steps {
                sh 'deploy-app.sh'
            }
        }
    }
}
```

### 2.3. Acquire Agents Within Parallel Branches
When using `parallel` stages, ensure each branch acquires its own agent. This maximizes parallelism and prevents executor starvation, significantly speeding up pipelines with independent tasks (e.g., parallel test suites).

❌ BAD: Parallel stages sharing a single agent or not explicitly acquiring agents.
```groovy
pipeline {
    agent { label 'single-agent' } // All parallel stages share this agent
    stages {
        stage('Run Tests') {
            parallel {
                stage('Unit Tests') {
                    steps { sh 'run-unit-tests.sh' }
                }
                stage('Integration Tests') {
                    steps { sh 'run-integration-tests.sh' }
                }
            }
        }
    }
}
```

✅ GOOD: Each parallel branch acquires its own agent.
```groovy
pipeline {
    agent none
    stages {
        stage('Run Tests in Parallel') {
            parallel {
                stage('Unit Tests') {
                    agent { label 'unit-test-agent' } // Dedicated agent
                    steps { sh 'run-unit-tests.sh' }
                }
                stage('Integration Tests') {
                    agent { label 'integration-test-agent' } // Dedicated agent
                    steps { sh 'run-integration-tests.sh' }
                }
            }
        }
    }
}
```

## 3. Performance Considerations

### 3.1. Optimize Agent Provisioning
Use ephemeral, container-based agents (e.g., Kubernetes agents, Docker agents) that are spun up on demand and torn down after use. This ensures clean, reproducible environments and prevents "executor starvation" by dynamically scaling resources.

❌ BAD: Long-lived, manually configured agents that accumulate state.
```groovy
// Agent setup is outside Jenkinsfile, but pipeline relies on a
// 'static-agent' that might be busy or have stale state.
agent { label 'static-agent' }
```

✅ GOOD: Use dynamic, containerized agents.
```groovy
pipeline {
    agent {
        kubernetes {
            cloud 'kubernetes'
            yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: jnlp
    image: 'jenkins/jnlp-agent:latest'
    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
  - name: maven
    image: 'maven:3.8.1-jdk-11'
    command: ['cat']
    tty: true
"""
            container 'maven'
        }
    }
    stages {
        stage('Build') {
            steps {
                sh 'mvn clean package'
            }
        }
    }
}
```

### 3.2. Implement Caching for Dependencies
Cache build dependencies (e.g., Maven local repository, npm modules) to significantly reduce build times. Use Jenkins' built-in caching mechanisms or leverage agent-specific caching where available (e.g., Docker layer caching, Kubernetes persistent volumes).

❌ BAD: Re-downloading all dependencies on every build.
```groovy
pipeline {
    agent { label 'build-agent' }
    stages {
        stage('Install Dependencies') {
            steps {
                sh 'npm install' // Downloads everything every time