---
description: This rule file provides best practices, coding standards, and security guidelines for developing, deploying, and maintaining applications using the amazon-ec2 library within the AWS ecosystem. It focuses on infrastructure as code (IaC), resource management, performance, and security considerations for robust and scalable EC2-based solutions.
globs: *.tf,*.json,*.yml,*.yaml,*.py,*.js,*.ts,*.sh,*.java,*.go,*.rb,*.m
alwaysApply: false
---
- ## General Principles
  - **Infrastructure as Code (IaC):** Treat your infrastructure as code. Define and provision AWS resources (EC2 instances, security groups, networks) using code (e.g., AWS CloudFormation, AWS CDK, Terraform). This ensures consistency, repeatability, and version control.
  - **Security First:** Integrate security best practices into every stage of development, from IaC template creation to instance configuration. Implement the principle of least privilege, regularly patch instances, and utilize security assessment tools.
  - **Modularity and Reusability:** Design your infrastructure and application code in modular components that can be reused across multiple projects or environments.
  - **Automation:** Automate as much of the infrastructure provisioning, deployment, and management processes as possible. Use CI/CD pipelines for automated testing and deployment.
  - **Monitoring and Logging:** Implement comprehensive monitoring and logging to track the health, performance, and security of your EC2 instances and applications.

- ## 1. Code Organization and Structure

  - **Directory Structure Best Practices:**
    - Adopt a logical directory structure that reflects the architecture of your application and infrastructure.
    - Example:
      
      project-root/
      ├── modules/                  # Reusable infrastructure modules (e.g., VPC, security groups)
      │   ├── vpc/                # VPC module
      │   │   ├── main.tf          # Terraform configuration for the VPC
      │   │   ├── variables.tf     # Input variables for the VPC module
      │   │   ├── outputs.tf       # Output values for the VPC module
      │   ├── security_group/    # Security Group module
      │   │   ├── ...
      ├── environments/            # Environment-specific configurations
      │   ├── dev/               # Development environment
      │   │   ├── main.tf          # Terraform configuration for the Dev environment
      │   │   ├── variables.tf     # Environment specific variables
      │   ├── prod/              # Production environment
      │   │   ├── ...
      ├── scripts/                 # Utility scripts (e.g., deployment scripts, automation scripts)
      │   ├── deploy.sh          # Deployment script
      │   ├── update_ami.py      # Python script to update AMI
      ├── application/            # Application code
      │   ├── src/                # Source code
      │   ├── tests/              # Unit and integration tests
      ├── README.md
      └── ...
      
  - **File Naming Conventions:**
    - Use consistent and descriptive file names.
    - Examples:
      - `main.tf`: Main Terraform configuration file
      - `variables.tf`: Terraform variables file
      - `outputs.tf`: Terraform output values file
      - `deploy.sh`: Deployment script
      - `instance.py`: Python module for instance management
  - **Module Organization:**
    - Encapsulate reusable infrastructure components into modules (e.g., VPC, security groups, load balancers).
    - Each module should have:
      - A clear purpose.
      - Well-defined input variables and output values.
      - Comprehensive documentation.
    - Keep modules small and focused.
  - **Component Architecture:**
    - Design your application as a collection of loosely coupled components.
    - Each component should have:
      - A well-defined interface.
      - Clear responsibilities.
      - Independent deployment lifecycle.
  - **Code Splitting:**
    - Break down large application codebases into smaller, manageable modules.
    - Use lazy loading to load modules on demand, reducing initial load time.
    - Example (Python):
      python
      # main.py
      import importlib

      def load_module(module_name):
          module = importlib.import_module(module_name)
          return module

      # Load the module when needed
      my_module = load_module('my_module')
      my_module.my_function()
      

- ## 2. Common Patterns and Anti-patterns

  - **Design Patterns:**
    - **Singleton:** Use when exactly one instance of a class is needed (e.g., a configuration manager).
    - **Factory:** Use to create objects without specifying their concrete classes (e.g., creating different types of EC2 instances).
    - **Strategy:** Use to define a family of algorithms, encapsulate each one, and make them interchangeable (e.g., different instance termination strategies).
  - **Common Tasks:**
    - **Creating an EC2 Instance (AWS CLI):**
      bash
      aws ec2 run-instances --image-id ami-xxxxxxxxxxxxxxxxx --instance-type t2.micro --key-name MyKeyPair --security-group-ids sg-xxxxxxxxxxxxxxxxx
      
    - **Creating an EC2 Instance (AWS CDK):
      typescript
      import * as ec2 from 'aws-cdk-lib/aws-ec2';

      const vpc = new ec2.Vpc(this, 'TheVPC', { maxAzs: 3 });

      const instance = new ec2.Instance(this, 'EC2Instance', {
        vpc: vpc,
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.T2, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinux2023Image({generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2}),
      });
      
    - **Attaching an EBS Volume:**
      - Ensure the EBS volume is in the same Availability Zone as the EC2 instance.
      - Use the `aws ec2 attach-volume` command or the equivalent SDK call.
  - **Anti-patterns:**
    - **Hardcoding AWS Credentials:** Never hardcode AWS credentials in your code. Use IAM roles for EC2 instances and IAM users with restricted permissions for local development.
    - **Creating Publicly Accessible S3 Buckets:** Avoid creating S3 buckets that are publicly accessible without proper security controls.
    - **Ignoring Error Handling:** Always handle exceptions and errors gracefully. Provide meaningful error messages and logging.
    - **Over-Permissive Security Groups:** Implement the principle of least privilege. Grant only the minimum necessary permissions to your security groups.
  - **State Management:**
    - Use state files (e.g., Terraform state) to track the current state of your infrastructure.
    - Store state files securely (e.g., in an S3 bucket with encryption and versioning).
    - Use locking mechanisms to prevent concurrent modifications to the state file.
  - **Error Handling:**
    - Implement robust error handling to catch exceptions and prevent application crashes.
    - Use try-except blocks to handle potential errors.
    - Log error messages with sufficient detail for debugging.

- ## 3. Performance Considerations

  - **Optimization Techniques:**
    - **Instance Type Selection:** Choose the appropriate EC2 instance type based on your application's requirements (CPU, memory, network).
    - **EBS Optimization:** Use Provisioned IOPS (PIOPS) EBS volumes for high-performance applications.
    - **Caching:** Implement caching mechanisms to reduce database load and improve response times (e.g., using Amazon ElastiCache).
    - **Load Balancing:** Distribute traffic across multiple EC2 instances using an Elastic Load Balancer (ELB).
    - **Auto Scaling:** Use Auto Scaling groups to automatically scale your EC2 instances based on demand.
  - **Memory Management:**
    - Monitor memory usage on your EC2 instances.
    - Optimize application code to reduce memory consumption.
    - Use memory profiling tools to identify memory leaks.
  - **Bundle Size Optimization:**
    - Minimize the size of your application's deployment package.
    - Remove unnecessary dependencies.
    - Use code minification and compression.
    - Example (Python):
      bash
      # Create a virtual environment
      python3 -m venv .venv
      source .venv/bin/activate

      # Install only necessary dependencies
      pip install --no-cache-dir -r requirements.txt

      # Create deployment package
      zip -r deployment_package.zip *
      
  - **Lazy Loading:**
    - Load application modules on demand to reduce initial load time.
    - Use code splitting to break down large modules into smaller chunks.
    - Example (JavaScript):
      javascript
      // main.js
      async function loadModule() {
        const module = await import('./my_module.js');
        module.myFunction();
      }

      loadModule();
      

- ## 4. Security Best Practices

  - **Common Vulnerabilities:**
    - **SQL Injection:** Prevent SQL injection by using parameterized queries and input validation.
    - **Cross-Site Scripting (XSS):** Prevent XSS by sanitizing user input and encoding output.
    - **Remote Code Execution (RCE):** Prevent RCE by validating user input and using secure coding practices.
    - **Unsecured API endpoints:** Secure API endpoints using authentication and authorization mechanisms.
  - **Input Validation:**
    - Validate all user input to prevent malicious code from being injected into your application.
    - Use regular expressions and data type validation.
  - **Authentication and Authorization:**
    - Use strong authentication mechanisms (e.g., multi-factor authentication).
    - Implement role-based access control (RBAC) to restrict access to sensitive resources.
    - Use AWS IAM roles for EC2 instances to grant access to AWS resources.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use HTTPS for all API communication.
    - Store sensitive data in secure storage (e.g., AWS Secrets Manager).
  - **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Validate API requests and responses.
    - Implement rate limiting to prevent abuse.
    - Use AWS API Gateway to manage and secure your APIs.

- ## 5. Testing Approaches

  - **Unit Testing:**
    - Write unit tests for individual components to verify their functionality.
    - Use mocking and stubbing to isolate components from external dependencies.
    - Example (Python):
      python
      import unittest
      from unittest.mock import Mock

      class MyComponent:
          def __init__(self, external_dependency):
              self.dependency = external_dependency

          def my_function(self, input_data):
              result = self.dependency.process_data(input_data)
              return result

      class TestMyComponent(unittest.TestCase):
          def test_my_function(self):
              # Create a mock for the external dependency
              mock_dependency = Mock()
              mock_dependency.process_data.return_value = "Mocked Result"

              # Create an instance of MyComponent with the mock dependency
              component = MyComponent(mock_dependency)

              # Call the function to be tested
              result = component.my_function("Test Input")

              # Assert the expected behavior
              self.assertEqual(result, "Mocked Result")
              mock_dependency.process_data.assert_called_once_with("Test Input")

      if __name__ == '__main__':
          unittest.main()
      
  - **Integration Testing:**
    - Write integration tests to verify the interaction between components.
    - Test the integration of your application with AWS services.
  - **End-to-End Testing:**
    - Write end-to-end tests to verify the entire application flow.
    - Simulate real user scenarios.
  - **Test Organization:**
    - Organize your tests into a logical directory structure.
    - Use meaningful test names.
    - Keep tests independent of each other.
  - **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate components from external dependencies.
    - Example (AWS CDK):
      typescript
      // Mocking AWS SDK calls in Jest
      jest.mock('aws-sdk', () => {
        const mEC2 = {
          describeInstances: jest.fn().mockReturnValue({
            promise: jest.fn().mockResolvedValue({ Reservations: [] }),
          }),
        };
        return {
          EC2: jest.fn().mockImplementation(() => mEC2),
        };
      });
      

- ## 6. Common Pitfalls and Gotchas

  - **Frequent Mistakes:**
    - **Incorrect Security Group Configuration:** Incorrectly configured security groups can expose your EC2 instances to security risks.
    - **Insufficient Resource Limits:** Exceeding AWS resource limits can cause application failures.
    - **Not Using Auto Scaling:** Not using Auto Scaling can lead to performance bottlenecks and outages during periods of high demand.
    - **Forgetting to Terminate Unused Instances:** Forgetting to terminate unused EC2 instances can lead to unnecessary costs.
  - **Edge Cases:**
    - **Spot Instance Interruptions:** Spot instances can be interrupted with short notice. Design your application to handle spot instance interruptions gracefully.
    - **Network Connectivity Issues:** Network connectivity issues can prevent your application from accessing AWS services or other resources.
  - **Version-Specific Issues:**
    - Be aware of version-specific issues with the amazon-ec2 library or AWS services.
    - Consult the documentation for the specific versions you are using.
  - **Compatibility Concerns:**
    - Ensure compatibility between your application and the underlying operating system and libraries.
    - Test your application on different operating systems and browsers.
  - **Debugging Strategies:**
    - Use logging and monitoring to track the behavior of your application.
    - Use debugging tools to identify and fix errors.
    - Consult the AWS documentation and community forums for help.

- ## 7. Tooling and Environment

  - **Recommended Tools:**
    - **AWS CLI:** Command-line interface for interacting with AWS services.
    - **AWS Management Console:** Web-based interface for managing AWS resources.
    - **AWS CloudFormation:** Infrastructure as code service for provisioning and managing AWS resources.
    - **AWS CDK:** Cloud Development Kit for defining cloud infrastructure in code.
    - **Terraform:** Infrastructure as code tool for provisioning and managing cloud resources.
    - **Packer:** Tool for creating machine images.
    - **Ansible:** Configuration management tool.
  - **Build Configuration:**
    - Use a build tool (e.g., Make, Gradle, Maven) to automate the build process.
    - Define dependencies and build steps in a build file.
    - Example (Python):
      makefile
      # Makefile
      venv: 
      	python3 -m venv .venv
      	. .venv/bin/activate
      	pip install -r requirements.txt

      deploy:
      	zip -r deployment_package.zip *
      	aws s3 cp deployment_package.zip s3://my-bucket/deployment_package.zip
      	aws lambda update-function-code --function-name my-function --s3-bucket my-bucket --s3-key deployment_package.zip
      
  - **Linting and Formatting:**
    - Use a linter (e.g., pylint, eslint) to enforce code style and identify potential errors.
    - Use a formatter (e.g., black, prettier) to automatically format your code.
  - **Deployment Best Practices:**
    - Use a deployment pipeline to automate the deployment process.
    - Deploy to a staging environment before deploying to production.
    - Use blue/green deployments to minimize downtime.
  - **CI/CD Integration:**
    - Integrate your application with a CI/CD system (e.g., Jenkins, CircleCI, GitLab CI).
    - Automate testing, building, and deployment.

- ## Additional Considerations

  - **Cost Optimization:** Regularly review your AWS resource usage and identify opportunities for cost savings. Consider using Reserved Instances or Spot Instances to reduce costs.
  - **Disaster Recovery:** Implement a disaster recovery plan to ensure business continuity in the event of an outage. Use AWS Backup or other backup solutions to protect your data.
  - **Compliance:** Ensure that your application complies with relevant regulations and standards (e.g., PCI DSS, HIPAA).

- ## References

  - [AWS EC2 Best Practices](mdc:https:/docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-best-practices.html)
  - [AWS CDK Best Practices](mdc:https:/docs.aws.amazon.com/cdk/v2/guide/best-practices.html)
  - [Terraform AWS Provider Best Practices](mdc:https:/docs.aws.amazon.com/prescriptive-guidance/latest/terraform-aws-provider-best-practices/structure.html)