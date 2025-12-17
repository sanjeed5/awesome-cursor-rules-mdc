---
description: This guide provides definitive best practices for deploying and managing Amazon EC2 instances, focusing on security, cost optimization, and operational excellence through Infrastructure as Code (IaC).
globs: **/*
---
# amazon-ec2 Best Practices

Treat every EC2 instance as versioned, secured, and continuously optimized code. Adhere strictly to the AWS Well-Architected Framework and Foundational Security Best Practices (FSBP).

## 1. Code Organization and Structure

**Always define EC2 resources using Infrastructure as Code (IaC).** CloudFormation is the standard for AWS-native deployments. Break down templates into modular, reusable components.

### 1.1. Modular CloudFormation Templates

Organize your CloudFormation templates logically. Separate networking, IAM, and EC2 instance definitions into distinct, composable stacks or modules.

❌ **BAD: Monolithic Template**
```yaml
# all-in-one-stack.yaml
Resources:
  VPC: # ...
  Subnet: # ...
  SecurityGroup: # ...
  IAMRole: # ...
  EC2Instance: # ...
```

✅ **GOOD: Modular Stacks**
```yaml
# network-stack.yaml
Resources:
  VPC: # ...
  Subnet: # ...
Outputs: # Export VPC ID, Subnet IDs
# iam-stack.yaml
Resources:
  IAMRole: # ...
Outputs: # Export Role ARN
# ec2-instance-stack.yaml
Resources:
  EC2Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0abcdef1234567890
      InstanceType: !Ref InstanceTypeParam
      SecurityGroupIds: [!ImportValue MySecurityGroup]
      IamInstanceProfile: !ImportValue MyInstanceProfile
      # ...
```

### 1.2. Enforce Consistent Tagging

Implement a unified tagging strategy from your IaC templates. Tags are critical for cost allocation, resource discovery, and automation.

❌ **BAD: Missing or Inconsistent Tags**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    # ... no tags or only a few
```

✅ **GOOD: Mandatory Tags via IaC**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    # ...
    Tags:
      - Key: Environment
        Value: !Ref EnvironmentTag
      - Key: Project
        Value: !Ref ProjectTag
      - Key: Owner
        Value: !Ref OwnerTag
      - Key: CostCenter
        Value: !Ref CostCenterTag
```

## 2. Common Patterns and Anti-patterns

### 2.1. Right-Sizing Instances

**Analyze workloads to select the optimal instance family and size.** This is the single biggest driver for performance efficiency and cost optimization.

❌ **BAD: Defaulting to `t2.micro` or Over-provisioning**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    InstanceType: t2.micro # For a production database!
```

✅ **GOOD: Workload-Specific Instance Types**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    InstanceType: m5.large # General purpose, balanced CPU/memory
    # or c5.xlarge for compute-intensive, r5.large for memory-intensive
```

### 2.2. Least Privilege IAM Instance Profiles

**Attach IAM roles to EC2 instances via Instance Profiles, granting only the minimum necessary permissions.** Never embed AWS credentials directly on an instance.

❌ **BAD: Over-privileged IAM Role**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "*",
      "Resource": "*"
    }
  ]
}
```

✅ **GOOD: Tightly Scoped IAM Role**
```yaml
# CloudFormation IAM Role for EC2
BastionHostRole:
  Type: AWS::IAM::Role
  Properties:
    AssumeRolePolicyDocument:
      Statement:
        - Effect: Allow
          Principal: { Service: [ 'ec2.amazonaws.com' ] }
          Action: [ 'sts:AssumeRole' ]
    ManagedPolicyArns:
      - arn:aws:iam::aws:policy/service-role/AmazonEC2RoleforSSM # For SSM access
    Policies:
      - PolicyName: S3ReadAccess
        PolicyDocument:
          Statement:
            - Effect: Allow
              Action: [ 's3:GetObject' ]
              Resource: !Sub 'arn:${AWS::Partition}:s3:::my-config-bucket/*'
```

### 2.3. Strict Security Group Rules

**Security groups must enforce the principle of least privilege.** Restrict inbound and outbound traffic to only what is absolutely required.

❌ **BAD: Open Ingress/Egress**
```yaml
SecurityGroupIngress:
  - IpProtocol: tcp
    FromPort: 22
    ToPort: 22
    CidrIp: 0.0.0.0/0 # SSH open to the world!
```

✅ **GOOD: Restricted Access**
```yaml
SecurityGroupIngress:
  - IpProtocol: tcp
    FromPort: 22
    ToPort: 22
    CidrIp: !Ref RemoteAccessCIDR # Parameter for specific IP range
  - IpProtocol: tcp
    FromPort: 80
    ToPort: 80
    SourceSecurityGroupId: !Ref LoadBalancerSecurityGroup # Only allow from LB
```

### 2.4. Require IMDSv2

**Always configure EC2 instances to require Instance Metadata Service Version 2 (IMDSv2).** This prevents SSRF vulnerabilities.

❌ **BAD: Relying on IMDSv1**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    # ... no MetadataOptions specified
```

✅ **GOOD: Enforce IMDSv2**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    # ...
    MetadataOptions:
      HttpPutResponseHopLimit: 1
      HttpTokens: required
```

## 3. Performance Considerations

### 3.1. Leverage Launch Templates for Auto Scaling

**Use EC2 Launch Templates with Auto Scaling Groups.** This ensures consistent configuration, supports multiple instance types, and simplifies updates.

❌ **BAD: Launch Configurations (deprecated)**
```yaml
AutoScalingGroup:
  Type: AWS::AutoScaling::AutoScalingGroup
  Properties:
    LaunchConfigurationName: !Ref MyLaunchConfig # Avoid
```

✅ **GOOD: Launch Templates**
```yaml
AutoScalingGroup:
  Type: AWS::AutoScaling::AutoScalingGroup
  Properties:
    LaunchTemplate:
      LaunchTemplateId: !Ref MyLaunchTemplate
      Version: !GetAtt MyLaunchTemplate.LatestVersionNumber
    # ...
```

### 3.2. EBS Optimization and Encryption

**Use EBS-optimized instances for I/O-intensive workloads and encrypt all EBS volumes by default.**

❌ **BAD: Unencrypted Volumes, Non-EBS Optimized**
```yaml
BlockDeviceMappings:
  - DeviceName: /dev/sdh
    Ebs:
      VolumeSize: 100
      Encrypted: false # Never!
```

✅ **GOOD: Encrypted, EBS-Optimized**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    InstanceType: m5.large # Often EBS-optimized by default or configurable
    BlockDeviceMappings:
      - DeviceName: /dev/sdh
        Ebs:
          VolumeSize: 100
          Encrypted: true # Always!
          KmsKeyId: !Ref MyEBSKMSKey # Use a customer-managed key
```

## 4. Common Pitfalls and Gotchas

### 4.1. Avoid Hardcoding Sensitive Information

**Never hardcode secrets, ARNs, or environment-specific values.** Use CloudFormation parameters, SSM Parameter Store, or AWS Secrets Manager.

❌ **BAD: Hardcoded S3 Bucket Name**
```bash
#!/bin/bash
aws s3 cp /var/log/app.log s3://my-prod-bucket-123/logs/
```

✅ **GOOD: Parameterized Values**
```yaml
# CloudFormation UserData
UserData:
  Fn::Base64: !Sub |
    #!/bin/bash
    aws s3 cp /var/log/app.log s3://${LogBucketName}/logs/
Parameters:
  LogBucketName:
    Type: String
    Description: S3 bucket for application logs
```

### 4.2. Enable Termination Protection

**Enable termination protection for critical EC2 instances.** This prevents accidental deletion.

❌ **BAD: No Termination Protection**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    # ... no DisableApiTermination
```

✅ **GOOD: Termination Protection Enabled**
```yaml
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    # ...
    DisableApiTermination: true
```

## 5. Testing Approaches

### 5.1. IaC Linting and Validation

**Integrate IaC linting and validation into your CI/CD pipeline.** Catch policy violations and syntax errors *before* deployment.

❌ **BAD: Deploying without Pre-checks**
```bash
aws cloudformation deploy --template-file my-ec2-stack.yaml --stack-name my-app
```

✅ **GOOD: CI/CD with CloudFormation Guard**
```bash
# In CI/CD pipeline
cfn-guard validate --template my-ec2-stack.yaml --rules policies/ec2-rules.guard
aws cloudformation deploy --template-file my-ec2-stack.yaml --stack-name my-app
```

### 5.2. Automated Security Hub Compliance

**Ensure your CI/CD pipeline includes checks against AWS Foundational Security Best Practices (FSBP) via AWS Security Hub.**

❌ **BAD: Manual Security Audits Post-Deployment**
```bash
# Hope for the best, check manually later
```

✅ **GOOD: Proactive Security Checks**
```bash
# Example rule in CloudFormation Guard to enforce IMDSv2
rule imdsv2_required {
  EC2_Instance {
    Properties {
      MetadataOptions {
        HttpTokens == "required"
        HttpPutResponseHopLimit == 1
      }
    }
  }
}
```