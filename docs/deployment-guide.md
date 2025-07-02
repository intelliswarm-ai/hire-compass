# Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployments](#cloud-deployments)
7. [Production Configuration](#production-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## Overview

This guide covers deployment options for the HR Matcher system, from local development to production-grade cloud deployments.

### Deployment Options

| Environment | Use Case | Components |
|------------|----------|------------|
| Local | Development | Single instance, SQLite, local Ollama |
| Docker | Testing, Small teams | Containerized services |
| Kubernetes | Production | Scalable, HA deployment |
| Cloud | Enterprise | Managed services, auto-scaling |

## Prerequisites

### System Requirements

- **Minimum Requirements:**
  - 8 GB RAM
  - 4 CPU cores
  - 50 GB storage
  - Ubuntu 20.04+ or similar

- **Recommended Production:**
  - 32 GB RAM
  - 16 CPU cores
  - 500 GB SSD storage
  - GPU for LLM inference

### Software Dependencies

```bash
# Check versions
docker --version      # 20.10+
kubectl version      # 1.24+
helm version         # 3.8+
terraform --version  # 1.3+
```

## Local Development

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-org/hire-compass.git
cd hire-compass

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp env.example .env
# Edit .env file

# 5. Start Ollama
ollama serve

# 6. Initialize database
python scripts/init_db.py

# 7. Run application
uvicorn api.async_main:app --reload
```

### Development Docker Compose

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  app:
    build: 
      context: .
      target: development
    volumes:
      - .:/app
      - /app/venv
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    command: uvicorn api.async_main:app --reload --host 0.0.0.0

  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: devpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  postgres_data:
  ollama_data:
```

## Docker Deployment

### Multi-Stage Dockerfile

```dockerfile
# Dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Development stage
FROM python:3.9-slim as development

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application
COPY . .

# Development command
CMD ["uvicorn", "api.async_main:app", "--reload", "--host", "0.0.0.0"]

# Production stage
FROM python:3.9-slim as production

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["gunicorn", "api.async_main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - app

  app:
    build:
      context: .
      target: production
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/hrdb
      - REDIS_URL=redis://redis:6379
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: hrdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    deploy:
      placement:
        constraints:
          - node.role == manager

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    deploy:
      replicas: 1

  ollama:
    image: ollama/ollama:latest
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

volumes:
  postgres_data:
  redis_data:
  chroma_data:
```

### Nginx Configuration

```nginx
# nginx/nginx.conf
upstream app_backend {
    least_conn;
    server app:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # API rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://app_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://app_backend/health;
    }
}
```

## Kubernetes Deployment

### Kubernetes Architecture

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hr-matcher
```

### Application Deployment

```yaml
# k8s/app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hr-matcher-api
  namespace: hr-matcher
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hr-matcher-api
  template:
    metadata:
      labels:
        app: hr-matcher-api
    spec:
      containers:
      - name: api
        image: your-registry/hr-matcher:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: hr-matcher-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: hr-matcher-config
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hr-matcher-api
  namespace: hr-matcher
spec:
  selector:
    app: hr-matcher-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### PostgreSQL StatefulSet

```yaml
# k8s/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: hr-matcher
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: hrdb
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hr-matcher-ingress
  namespace: hr-matcher
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
  - hosts:
    - api.hr-matcher.com
    secretName: hr-matcher-tls
  rules:
  - host: api.hr-matcher.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hr-matcher-api
            port:
              number: 80
```

### Helm Chart

```yaml
# helm/hr-matcher/values.yaml
replicaCount: 3

image:
  repository: your-registry/hr-matcher
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.hr-matcher.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: hr-matcher-tls
      hosts:
        - api.hr-matcher.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: changeme
    database: hrdb
  persistence:
    enabled: true
    size: 50Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: changeme
  master:
    persistence:
      enabled: true
      size: 10Gi

ollama:
  enabled: true
  replicaCount: 2
  persistence:
    enabled: true
    size: 100Gi
```

### Deployment Script

```bash
#!/bin/bash
# scripts/deploy-k8s.sh

set -e

NAMESPACE="hr-matcher"
RELEASE_NAME="hr-matcher"

echo "Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo "Creating secrets..."
kubectl create secret generic hr-matcher-secrets \
  --from-literal=database-url=$DATABASE_URL \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Creating configmap..."
kubectl create configmap hr-matcher-config \
  --from-file=config.yaml \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Installing Helm chart..."
helm upgrade --install $RELEASE_NAME ./helm/hr-matcher \
  --namespace $NAMESPACE \
  --values helm/hr-matcher/values.yaml \
  --values helm/hr-matcher/values.prod.yaml

echo "Waiting for deployment..."
kubectl rollout status deployment/hr-matcher-api -n $NAMESPACE

echo "Deployment complete!"
kubectl get pods -n $NAMESPACE
```

## Cloud Deployments

### AWS Deployment

#### Terraform Configuration

```hcl
# terraform/aws/main.tf
provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "3.14.0"
  
  name = "hr-matcher-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  
  tags = {
    Environment = var.environment
  }
}

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "18.26.3"
  
  cluster_name    = "hr-matcher-cluster"
  cluster_version = "1.24"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  eks_managed_node_group_defaults = {
    disk_size = 50
    instance_types = ["t3.medium"]
  }
  
  eks_managed_node_groups = {
    main = {
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      instance_types = ["t3.large"]
      
      k8s_labels = {
        Environment = var.environment
      }
    }
    
    gpu = {
      min_size     = 0
      max_size     = 3
      desired_size = 1
      
      instance_types = ["g4dn.xlarge"]
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      
      k8s_labels = {
        "accelerator" = "nvidia-tesla-t4"
      }
    }
  }
}

module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "5.0.0"
  
  identifier = "hr-matcher-db"
  
  engine            = "postgres"
  engine_version    = "15.2"
  instance_class    = "db.t3.medium"
  allocated_storage = 100
  
  db_name  = "hrdb"
  username = "hradmin"
  port     = "5432"
  
  vpc_security_group_ids = [module.security_group.security_group_id]
  
  maintenance_window = "Mon:00:00-Mon:03:00"
  backup_window      = "03:00-06:00"
  
  backup_retention_period = 7
  
  tags = {
    Environment = var.environment
  }
}

module "elasticache" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "0.1.0"
  
  cluster_id           = "hr-matcher-cache"
  engine              = "redis"
  node_type           = "cache.t3.micro"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "storage" {
  bucket = "hr-matcher-storage-${var.environment}"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "storage" {
  bucket = aws_s3_bucket.storage.id
  
  versioning_configuration {
    status = "Enabled"
  }
}
```

### GCP Deployment

```yaml
# gcp/deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gcp-config
data:
  PROJECT_ID: "your-project-id"
  REGION: "us-central1"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hr-matcher-gke
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hr-matcher
  template:
    metadata:
      labels:
        app: hr-matcher
    spec:
      serviceAccountName: hr-matcher-sa
      containers:
      - name: app
        image: gcr.io/PROJECT_ID/hr-matcher:latest
        env:
        - name: GOOGLE_CLOUD_PROJECT
          valueFrom:
            configMapKeyRef:
              name: gcp-config
              key: PROJECT_ID
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cloudsql-db-credentials
              key: connection_string
```

### Azure Deployment

```yaml
# azure/azure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hr-matcher-aks
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hr-matcher
  template:
    metadata:
      labels:
        app: hr-matcher
        aadpodidbinding: hr-matcher-identity
    spec:
      containers:
      - name: app
        image: hrmatcher.azurecr.io/hr-matcher:latest
        env:
        - name: AZURE_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: azure-identity
              key: client-id
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Production Configuration

### Environment Variables

```bash
# .env.production
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=api.hr-matcher.com,*.hr-matcher.com

# Database
DATABASE_URL=postgresql://user:pass@db-host:5432/hrdb
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30

# Cache
REDIS_URL=redis://:password@redis-host:6379/0
CACHE_TTL=3600

# LLM
OLLAMA_BASE_URL=http://ollama-service:11434
LLM_MODEL=llama2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Vector Store
CHROMA_PERSIST_DIRECTORY=/data/chroma
CHROMA_HOST=chromadb-service
CHROMA_PORT=8000

# Security
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO

# External Services
LINKEDIN_CLIENT_ID=your-client-id
LINKEDIN_CLIENT_SECRET=your-client-secret
```

### Security Hardening

```yaml
# k8s/security-policies.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: hr-matcher-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: hr-matcher-netpol
spec:
  podSelector:
    matchLabels:
      app: hr-matcher-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
    - job_name: 'hr-matcher-api'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - hr-matcher
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: hr-matcher-api
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "HR Matcher Monitoring",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"hr-matcher-api\"}[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Match Operations",
        "targets": [
          {
            "expr": "rate(matches_processed_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

### Logging Stack

```yaml
# logging/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.var.log.containers.hr-matcher-**.log>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      logstash_format true
      logstash_prefix hr-matcher
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        retry_type exponential_backoff
        flush_interval 5s
      </buffer>
    </match>
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup-postgres.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="hr-matcher-db-$TIMESTAMP"
S3_BUCKET="hr-matcher-backups"

echo "Starting PostgreSQL backup..."

# Create backup
kubectl exec -n hr-matcher postgres-0 -- \
  pg_dump -U postgres hrdb | gzip > /tmp/$BACKUP_NAME.sql.gz

# Upload to S3
aws s3 cp /tmp/$BACKUP_NAME.sql.gz s3://$S3_BUCKET/postgres/

# Clean up
rm /tmp/$BACKUP_NAME.sql.gz

echo "Backup completed: $BACKUP_NAME"

# Rotate old backups (keep last 30 days)
aws s3 ls s3://$S3_BUCKET/postgres/ | \
  awk '{print $4}' | \
  sort -r | \
  tail -n +31 | \
  xargs -I {} aws s3 rm s3://$S3_BUCKET/postgres/{}
```

### Vector Store Backup

```python
# scripts/backup_vector_store.py
import asyncio
import tarfile
import boto3
from datetime import datetime
from pathlib import Path

async def backup_chromadb():
    """Backup ChromaDB vector store to S3."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"chromadb-backup-{timestamp}.tar.gz"
    
    # Create tar archive
    with tarfile.open(f"/tmp/{backup_name}", "w:gz") as tar:
        tar.add("/data/chroma", arcname="chroma")
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(
        f"/tmp/{backup_name}",
        "hr-matcher-backups",
        f"chromadb/{backup_name}"
    )
    
    print(f"ChromaDB backup completed: {backup_name}")

if __name__ == "__main__":
    asyncio.run(backup_chromadb())
```

### Disaster Recovery Plan

```yaml
# dr/disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-plan
data:
  recovery-steps: |
    1. Database Recovery:
       - Restore PostgreSQL from S3 backup
       - Verify data integrity
       - Update connection strings
    
    2. Vector Store Recovery:
       - Restore ChromaDB from backup
       - Rebuild indexes if needed
       - Verify vector search functionality
    
    3. Application Recovery:
       - Deploy application from latest image
       - Update configuration
       - Run health checks
    
    4. Data Validation:
       - Run integrity checks
       - Verify match results
       - Test all endpoints
```

## Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
```bash
# Check pod logs
kubectl logs -n hr-matcher pod-name --previous

# Describe pod
kubectl describe pod -n hr-matcher pod-name

# Common causes:
# - Missing environment variables
# - Database connection issues
# - Insufficient resources
```

2. **Database Connection Issues**
```bash
# Test connection from pod
kubectl exec -it -n hr-matcher app-pod -- psql $DATABASE_URL

# Check secrets
kubectl get secret -n hr-matcher hr-matcher-secrets -o yaml
```

3. **High Memory Usage**
```bash
# Check resource usage
kubectl top pods -n hr-matcher

# Adjust limits
kubectl set resources deployment/hr-matcher-api \
  --limits=memory=4Gi --requests=memory=2Gi
```

4. **Slow Performance**
```bash
# Check metrics
kubectl port-forward -n monitoring prometheus-0 9090
# Access http://localhost:9090

# Scale up
kubectl scale deployment/hr-matcher-api --replicas=5
```

### Debug Mode

```yaml
# k8s/debug-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug-pod
  namespace: hr-matcher
spec:
  containers:
  - name: debug
    image: your-registry/hr-matcher:latest
    command: ["/bin/bash"]
    args: ["-c", "sleep 3600"]
    env:
    - name: DEBUG
      value: "true"
    - name: LOG_LEVEL
      value: "DEBUG"
```

### Performance Tuning

```yaml
# k8s/performance-tuning.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: performance-config
data:
  gunicorn.conf.py: |
    workers = 4
    worker_class = "uvicorn.workers.UvicornWorker"
    worker_connections = 1000
    max_requests = 1000
    max_requests_jitter = 50
    timeout = 60
    keepalive = 5
    
  nginx.conf: |
    worker_processes auto;
    worker_rlimit_nofile 65535;
    
    events {
        multi_accept on;
        worker_connections 65535;
    }
    
    http {
        sendfile on;
        tcp_nopush on;
        tcp_nodelay on;
        keepalive_timeout 65;
        
        # Enable compression
        gzip on;
        gzip_vary on;
        gzip_proxied any;
        gzip_comp_level 6;
    }
```

This comprehensive deployment guide covers all aspects of deploying the HR Matcher system from local development to production cloud environments.