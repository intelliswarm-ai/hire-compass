# System Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Agent Architecture](#agent-architecture)
5. [Infrastructure Design](#infrastructure-design)
6. [Security Architecture](#security-architecture)
7. [Deployment Architecture](#deployment-architecture)

## System Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Application]
        MOB[Mobile Apps]
        API[API Clients]
        MCP[MCP Clients]
    end
    
    subgraph "API Gateway Layer"
        APIGW[API Gateway<br/>FastAPI]
        AUTH[Authentication<br/>Service]
        RATE[Rate Limiter]
    end
    
    subgraph "Application Layer"
        ORCH[Orchestrator Agent]
        subgraph "Agent Pool"
            RPA[Resume Parser<br/>Agent]
            JPA[Job Parser<br/>Agent]
            MA[Matching<br/>Agent]
            SA[Salary<br/>Agent]
            AA[Aspiration<br/>Agent]
        end
    end
    
    subgraph "MCP Server Layer"
        MCP1[Kaggle Resume<br/>MCP Server]
        MCP2[LinkedIn Jobs<br/>MCP Server]
        MCP3[Advanced Resume<br/>Analyzer]
    end
    
    subgraph "Infrastructure Layer"
        VS[Vector Store<br/>ChromaDB]
        DB[(PostgreSQL<br/>Database)]
        CACHE[(Redis<br/>Cache)]
        LLM[Ollama<br/>LLM Service]
        QUEUE[Message Queue<br/>Kafka/RabbitMQ]
    end
    
    WEB --> APIGW
    MOB --> APIGW
    API --> APIGW
    MCP --> MCP1
    MCP --> MCP2
    
    APIGW --> AUTH
    APIGW --> RATE
    AUTH --> ORCH
    
    ORCH --> RPA
    ORCH --> JPA
    ORCH --> MA
    ORCH --> SA
    ORCH --> AA
    
    RPA --> VS
    JPA --> VS
    MA --> VS
    MA --> LLM
    SA --> CACHE
    AA --> LLM
    
    RPA --> DB
    JPA --> DB
    ORCH --> QUEUE
    
    MCP1 --> VS
    MCP2 --> CACHE
    MCP3 --> LLM
```

## Component Architecture

### Clean Architecture Layers

```mermaid
graph TD
    subgraph "Presentation Layer"
        REST[REST API<br/>Controllers]
        GQL[GraphQL<br/>Resolvers]
        MCPS[MCP Server<br/>Endpoints]
        WS[WebSocket<br/>Handlers]
    end
    
    subgraph "Application Layer"
        UC[Use Cases]
        DTO[DTOs]
        MAP[Mappers]
        VAL[Validators]
    end
    
    subgraph "Domain Layer"
        ENT[Entities]
        VO[Value Objects]
        AGG[Aggregates]
        EVT[Domain Events]
        REPO[Repository<br/>Interfaces]
        SVC[Domain Services]
    end
    
    subgraph "Infrastructure Layer"
        REPOIMPL[Repository<br/>Implementations]
        EXT[External Services]
        ADAPT[Adapters]
        PERS[Persistence]
    end
    
    REST --> UC
    GQL --> UC
    MCPS --> UC
    WS --> UC
    
    UC --> DTO
    UC --> MAP
    UC --> VAL
    
    UC --> ENT
    UC --> VO
    UC --> AGG
    UC --> EVT
    UC --> REPO
    UC --> SVC
    
    REPO --> REPOIMPL
    REPOIMPL --> PERS
    UC --> EXT
    EXT --> ADAPT
```

### Component Interaction Diagram

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as API Gateway
    participant A as Auth Service
    participant O as Orchestrator
    participant RP as Resume Parser
    participant VS as Vector Store
    participant DB as Database
    
    C->>GW: Upload Resume
    GW->>A: Validate Token
    A-->>GW: Token Valid
    GW->>O: Process Resume
    O->>RP: Parse Resume
    RP->>DB: Store Metadata
    RP->>VS: Store Embeddings
    RP-->>O: Parsed Data
    O-->>GW: Success Response
    GW-->>C: Upload Complete
```

## Data Flow

### Resume Processing Flow

```mermaid
flowchart LR
    subgraph "Input"
        PDF[PDF File]
        DOCX[DOCX File]
        TXT[Text File]
    end
    
    subgraph "Processing"
        LOAD[Document<br/>Loader]
        PARSE[Resume<br/>Parser]
        NLP[NLP<br/>Processing]
        EMBED[Embedding<br/>Generation]
    end
    
    subgraph "Storage"
        META[Metadata<br/>Storage]
        VEC[Vector<br/>Storage]
        CACHE[Cache<br/>Layer]
    end
    
    subgraph "Output"
        JSON[Structured<br/>JSON]
        SCORE[Match<br/>Scores]
    end
    
    PDF --> LOAD
    DOCX --> LOAD
    TXT --> LOAD
    
    LOAD --> PARSE
    PARSE --> NLP
    NLP --> EMBED
    
    PARSE --> META
    EMBED --> VEC
    NLP --> CACHE
    
    META --> JSON
    VEC --> SCORE
```

### Matching Algorithm Flow

```mermaid
flowchart TD
    START[Start Matching]
    INPUT[Resume + Position]
    
    subgraph "Feature Extraction"
        SKILL[Skill Extraction]
        EXP[Experience Analysis]
        EDU[Education Matching]
        LOC[Location Check]
    end
    
    subgraph "Scoring"
        SKILLSCORE[Skill Score<br/>40%]
        EXPSCORE[Experience Score<br/>30%]
        EDUSCORE[Education Score<br/>20%]
        LOCSCORE[Location Score<br/>10%]
    end
    
    COMBINE[Weighted<br/>Combination]
    VECTOR[Vector<br/>Similarity]
    FINAL[Final Score]
    
    THRESHOLD{Score ><br/>Threshold?}
    MATCH[Match Found]
    NOMATCH[No Match]
    
    START --> INPUT
    INPUT --> SKILL
    INPUT --> EXP
    INPUT --> EDU
    INPUT --> LOC
    
    SKILL --> SKILLSCORE
    EXP --> EXPSCORE
    EDU --> EDUSCORE
    LOC --> LOCSCORE
    
    SKILLSCORE --> COMBINE
    EXPSCORE --> COMBINE
    EDUSCORE --> COMBINE
    LOCSCORE --> COMBINE
    
    INPUT --> VECTOR
    COMBINE --> FINAL
    VECTOR --> FINAL
    
    FINAL --> THRESHOLD
    THRESHOLD -->|Yes| MATCH
    THRESHOLD -->|No| NOMATCH
```

## Agent Architecture

### Multi-Agent System Design

```mermaid
graph TB
    subgraph "Orchestrator Layer"
        ORCH[Orchestrator Agent]
        ROUTER[Request Router]
        COORD[Coordinator]
    end
    
    subgraph "Specialized Agents"
        subgraph "Parsing Agents"
            RPA[Resume Parser]
            JPA[Job Parser]
        end
        
        subgraph "Analysis Agents"
            MA[Matching Agent]
            SA[Salary Agent]
            AA[Aspiration Agent]
        end
        
        subgraph "Integration Agents"
            LIA[LinkedIn Agent]
            KA[Kaggle Agent]
        end
    end
    
    subgraph "Shared Resources"
        POOL[Agent Pool]
        QUEUE[Task Queue]
        MEMORY[Shared Memory]
    end
    
    ORCH --> ROUTER
    ROUTER --> COORD
    
    COORD --> RPA
    COORD --> JPA
    COORD --> MA
    COORD --> SA
    COORD --> AA
    COORD --> LIA
    COORD --> KA
    
    RPA --> POOL
    JPA --> POOL
    MA --> POOL
    
    POOL --> QUEUE
    QUEUE --> MEMORY
```

### Agent Communication Protocol

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant Q as Message Queue
    participant A1 as Agent 1
    participant A2 as Agent 2
    participant R as Result Store
    
    O->>Q: Publish Task
    Q->>A1: Deliver Task
    A1->>A1: Process Task
    A1->>Q: Publish Intermediate Result
    Q->>A2: Deliver for Further Processing
    A2->>A2: Process Result
    A2->>R: Store Final Result
    A2->>Q: Publish Completion Event
    Q->>O: Notify Completion
    O->>R: Retrieve Result
```

## Infrastructure Design

### Database Schema

```mermaid
erDiagram
    RESUME {
        uuid id PK
        string name
        string email
        string phone
        string location
        text summary
        float experience_years
        timestamp created_at
        timestamp updated_at
    }
    
    POSITION {
        uuid id PK
        string title
        string department
        string location
        text description
        string experience_level
        int min_experience_years
        float salary_min
        float salary_max
        timestamp created_at
    }
    
    MATCH {
        uuid id PK
        uuid resume_id FK
        uuid position_id FK
        float overall_score
        float skill_score
        float experience_score
        float education_score
        float location_score
        json match_details
        timestamp created_at
    }
    
    SKILL {
        uuid id PK
        uuid resume_id FK
        string name
        string category
        string level
    }
    
    EXPERIENCE {
        uuid id PK
        uuid resume_id FK
        string company
        string position
        date start_date
        date end_date
        text description
    }
    
    EDUCATION {
        uuid id PK
        uuid resume_id FK
        string degree
        string field
        string institution
        int graduation_year
    }
    
    RESUME ||--o{ SKILL : has
    RESUME ||--o{ EXPERIENCE : has
    RESUME ||--o{ EDUCATION : has
    RESUME ||--o{ MATCH : matches
    POSITION ||--o{ MATCH : matches
```

### Caching Strategy

```mermaid
flowchart LR
    subgraph "Cache Layers"
        L1[L1 Cache<br/>In-Memory]
        L2[L2 Cache<br/>Redis]
        L3[L3 Cache<br/>CDN]
    end
    
    subgraph "Cache Keys"
        USER[User Data<br/>TTL: 5min]
        MATCH[Match Results<br/>TTL: 1hr]
        SALARY[Salary Data<br/>TTL: 24hr]
        STATIC[Static Data<br/>TTL: 7d]
    end
    
    REQ[Request] --> L1
    L1 -->|Miss| L2
    L2 -->|Miss| L3
    L3 -->|Miss| DB[(Database)]
    
    USER --> L1
    MATCH --> L2
    SALARY --> L2
    STATIC --> L3
```

## Security Architecture

### Security Layers

```mermaid
graph TB
    subgraph "External Security"
        WAF[Web Application<br/>Firewall]
        DDOS[DDoS<br/>Protection]
        SSL[SSL/TLS<br/>Termination]
    end
    
    subgraph "Application Security"
        AUTH[Authentication<br/>JWT/OAuth2]
        AUTHZ[Authorization<br/>RBAC]
        VALID[Input<br/>Validation]
        AUDIT[Audit<br/>Logging]
    end
    
    subgraph "Data Security"
        ENCRYPT[Encryption<br/>at Rest]
        TRANSIT[Encryption<br/>in Transit]
        MASK[Data<br/>Masking]
        BACKUP[Secure<br/>Backup]
    end
    
    subgraph "Infrastructure Security"
        FIREWALL[Network<br/>Firewall]
        SECRETS[Secrets<br/>Management]
        MONITOR[Security<br/>Monitoring]
        PATCH[Patch<br/>Management]
    end
    
    WAF --> AUTH
    SSL --> AUTHZ
    AUTH --> VALID
    AUTHZ --> AUDIT
    
    VALID --> ENCRYPT
    AUDIT --> MASK
    
    ENCRYPT --> FIREWALL
    TRANSIT --> SECRETS
    MASK --> MONITOR
```

### Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Client
    participant API as API Gateway
    participant AUTH as Auth Service
    participant IDP as Identity Provider
    participant DB as User Database
    
    U->>C: Login Request
    C->>API: POST /auth/login
    API->>AUTH: Validate Credentials
    AUTH->>IDP: Verify Identity
    IDP->>DB: Check User
    DB-->>IDP: User Data
    IDP-->>AUTH: Identity Confirmed
    AUTH->>AUTH: Generate JWT
    AUTH-->>API: JWT Token
    API-->>C: Auth Response
    C->>C: Store Token
    C-->>U: Login Success
```

## Deployment Architecture

### Kubernetes Deployment

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress"
            ING[Ingress Controller]
        end
        
        subgraph "API Pods"
            API1[API Pod 1]
            API2[API Pod 2]
            API3[API Pod 3]
        end
        
        subgraph "Agent Pods"
            AGENT1[Agent Pod 1]
            AGENT2[Agent Pod 2]
        end
        
        subgraph "MCP Pods"
            MCP1[MCP Pod 1]
            MCP2[MCP Pod 2]
        end
        
        subgraph "Data Layer"
            PG[PostgreSQL<br/>StatefulSet]
            REDIS[Redis<br/>StatefulSet]
            CHROMA[ChromaDB<br/>StatefulSet]
        end
        
        subgraph "Support Services"
            OLLAMA[Ollama<br/>Deployment]
            MONITOR[Monitoring<br/>Stack]
        end
    end
    
    ING --> API1
    ING --> API2
    ING --> API3
    
    API1 --> AGENT1
    API2 --> AGENT2
    API3 --> AGENT1
    
    AGENT1 --> PG
    AGENT2 --> REDIS
    AGENT1 --> CHROMA
    AGENT2 --> OLLAMA
```

### CI/CD Pipeline

```mermaid
flowchart LR
    subgraph "Development"
        DEV[Developer]
        LOCAL[Local Testing]
    end
    
    subgraph "Source Control"
        GIT[Git Repository]
        PR[Pull Request]
    end
    
    subgraph "CI Pipeline"
        TEST[Unit Tests]
        LINT[Code Linting]
        SEC[Security Scan]
        BUILD[Docker Build]
    end
    
    subgraph "CD Pipeline"
        STAGE[Staging Deploy]
        E2E[E2E Tests]
        PROD[Production Deploy]
        ROLL[Rollback]
    end
    
    DEV --> LOCAL
    LOCAL --> GIT
    GIT --> PR
    PR --> TEST
    TEST --> LINT
    LINT --> SEC
    SEC --> BUILD
    BUILD --> STAGE
    STAGE --> E2E
    E2E --> PROD
    PROD -.-> ROLL
```

### Monitoring Architecture

```mermaid
graph TB
    subgraph "Data Collection"
        APP[Application<br/>Metrics]
        LOG[Application<br/>Logs]
        TRACE[Distributed<br/>Traces]
        INFRA[Infrastructure<br/>Metrics]
    end
    
    subgraph "Processing"
        PROM[Prometheus]
        LOKI[Loki]
        JAEGER[Jaeger]
        ELASTIC[Elasticsearch]
    end
    
    subgraph "Visualization"
        GRAF[Grafana<br/>Dashboards]
        ALERT[Alert Manager]
        KIBANA[Kibana]
    end
    
    subgraph "Notification"
        SLACK[Slack]
        EMAIL[Email]
        PAGE[PagerDuty]
    end
    
    APP --> PROM
    LOG --> LOKI
    TRACE --> JAEGER
    INFRA --> PROM
    
    PROM --> GRAF
    LOKI --> GRAF
    JAEGER --> GRAF
    ELASTIC --> KIBANA
    
    GRAF --> ALERT
    ALERT --> SLACK
    ALERT --> EMAIL
    ALERT --> PAGE
```

## Performance Architecture

### Load Balancing Strategy

```mermaid
graph TB
    subgraph "Load Balancers"
        GLB[Global Load<br/>Balancer]
        RLB1[Regional LB<br/>US-West]
        RLB2[Regional LB<br/>US-East]
        RLB3[Regional LB<br/>EU]
    end
    
    subgraph "Application Tiers"
        subgraph "US-West"
            USW1[API Servers]
            USW2[Agent Pool]
            USW3[Cache Layer]
        end
        
        subgraph "US-East"
            USE1[API Servers]
            USE2[Agent Pool]
            USE3[Cache Layer]
        end
        
        subgraph "EU"
            EU1[API Servers]
            EU2[Agent Pool]
            EU3[Cache Layer]
        end
    end
    
    subgraph "Data Replication"
        MASTER[(Master DB)]
        SLAVE1[(Slave US-W)]
        SLAVE2[(Slave US-E)]
        SLAVE3[(Slave EU)]
    end
    
    GLB --> RLB1
    GLB --> RLB2
    GLB --> RLB3
    
    RLB1 --> USW1
    RLB2 --> USE1
    RLB3 --> EU1
    
    USW1 --> USW2
    USE1 --> USE2
    EU1 --> EU2
    
    MASTER --> SLAVE1
    MASTER --> SLAVE2
    MASTER --> SLAVE3
```

This comprehensive architecture documentation provides detailed insights into the system design, component interactions, and deployment strategies for the HR Matcher system.