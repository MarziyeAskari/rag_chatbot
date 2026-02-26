# 🧠 Production RAG Chatbot with Full MLOps & AWS Deployment

FastAPI • LangChain • PGVector / Chroma • MLflow • AWS ECS • S3 • SQS • RDS • Docker

A production-ready **Retrieval-Augmented Generation (RAG)** platform designed with modern AI engineering and MLOps principles.
This system supports scalable document ingestion, conversational AI, experiment tracking, and cloud deployment on AWS.

The project demonstrates a **real-world end-to-end AI system** including backend services, asynchronous pipelines, vector databases, and infrastructure deployment.

---

# 🚀 Key Features

## 🔹 Core AI Capabilities

* Retrieval-Augmented Generation (RAG)
* Multi-provider embeddings:

  * OpenAI
  * HuggingFace Sentence Transformers
* Vector databases:

  * PostgreSQL + pgvector (production)
  * Chroma (local development)
* Conversational memory & session management
* Similarity filtering with configurable thresholds

---

## 🔹 MLOps & Experimentation

* MLflow experiment tracking server
* Model registry support
* Evaluation pipelines
* Metrics logging
* Hyperparameter tuning with Optuna
* Embedding fine-tuning workflows

---

## 🔹 Cloud & Production Architecture

* Dockerized microservices
* AWS ECS Fargate deployment
* Application Load Balancer (ALB)
* PostgreSQL (Amazon RDS)
* S3 document storage
* SQS asynchronous processing
* AWS Secrets Manager
* CloudWatch logging
* IAM role-based security

---

## 🔹 Async Document Processing Pipeline

```
User Upload → API → S3 → SQS Queue → Worker → Vector Database
```

This allows scalable ingestion without blocking API requests.

---

# 🏗️ System Architecture

```
                    ┌──────────────┐
User ─────────────► │  ALB (HTTP)   │
                    └──────┬───────┘
                           │
                    ┌──────▼────────┐
                    │ ECS API Service│
                    │ FastAPI + RAG  │
                    └──────┬─────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   PostgreSQL        S3 Storage         OpenAI API
   (RDS + pgvector)  Documents          LLM / Embeddings

                           │
                           ▼
                    ┌──────────────┐
                    │   SQS Queue   │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ ECS Worker    │
                    │ Document Proc │
                    └──────────────┘
```

---

# 📁 Project Structure

```
rag_chatbot/
│
├── app/
│   ├── main.py              # FastAPI API service
│   ├── worker.py            # SQS worker service
│   └── run.py
│
├── src/
│   ├── vector_store.py
│   ├── rag_chain.py
│   ├── documents_processor.py
│   ├── session_manager.py
│   ├── db_session_manager.py
│   ├── upload_storage.py
│   ├── config_loader.py
│   └── queue.py
│
├── mlops/
│   ├── train_rag.py
│   ├── evaluate_rag.py
│   ├── finetune_embedding.py
│   ├── tune_optuna.py
│   ├── registry.py
│   └── mlflow_utils.py
│
├── configs/
├── docker/
├── frontend/
└── README.md
```

---

# ⚙️ Local Development

## 1. Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install python-multipart
```

## 3. Run API

```bash
uvicorn app.main:app --reload
```

API documentation:

```
http://localhost:8000/docs
```

---

# 🧪 Training & Evaluation

## Train Vector Database

```bash
python -m mlops.train_rag --documents ./data/documents --clear
```

## Evaluate System

```bash
python -m mlops.evaluate_rag
```

---

# ☁️ AWS Deployment Overview

## Services Used

* ECS Fargate
* Elastic Container Registry (ECR)
* RDS PostgreSQL (pgvector)
* S3 Buckets
* SQS Queue
* Application Load Balancer
* Secrets Manager
* CloudWatch Logs
* IAM Roles

---

# 🔥 Deployment Steps (High Level)

## 1️⃣ Build & Push Docker Image

```bash
docker build -t rag-api .
docker tag rag-api:latest <aws_account>.dkr.ecr.<region>.amazonaws.com/rag-api
docker push <repository>
```

---

## 2️⃣ Infrastructure Setup

### Networking

* Default VPC
* Public subnets
* Security groups

### RDS PostgreSQL

* Enable pgvector extension

### S3 Buckets

* Document uploads
* MLflow artifacts

### SQS Queue

* Async processing jobs

---

## 3️⃣ ECS Services

### API Service

Command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Worker Service

Command:

```bash
python app/worker.py
```

---

## 4️⃣ MLflow Service

Command:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
  --default-artifact-root s3://<bucket>/mlflow
```

---

# 🔐 Secrets Management

Stored in AWS Secrets Manager:

* OpenAI API key
* Database URLs
* MLflow backend URI
* AWS credentials (if needed)

Injected into containers via environment variables.

---

# 📡 API Endpoints

## Health Check

```
GET /health
```

## Query

```
POST /query
```

## Upload Document

```
POST /upload
```

## Sessions

```
POST /sessions
```

---

# 📊 Monitoring & Logs

Logs available via:

AWS Console → CloudWatch → Log Groups → ECS Tasks (accessible only to users with appropriate AWS account permissions; not public)

---

# 🧠 Vector Database Access

Production database:

PostgreSQL with pgvector

Note: The RDS database is deployed inside a private VPC and is not publicly accessible. It can only be reached from AWS resources within the same network (such as ECS tasks or an EC2 instance with the correct security group permissions).

Example connection from an EC2 instance:

```bash
psql -h <rds-endpoint> -U postgres -d postgres
```

---

# 🎯 Learning & Engineering Value

This project demonstrates:

✅ Production AI architecture
✅ Cloud deployment skills
✅ Async distributed systems
✅ MLOps lifecycle
✅ Vector database integration
✅ Backend engineering with FastAPI

Suitable for:

* AI Engineer roles
* ML Engineer roles
* Backend AI Developer roles

---

# 👩‍💻 Author

**Marziye Askari**
AI Developer — Vienna, Austria

Specializations:

* RAG Systems
* LLM Engineering
* MLOps
* Cloud AI Deployment

---

# ⭐ Future Improvements

* Kubernetes deployment
* CI/CD automation
* GPU inference support
* Multi-tenant architecture
* Advanced evaluation dashboards

---
