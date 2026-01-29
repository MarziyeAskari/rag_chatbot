# 🧠 RAG Chatbot with Full MLOps Pipeline  
FastAPI • LangChain • Chroma • MLflow • Optuna • SQLite

A production-style **Retrieval-Augmented Generation (RAG)** system with:
- document ingestion  
- vector search  
- chat memory  
- evaluation  
- fine-tuning  
- hyperparameter tuning  
- model registry  
- monitoring  

This project is designed as a **real AI Engineering + MLOps system**.

---

## 📁 Project Structure

```
chat-bot-rag/
│
├── app/
│   ├── main.py
│   └── run.py
│
├── src/
│   ├── config_loader.py
│   ├── documents_processor.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   ├── session_manager.py
│   ├── db_session_manager.py
│   └── database.py
│
├── mlops/
│   ├── train_rag.py
│   ├── evaluate_rag.py
│   ├── finetune_embedding.py
│   ├── hyperparam_tuning.py
│   ├── metrics.py
│   ├── mlflow_utils.py
│   ├── model_registry.py
│   └── monitoring.py
│
├── configs/
│   └── config.yaml
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── data/
│   ├── documents/
│   ├── vectorstore/
│   └── sessions.db
│
├── .env
└── README.md
```

---

## 🚀 Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-multipart
```

---

## ▶️ Run

```bash
uvicorn app.main:app --reload
```

---

## 📄 Train

```bash
python -m mlops.train_rag --documents ./data/documents --clear
```

---

## 📊 Evaluate

```bash
python -m mlops.evaluate_rag
```

---

## 🎯 Purpose

This project demonstrates a **complete RAG + MLOps system** suitable for production and portfolio use.
