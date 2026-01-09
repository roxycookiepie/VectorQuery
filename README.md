# 📌 Vector Data Ingestion & Sync Pipeline

This repository demonstrates a **production-style vector ingestion and synchronization pipeline** designed for document processing, embedding generation, and vector database synchronization.  
It is intended as a **public demo / reference implementation** showcasing architecture patterns rather than a fully wired production system.

---

## 🚀 Project Overview

The system ingests documents and structured records, processes and chunks text, generates vector embeddings, and synchronizes them with a vector database.  
It is built using **Azure Functions**, **Python**, and **.NET**, following scalable and modular design principles.

**Key use cases demonstrated:**
- Document ingestion and parsing  
- Text chunking and embedding workflows  
- Vector database synchronization  
- Orchestration using serverless functions  
- Secure configuration via environment variables  

---

## 🧱 Architecture Highlights

- Serverless-first design using Azure Functions  
- Separation of concerns between ingestion, processing, and orchestration  
- Environment-based configuration (no secrets committed)  
- Vector database abstraction (Milvus / Zilliz compatible)  
- Hybrid stack (Python for data workflows, C# for orchestration)  

---

## 📂 Repository Structure

```
.
├── document_parsing_and_chunking_service.py
├── proposal_ingest.py
├── Dynamics_sync.py
├── vector_sync_orchestrator.py
├── zilliz_query_all.py
├── function_app.py
├── VectorFunctionsOrchestrator.cs
├── config_public.py
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔐 Configuration & Security

This repository **does not contain any secrets**.

All sensitive values are provided via **environment variables**:

```
VECTOR_DB_ENDPOINT=
VECTOR_DB_API_KEY=
EMBEDDING_MODEL_NAME=
AZURE_FUNCTION_ENDPOINT=
```

A template is provided in `.env.example`.

---

## 🧠 Key Concepts Demonstrated

- Vector embeddings & semantic search  
- Document chunking strategies  
- Serverless orchestration patterns  
- Cloud-ready configuration management  
- Production-oriented logging & error handling  

---

## ▶️ Running Locally (Demo)

1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies  
4. Configure environment variables  
5. Run scripts or Azure Functions locally  

> ⚠️ This project is intended for **demonstration and portfolio purposes**.

---

## 📄 License

Provided for educational and demonstration purposes.

---

## 🙌 Acknowledgements

Inspired by real-world RAG pipelines, vector databases, and cloud-native AI systems.
