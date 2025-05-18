# 🧠 Chatbot RAG Local — Ollama + Qdrant + PostgreSQL + Flask

Este proyecto implementa un chatbot local con RAG (*Retrieval Augmented Generation*), sin dependencias externas ni API keys, utilizando:

- 🧠 [Ollama](https://ollama.com/) para ejecutar modelos LLM y generar embeddings.
- 🗂️ [Qdrant](https://qdrant.tech/) como base vectorial para documentos.
- 🐘 PostgreSQL para guardar el historial de conversaciones.
- 🧪 Flask como servidor backend.
- 📄 Subida y procesamiento de archivos `.pdf`, `.docx`, `.txt`, `.md`.

---

## 📦 Requirements

- Python 3.10+
- Docker + Docker Compose
- Ollama installed locally (not in Docker)
- Git

---

## 🚀 Installation

### 1. Clone this repository

```bash
git clone 'repo-url-here'
cd chatbot-rag-local
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows. On Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure the `.env` file

The project already includes a `.env` file with default values:

```env
# PostgreSQL Configuration
PG_HOST=localhost
PG_PORT=5432
PG_USER=myuser  # Used by docker-compose.yml as POSTGRES_USER
PG_PASSWORD=mypassword  # Used by docker-compose.yml as POSTGRES_PASSWORD
PG_DBNAME=mychatdb  # Used by docker-compose.yml as POSTGRES_DB
```

---

## 🐳 Start Qdrant and PostgreSQL

```powershell
docker-compose up -d
```

This will start:

* Qdrant at [http://localhost:6333](http://localhost:6333)
* PostgreSQL at localhost:5432

Both services store their data in persistent Docker volumes.

---

## 🧠 Run Ollama (outside Docker)

Make sure you have Ollama installed on your system ([https://ollama.com](https://ollama.com)):

```powershell
ollama serve
```

Then download the required models:

```powershell
ollama pull phi4-mini
ollama pull nomic-embed-text
```

You can change the models used in the `AVAILABLE_LLMS` dictionary in `server.py`.

---

## ▶️ Start the Flask server

```powershell
python server.py
```

Then open [http://localhost:5001](http://localhost:5001) in your browser.

---

## 📄 Features

* Upload documents (.pdf, .docx, .txt, .md)
* Chunking and embeddings with Ollama
* Semantic storage in Qdrant
* Dynamic RAG when the question requires it
* Chat with streaming responses (SSE)
* Persistent conversation history in PostgreSQL

---

## 🗃 Database structure

Create this table if it does not exist:

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    user_message TEXT,
    assistant_response TEXT,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 📁 Project structure

```
.
├── server.py               # Main Flask backend
├── requirements.txt        # Dependencies
├── templates/              # HTML for the interface
│   └── index.html
├── static/                 # Static resources if any
├── uploads/                # Temporarily uploaded files
├── docker-compose.yml      # Qdrant + Postgres
└── .env                    # Environment variables (Postgres)
```

---

## 🧽 Cleanup

To stop the services:

```powershell
docker-compose down
```

To completely remove containers and volumes:

```powershell
docker-compose down -v
```

---

## ✅ Summary

* Everything works **locally** (no API keys or external services required).
* Ollama runs on your system (GPU or CPU).
* Qdrant and PostgreSQL run in Docker containers.
* Flask orchestrates everything.
* It is a flexible, lightweight, and extensible solution.
