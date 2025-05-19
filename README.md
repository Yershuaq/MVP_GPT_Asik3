# Constitutional Assistant RK v3

## Overview
A minimum viable product (MVP) AI assistant that answers questions about the Constitution of the Republic of Kazakhstan.  
Built with Streamlit, LangChain, Ollama, ChromaDB, and MongoDB.

---

## Features
- 📄 Upload PDF/DOCX files of the Constitution  
- 🔍 RAG (Retrieval-Augmented Generation): find relevant text chunks via embeddings  
- 🤖 LLM integration via Ollama for answer generation  
- 🗄 Vector store (ChromaDB) for fast semantic search  
- 💾 Chat history persisted in MongoDB  
- 📊 In-depth analysis: expand on initial answers with contextual prompts

---

## Technologies
- Python 3.9+  
- Streamlit — UI & chat interface  
- LangChain — document loading, splitting, RAG pipeline  
- OllamaEmbeddings & ChatOllama — local LLM & embeddings server  
- ChromaDB — vector database  
- MongoDB + PyMongo — chat history storage  
- python-dotenv — environment variable management  

---

## Setup & Installation

### 1. Prerequisites
- Python 3.9+  
- Docker (for MongoDB) or a running MongoDB instance  
- Ollama server running locally (port 11434 by default)

### 2. Clone the repository
```bash
git clone https://github.com/your-username/constitutional-assistant-rk.git
cd constitutional-assistant-rk
```

### 3. Create & activate a virtual environment

```bash

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate    # Windows
```

### 4. Install dependencies


``` bash
pip install -r requirements.txt
5. Configure environment variables
Create a file named .env in the project root with:

```

# .env
``` bash 
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama2
MONGO_URI=mongodb://localhost:27017/
CHROMA_PERSIST_DIR=./chroma_db_constitution_v3
```

### 6. Run the app

``` bash
streamlit run app.py
Usage
Connect
```

Enter your Ollama URL, embedding & chat model names, MongoDB URI, and ChromaDB path in the sidebar.

Click “Apply Settings & Connect”.

Upload Documents

Use the PDF/DOCX uploader to load Constitution files.

The app will split them into semantic chunks and build/update the ChromaDB index.

Chat Interface

Ask any question about the Constitution in the chat box.

The system retrieves the top-3 relevant chunks and generates an answer via Ollama.

In-Depth Analysis

After receiving an answer, click “Get In-depth Analysis” to get a deeper, structured breakdown.

### Demo
![image](https://github.com/user-attachments/assets/fea8f05a-cda7-45df-a800-b6e71f8389b7)


### Examples

txt
User: What are the fundamental rights guaranteed by the Constitution?
Assistant: According to Article 17, the Constitution guarantees ...


### Repository Structure


``` bash

.
├── app.py                # Main Streamlit application

├── requirements.txt      # Python dependencies

├── .env.example          # Example environment variables

├── README.md             # This document

└── docs/
    └── demo.png    # Demo screenshot
```

### License
This project is licensed under the MIT License. See LICENSE for details.
