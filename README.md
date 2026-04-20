# RAG Pipeline

A basic Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Qdrant, and Gemini.

## Stack
- **LangChain** — document loading, chunking, and chain orchestration
- **Qdrant** — vector database (runs locally via Docker)
- **Gemini** — embeddings (`gemini-embedding-001`) and LLM (`gemini-2.5-flash`)

## How it works
1. Load a document and split it into chunks
2. Embed each chunk and store in Qdrant
3. At query time, embed the question, retrieve the top-k similar chunks, and generate an answer

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/RishabhGuptaIT2k2149/Rag_pipeline.git
cd Rag_pipeline
```

**2. Install dependencies**
```bash
pip install langchain langchain-google-genai langchain-qdrant langchain-community langchain-text-splitters qdrant-client python-dotenv
```

**3. Start Qdrant**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**4. Set up environment variables**
```bash
cp .env.example .env
# add your GOOGLE_API_KEY
```

**5. Ingest your document**
```bash
python3 ingest.py
```

**6. Query**
```bash
python3 query.py
```

## What I learned
Chunking is the most critical decision in a RAG pipeline. Chunk size and overlap directly determine what context the LLM sees at query time — bad chunks mean bad retrieval regardless of everything else downstream.

## Roadmap
- Document cross-referencing for long-form documents
- Smarter chunking strategies for better context density per chunk
- Streamlit frontend with file upload and chat history
