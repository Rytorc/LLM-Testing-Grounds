# Personal AI Assistant Architecture

This project implements a local AI assistant using the following components:

- Ollama for local LLM inference
- ChromaDB for vector storage
- SentenceTransformers for embeddings
- A custom Python RAG pipeline

## Retrieval Pipeline

1. User sends a query
2. Query embedding is generated
3. Vector search retrieves relevant chunks
4. Reranker improves relevance
5. Context is injected into the prompt

## Memory

The chatbot stores conversation history in a JSON file and compresses older messages using summarization.

## Future Improvements

- Hybrid search
- Query rewriting
- Tool usage