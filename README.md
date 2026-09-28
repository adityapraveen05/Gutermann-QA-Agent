# Gutermann CLI Q&A Agent

A local RAG-based CLI question-answering agent for the Gutermann water leak detection product knowledge base.

The agent retrieves relevant product information from a local knowledge base and generates grounded answers using a local Ollama language model.

## Current Architecture

- Markdown/product-aware knowledge base
- One chunk per product with product and category metadata
- Sentence Transformers embeddings
- ChromaDB local vector store
- BM25 keyword retrieval
- Hybrid retrieval combining semantic and lexical search
- Controlled `TOP_K` and retrieval candidate limits
- Query relevance checking
- Triage for greetings, questions, follow-ups, and out-of-scope requests
- Conversation memory
- Follow-up question resolution
- Grounded local LLM generation
- Ollama-based local inference

## Models

- LLM: `qwen2.5:3b`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

## Requirements

- Python
- Ollama
- Qwen 2.5 3B model
- Python packages listed in `requirements.txt`

## First Setup

Create and activate the virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1