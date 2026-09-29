# Gutermann CLI Q&A Agent

A local RAG-based CLI question-answering agent for the Gutermann water leak detection product knowledge base.

The agent retrieves relevant product information from a local knowledge base and generates grounded answers using a local Ollama language model.

## Project Structure

```text
Gutermann-QA-Agent/
│
├── src/
│   ├── config.py
│   ├── follow_up.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── memory.py
│   ├── models.py
│   ├── qa_agent.py
│   ├── relevance_checker.py
│   ├── retrieval.py
│   ├── triage.py
│   └── vector_store.py
│
├── tests/
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_memory.py
│   ├── test_problem_statement.py
│   ├── test_qa_agent.py
│   ├── test_triage.py
│   └── test_vector_store.py
│
├── product_overview.md
├── requirements.txt
├── README.md
└── .gitignore