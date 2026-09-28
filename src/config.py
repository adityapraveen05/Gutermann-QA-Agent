from pathlib import Path


# ============================================================
# Paths
# ============================================================

# Project root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = BASE_DIR / "product_overview.md"
CHROMA_PATH = BASE_DIR / "chroma_db"


# ============================================================
# Embeddings
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Chroma
# ============================================================

CHROMA_COLLECTION_NAME = "gutermann_products"


# ============================================================
# Retrieval
# ============================================================

# Number of final documents passed to the generation model.
TOP_K = 4

# Number of candidates retrieved independently by vector search.
VECTOR_K = 10

# Number of candidates retrieved independently by BM25.
BM25_K = 10

# Weight assigned to semantic/vector retrieval.
VECTOR_WEIGHT = 0.5

# Weight assigned to lexical/BM25 retrieval.
BM25_WEIGHT = 0.5


# ============================================================
# Ollama
# ============================================================

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# Conversation Memory
# ============================================================

# Maximum number of conversation messages retained.
MAX_MEMORY_MESSAGES = 10