from pathlib import Path
from typing import List

import chromadb

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion import load_product_documents


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "gutermann_products"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """Persistent ChromaDB vector store for the product knowledge base."""

    def __init__(self):
        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Gutermann product knowledge base"},
        )

    def rebuild(self) -> None:
        """Rebuild the Chroma collection from the Markdown knowledge base."""

        documents = load_product_documents()

        if not documents:
            raise ValueError("No product documents were found.")

        # Clear previous collection contents.
        existing = self.collection.get()

        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        texts = [doc.page_content for doc in documents]

        print(f"Creating embeddings for {len(texts)} product chunks...")

        vectors = self.embeddings.embed_documents(texts)

        ids = [
            f"product_{index}"
            for index in range(len(documents))
        ]

        metadatas = [
            doc.metadata
            for doc in documents
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )

        print(
            f"Successfully indexed {len(documents)} product chunks."
        )

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Document]:
        """Perform basic semantic similarity search."""

        if not query.strip():
            return []

        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []

        result_documents = results.get("documents", [[]])[0]
        result_metadatas = results.get("metadatas", [[]])[0]

        for text, metadata in zip(
            result_documents,
            result_metadatas,
        ):
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return documents


if __name__ == "__main__":
    store = VectorStore()

    store.rebuild()

    query = (
        "What is the Minimum Level Profiling feature "
        "and which product has it?"
    )

    results = store.search(query, top_k=4)

    print("\n" + "=" * 70)
    print("TEST QUERY")
    print(query)

    for index, doc in enumerate(results, start=1):
        print("\n" + "-" * 70)
        print(f"Rank {index}")
        print(f"Product: {doc.metadata.get('product')}")
        print(f"Category: {doc.metadata.get('category')}")
        print(doc.page_content)