from pathlib import Path
import re
from typing import List

from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = BASE_DIR / "product_overview.md"


def clean_text(text: str) -> str:
    """Clean unnecessary Markdown whitespace while preserving content."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_product_documents() -> List[Document]:
    """
    Load the Markdown knowledge base and create one Document per product.

    The Markdown structure is:

        ## Category
        ### Product
        description
        feature bullets

    Each product remains within its own chunk so information from
    different products is not mixed.
    """

    if not KNOWLEDGE_BASE.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE}"
        )

    content = KNOWLEDGE_BASE.read_text(encoding="utf-8")

    lines = content.splitlines()

    documents: List[Document] = []

    current_category = None
    current_product = None
    current_content = []

    def save_current_product():
        nonlocal current_product, current_content

        if not current_product:
            return

        body = clean_text("\n".join(current_content))

        if not body:
            return

        full_text = (
            f"Category: {current_category}\n"
            f"Product: {current_product}\n\n"
            f"{body}"
        )

        documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "category": current_category or "Unknown",
                    "product": current_product,
                    "chunk_type": "product",
                },
            )
        )

        current_product = None
        current_content = []

    for line in lines:
        stripped = line.strip()

        # Category heading
        if stripped.startswith("## ") and not stripped.startswith("### "):
            save_current_product()

            current_category = stripped[3:].strip()
            continue

        # Product heading
        if stripped.startswith("### "):
            save_current_product()

            current_product = stripped[4:].strip()
            current_content = []
            continue

        # Ignore top-level title
        if stripped.startswith("# "):
            continue

        # Add normal content only when inside a product
        if current_product:
            current_content.append(line)

    # Save final product
    save_current_product()

    return documents


if __name__ == "__main__":
    docs = load_product_documents()

    print(f"Product chunks created: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print("\n" + "=" * 70)
        print(f"Chunk {i}")
        print(f"Category : {doc.metadata['category']}")
        print(f"Product  : {doc.metadata['product']}")
        print("-" * 70)
        print(doc.page_content)