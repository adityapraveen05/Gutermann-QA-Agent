from src.config import KNOWLEDGE_BASE_PATH
from src.ingestion import load_product_documents


def main() -> None:

    documents = load_product_documents()

    print(
        f"Product chunks created: "
        f"{len(documents)}\n"
    )

    print(
        f"Knowledge base: "
        f"{KNOWLEDGE_BASE_PATH}"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(
            f"\n{'=' * 72}"
        )

        print(
            f"Chunk {index}"
        )

        print(
            f"Category : "
            f"{document.metadata.get('category')}"
        )

        print(
            f"Product  : "
            f"{document.metadata.get('product')}"
        )

        print(
            f"Chunk type: "
            f"{document.metadata.get('chunk_type', 'N/A')}"
        )

        print("-" * 72)

        print(
            document.page_content
        )

        print()


if __name__ == "__main__":
    main()