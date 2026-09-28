from config import KNOWLEDGE_BASE
from ingestion import load_chunks


def main() -> None:
    chunks = load_chunks(KNOWLEDGE_BASE)

    print(f"Product chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"{'=' * 72}")
        print(f"Chunk {i}")
        print(f"Category : {chunk.metadata.get('category')}")
        print(f"Product  : {chunk.metadata.get('product')}")
        print(f"Page     : {chunk.metadata.get('page', 'N/A')}")
        print("-" * 72)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()
