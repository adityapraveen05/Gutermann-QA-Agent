import json
import urllib.error
import urllib.request
from typing import List

from langchain_core.documents import Document

from .config import OLLAMA_MODEL, OLLAMA_BASE_URL


class OllamaGenerationError(Exception):
    """Raised when the local Ollama generation fails."""


SYSTEM_PROMPT = """
You are a technical question-answering assistant for Gutermann
water leak detection products.

Your answers must be grounded ONLY in the supplied context.

Rules:

1. Use the retrieved context as the source of truth.
2. Do not invent product specifications or features.
3. If the context does not contain enough information, say:
   "I don't have enough information in the provided knowledge base
   to answer that."
4. If multiple products are mentioned, clearly distinguish them.
5. Answer the user's actual question directly.
6. Keep the answer concise but informative.
7. Do not mention retrieval, embeddings, BM25, ChromaDB, or internal
   implementation details unless the user explicitly asks about them.
8. Do not make unsupported recommendations.
"""


def build_context(documents: List[Document]) -> str:
    """
    Convert retrieved documents into a structured context block.
    """

    if not documents:
        return "No relevant information was retrieved."

    sections = []

    for index, document in enumerate(documents, start=1):

        product = document.metadata.get(
            "product",
            "Unknown product",
        )

        category = document.metadata.get(
            "category",
            "Unknown category",
        )

        sections.append(
            f"[Context {index}]\n"
            f"Product: {product}\n"
            f"Category: {category}\n\n"
            f"{document.page_content}"
        )

    return "\n\n".join(sections)


def build_prompt(
    query: str,
    documents: List[Document],
) -> str:
    """
    Build the grounded generation prompt.
    """

    context = build_context(documents)

    return f"""
{SYSTEM_PROMPT}

RETRIEVED CONTEXT
=================
{context}

USER QUESTION
=============
{query}

ANSWER
======
"""


def generate_answer(
    query: str,
    documents: List[Document],
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer using the local Qwen model through Ollama.
    """

    prompt = build_prompt(
        query=query,
        documents=documents,
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    url = f"{OLLAMA_BASE_URL}/api/generate"

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:

            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.URLError as error:

        raise OllamaGenerationError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running and the "
            f"'{OLLAMA_MODEL}' model is available.\n"
            f"Original error: {error}"
        ) from error

    except json.JSONDecodeError as error:

        raise OllamaGenerationError(
            "Ollama returned an invalid response."
        ) from error

    answer = response_data.get("response")

    if not answer:
        raise OllamaGenerationError(
            "Ollama returned an empty response."
        )

    return answer.strip()