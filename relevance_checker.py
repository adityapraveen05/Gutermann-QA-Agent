import re
from typing import List, Tuple


class RelevanceChecker:
    """
    Checks whether retrieved documents contain enough relevant
    information to answer a user's question.

    This is a lightweight deterministic guard placed between
    retrieval and LLM generation.

    It is intentionally not another LLM call because the goal
    is to keep the CLI agent local, fast, and predictable.
    """

    # Common words that do not provide useful evidence for
    # determining document relevance.
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "the",
        "to",
        "use",
        "uses",
        "what",
        "which",
        "who",
        "with",
        "you",
        "your",
    }

    def __init__(
        self,
        minimum_keyword_overlap: float = 0.15,
        minimum_results: int = 1,
    ):
        self.minimum_keyword_overlap = (
            minimum_keyword_overlap
        )

        self.minimum_results = minimum_results

    # ========================================================
    # PUBLIC METHOD
    # ========================================================

    def check(
        self,
        query: str,
        documents: List,
    ) -> Tuple[bool, str]:
        """
        Determine whether the retrieved documents contain
        enough relevant information.

        Returns:

            (True, reason)
                Retrieval is considered sufficient.

            (False, reason)
                Retrieval is considered insufficient.
        """

        if not query or not query.strip():
            return (
                False,
                "The query is empty.",
            )

        if not documents:
            return (
                False,
                "No relevant documents were retrieved.",
            )

        if len(documents) < self.minimum_results:
            return (
                False,
                "Too few relevant documents were retrieved.",
            )

        query_terms = self._extract_terms(query)

        if not query_terms:
            return (
                True,
                "The query does not contain enough "
                "content words for a lexical relevance check.",
            )

        combined_text = self._combine_documents(
            documents
        )

        document_terms = set(
            self._extract_terms(combined_text)
        )

        matching_terms = (
            query_terms.intersection(document_terms)
        )

        overlap = (
            len(matching_terms)
            / len(query_terms)
        )

        if overlap >= self.minimum_keyword_overlap:
            return (
                True,
                (
                    f"Relevant terms found: "
                    f"{len(matching_terms)}/"
                    f"{len(query_terms)}."
                ),
            )

        return (
            False,
            (
                "The retrieved documents do not contain "
                "enough terms related to the question."
            ),
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def _extract_terms(
        self,
        text: str,
    ) -> set:
        """
        Extract meaningful normalized terms from text.
        """

        tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if (
                len(token) > 2
                and token not in self.STOP_WORDS
            )
        }

    # ========================================================
    # DOCUMENT TEXT
    # ========================================================

    @staticmethod
    def _combine_documents(
        documents: List,
    ) -> str:
        """
        Combine page/chunk text and metadata into one string.

        Supports the current LangChain-style Document objects.
        """

        parts = []

        for document in documents:

            # Page/chunk content
            content = getattr(
                document,
                "page_content",
                "",
            )

            if content:
                parts.append(content)

            # Metadata
            metadata = getattr(
                document,
                "metadata",
                {},
            )

            if metadata:

                for value in metadata.values():

                    if value is not None:
                        parts.append(
                            str(value)
                        )

        return " ".join(parts)