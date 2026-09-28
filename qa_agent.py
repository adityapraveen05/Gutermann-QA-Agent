from typing import List

from triage import (
    TriageService,
    TriageCategory,
)

from memory import ConversationMemory

from follow_up import FollowUpResolver

from retrieval import HybridRetriever

from vector_store import VectorStore

from generation import (
    generate_answer,
    OllamaGenerationError,
)

from relevance_checker import RelevanceChecker


class QAAgent:
    """
    Central orchestrator for the Gutermann CLI RAG QA agent.

    Pipeline:

        User Message
            ↓
        Triage
            ↓
        Memory / Follow-up resolution
            ↓
        Hybrid Retrieval
            ↓
        Relevance Check
            ↓
        Context Preparation
            ↓
        Qwen 2.5 3B via Ollama
            ↓
        Memory update
            ↓
        CLI response
    """

    def __init__(
        self,
        top_k: int = 4,
    ):
        self.top_k = top_k

        # ----------------------------------------------------
        # Triage
        # ----------------------------------------------------

        self.triage = TriageService()

        # ----------------------------------------------------
        # Conversation memory
        # ----------------------------------------------------

        self.memory = ConversationMemory(
            max_turns=10,
        )

        # ----------------------------------------------------
        # Follow-up resolver
        # ----------------------------------------------------

        self.follow_up_resolver = (
            FollowUpResolver()
        )

        # ----------------------------------------------------
        # Relevance checker
        # ----------------------------------------------------

        self.relevance_checker = (
            RelevanceChecker(
                minimum_keyword_overlap=0.15,
                minimum_results=1,
            )
        )

        # ----------------------------------------------------
        # Vector store
        # ----------------------------------------------------

        self.vector_store = VectorStore()

        self.vector_store.rebuild()

        # ----------------------------------------------------
        # Hybrid retrieval
        # ----------------------------------------------------

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            vector_k=10,
            bm25_k=10,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def ask(
        self,
        message: str,
    ) -> str:
        """
        Process one user message.
        """

        # ----------------------------------------------------
        # Input validation
        # ----------------------------------------------------

        if not message or not message.strip():

            return (
                "Please enter a question or message."
            )

        message = message.strip()

        # ----------------------------------------------------
        # TRIAGE
        # ----------------------------------------------------

        triage_result = self.triage.classify(
            message=message,
            has_conversation_history=(
                self.memory.has_history()
            ),
        )

        category = triage_result.category

        # ----------------------------------------------------
        # GREETING
        # ----------------------------------------------------

        if category == TriageCategory.GREETING:

            answer = self._greeting_response()

            self.memory.add_turn(
                user_message=message,
                assistant_answer=answer,
                retrieved_products=[],
                mentioned_products=[],
            )

            return answer

        # ----------------------------------------------------
        # OUT OF SCOPE
        # ----------------------------------------------------

        if category == TriageCategory.OUT_OF_SCOPE:

            answer = self._out_of_scope_response()

            self.memory.add_turn(
                user_message=message,
                assistant_answer=answer,
                retrieved_products=[],
                mentioned_products=[],
            )

            return answer

        # ----------------------------------------------------
        # FOLLOW-UP
        # ----------------------------------------------------

        query = message

        if category == TriageCategory.FOLLOW_UP:

            resolution = (
                self.follow_up_resolver.resolve(
                    message=message,
                    memory=self.memory,
                )
            )

            # ------------------------------------------------
            # Ambiguous follow-up
            # ------------------------------------------------

            if resolution.needs_clarification:

                answer = (
                    resolution.clarification_question
                )

                self.memory.add_turn(
                    user_message=message,
                    assistant_answer=answer,
                    retrieved_products=[],
                    mentioned_products=[],
                )

                return answer

            query = resolution.resolved_query

        # ----------------------------------------------------
        # QUESTION ANSWERING
        # ----------------------------------------------------

        return self._answer_question(
            original_message=message,
            retrieval_query=query,
        )

    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    def _answer_question(
        self,
        original_message: str,
        retrieval_query: str,
    ) -> str:
        """
        Retrieve relevant information, check whether it is
        sufficient, and generate a grounded answer.
        """

        try:

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            results = (
                self.retriever.hybrid_search(
                    query=retrieval_query,
                    top_k=self.top_k,
                )
            )

            if not results:

                return self._insufficient_information(
                    original_message
                )

            # ------------------------------------------------
            # DOCUMENTS
            # ------------------------------------------------

            documents = [
                document
                for document, score in results
            ]

            # ------------------------------------------------
            # RELEVANCE CHECK
            # ------------------------------------------------

            is_relevant, reason = (
                self.relevance_checker.check(
                    query=retrieval_query,
                    documents=documents,
                )
            )

            if not is_relevant:

                print(
                    f"\n[RELEVANCE CHECK]"
                )

                print(
                    f"Rejected retrieval: {reason}"
                )

                return self._insufficient_information(
                    original_message
                )

            # ------------------------------------------------
            # RETRIEVED PRODUCTS
            # ------------------------------------------------

            retrieved_products = []

            for document in documents:

                metadata = getattr(
                    document,
                    "metadata",
                    {},
                )

                product = metadata.get(
                    "product",
                    "Unknown product",
                )

                if (
                    product
                    and product
                    not in retrieved_products
                ):
                    retrieved_products.append(
                        product
                    )

            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            answer = generate_answer(
                query=retrieval_query,
                documents=documents,
                temperature=0.1,
            )

            # ------------------------------------------------
            # PRODUCT MEMORY
            # ------------------------------------------------

            mentioned_products = (
                self._extract_mentioned_products(
                    answer=answer,
                    retrieved_products=(
                        retrieved_products
                    ),
                )
            )

            # ------------------------------------------------
            # STORE MEMORY
            # ------------------------------------------------

            self.memory.add_turn(
                user_message=original_message,
                assistant_answer=answer,
                retrieved_products=(
                    retrieved_products
                ),
                mentioned_products=(
                    mentioned_products
                ),
            )

            return answer

        except OllamaGenerationError as error:

            print(
                "\n[OLLAMA ERROR]"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            return (
                "I couldn't generate an answer because "
                "the local language model is currently "
                "unavailable. Please make sure Ollama is "
                "running and the configured model is "
                "available."
            )

        except Exception as error:

            print(
                "\n[QA AGENT ERROR]"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            return (
                "An unexpected error occurred while "
                "processing your question."
            )

    # ========================================================
    # INSUFFICIENT INFORMATION
    # ========================================================

    def _insufficient_information(
        self,
        message: str,
    ) -> str:
        """
        Return a grounded response when the knowledge base
        does not provide sufficient information.
        """

        answer = (
            "I don't have enough information in the "
            "provided Gutermann knowledge base to answer "
            "that question."
        )

        self.memory.add_turn(
            user_message=message,
            assistant_answer=answer,
            retrieved_products=[],
            mentioned_products=[],
        )

        return answer

    # ========================================================
    # PRODUCT MEMORY
    # ========================================================

    @staticmethod
    def _extract_mentioned_products(
        answer: str,
        retrieved_products: List[str],
    ) -> List[str]:
        """
        Identify retrieved products explicitly mentioned
        in the generated answer.
        """

        mentioned = []

        answer_lower = answer.lower()

        for product in retrieved_products:

            if product.lower() in answer_lower:

                mentioned.append(product)

        return mentioned

    # ========================================================
    # GREETING
    # ========================================================

    @staticmethod
    def _greeting_response() -> str:
        """Return the greeting response."""

        return (
            "Hello! I can help you with questions about "
            "Gutermann water leak detection products. "
            "What would you like to know?"
        )

    # ========================================================
    # OUT OF SCOPE
    # ========================================================

    @staticmethod
    def _out_of_scope_response() -> str:
        """Return the out-of-scope response."""

        return (
            "I can only answer questions related to the "
            "Gutermann product knowledge base. "
            "Please ask me about Gutermann leak detection "
            "products, features, or specifications."
        )

    # ========================================================
    # DEBUG / TESTING
    # ========================================================

    def get_last_retrieved_products(
        self,
    ) -> List[str]:
        """Return products retrieved during the latest turn."""

        return self.memory.get_last_products()

    def get_last_mentioned_products(
        self,
    ) -> List[str]:
        """Return products mentioned during the latest answer."""

        return self.memory.get_last_mentioned_products()

    def get_last_triage_category(
        self,
        message: str,
    ) -> str:
        """Run triage independently for testing."""

        result = self.triage.classify(
            message=message,
            has_conversation_history=(
                self.memory.has_history()
            ),
        )

        return result.category.value

    def clear_memory(self) -> None:
        """Clear conversation memory."""

        self.memory.clear()