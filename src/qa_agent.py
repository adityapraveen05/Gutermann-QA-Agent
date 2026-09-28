from typing import List

from .triage import (
    TriageCategory,
    TriageService,
)
from .memory import ConversationMemory
from .follow_up import FollowUpResolver
from .retrieval import HybridRetriever
from .vector_store import VectorStore
from .generation import (
    generate_answer,
    OllamaGenerationError,
)
from .relevance_checker import RelevanceChecker
from .config import (
    TOP_K,
    VECTOR_K,
    BM25_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    MAX_MEMORY_MESSAGES,
)


class QAAgent:
    """
    Main Gutermann RAG question-answering agent.

    Pipeline:

        User Query
            ↓
        Triage
            ↓
        Follow-up Resolution
            ↓
        Hybrid Retrieval
            ↓
        Relevance Check
            ↓
        Grounded Generation
            ↓
        Conversation Memory
    """

    def __init__(self):

        print("Initializing agent...")

        self.triage = TriageService()

        self.memory = ConversationMemory(
            max_turns=MAX_MEMORY_MESSAGES
        )

        self.follow_up = FollowUpResolver()

        self.relevance_checker = RelevanceChecker()

        self.vector_store = VectorStore()

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            vector_k=VECTOR_K,
            bm25_k=BM25_K,
            vector_weight=VECTOR_WEIGHT,
            bm25_weight=BM25_WEIGHT,
        )

        print("Agent ready.")

    # ========================================================
    # MAIN ANSWER PIPELINE
    # ========================================================

    def answer(
        self,
        query: str,
    ) -> str:

        # ----------------------------------------------------
        # 1. TRIAGE
        # ----------------------------------------------------

        triage_result = self.triage.classify(
            query,
            has_conversation_history=self.memory.has_history(),
        )

        # ----------------------------------------------------
        # GREETING
        # ----------------------------------------------------

        if (
            triage_result.category
            == TriageCategory.GREETING
        ):

            answer = (
                "Hello! I can help you with questions about "
                "Gutermann water leak detection products. "
                "What would you like to know?"
            )

            self.memory.add_turn(
                user_message=query,
                assistant_answer=answer,
            )

            return answer

        # ----------------------------------------------------
        # OUT OF SCOPE
        # ----------------------------------------------------

        if (
            triage_result.category
            == TriageCategory.OUT_OF_SCOPE
        ):

            answer = (
                "I can only answer questions related to the "
                "Gutermann product knowledge base. Please ask "
                "me about Gutermann leak detection products, "
                "features, or specifications."
            )

            self.memory.add_turn(
                user_message=query,
                assistant_answer=answer,
            )

            return answer

        # ----------------------------------------------------
        # 2. FOLLOW-UP RESOLUTION
        # ----------------------------------------------------

        resolved_query = query

        if (
            triage_result.category
            == TriageCategory.FOLLOW_UP
        ):

            resolution = self.follow_up.resolve(
                query,
                self.memory,
            )

            if resolution.needs_clarification:

                answer = (
                    resolution.clarification_question
                    or
                    "Could you clarify which product you mean?"
                )

                self.memory.add_turn(
                    user_message=query,
                    assistant_answer=answer,
                )

                return answer

            resolved_query = (
                resolution.resolved_query
            )

        # ----------------------------------------------------
        # 3. HYBRID RETRIEVAL
        # ----------------------------------------------------

        results = self.retriever.hybrid_search(
            resolved_query,
            top_k=TOP_K,
        )

        documents = [
            document
            for document, _score in results
        ]

        # ----------------------------------------------------
        # 4. RELEVANCE CHECK
        # ----------------------------------------------------

        is_relevant, reason = (
            self.relevance_checker.check(
                resolved_query,
                documents,
            )
        )

        if not is_relevant:

            print(
                "\n[RELEVANCE CHECK]\n"
                f"Rejected retrieval: {reason}"
            )

            answer = (
                "I don't have enough information in the "
                "provided Gutermann knowledge base to answer "
                "that question."
            )

            self.memory.add_turn(
                user_message=query,
                assistant_answer=answer,
                retrieved_products=self._get_products(
                    documents
                ),
            )

            return answer

        # ----------------------------------------------------
        # 5. GENERATION
        # ----------------------------------------------------

        try:

            answer = generate_answer(
                query=resolved_query,
                documents=documents,
            )

        except OllamaGenerationError as exc:

            print("\n[OLLAMA ERROR]")
            print(exc)

            answer = (
                "I couldn't generate an answer because the "
                "local language model is currently unavailable. "
                "Please make sure Ollama is running and the "
                "configured model is available."
            )

        # ----------------------------------------------------
        # 6. MEMORY
        # ----------------------------------------------------

        retrieved_products = (
            self._get_products(documents)
        )

        mentioned_products = (
            self._find_mentioned_products(
                answer,
                retrieved_products,
            )
        )

        self.memory.add_turn(
            user_message=query,
            assistant_answer=answer,
            retrieved_products=retrieved_products,
            mentioned_products=mentioned_products,
        )

        return answer

    # ========================================================
    # PRODUCT HELPERS
    # ========================================================

    @staticmethod
    def _get_products(
        documents,
    ) -> List[str]:

        products = []

        for document in documents:

            product = document.metadata.get(
                "product"
            )

            if (
                product
                and product not in products
            ):
                products.append(product)

        return products

    @staticmethod
    def _find_mentioned_products(
        answer: str,
        products: List[str],
    ) -> List[str]:

        mentioned = []

        answer_lower = answer.lower()

        for product in products:

            if product.lower() in answer_lower:
                mentioned.append(product)

        return mentioned

    # ========================================================
    # INTERACTIVE CLI
    # ========================================================

    def run(self):

        print("=" * 80)
        print("GUTERMANN RAG QA AGENT")
        print("=" * 80)

        while True:

            try:

                query = input(
                    "\nUSER: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print("\nGoodbye!")
                break

            if not query:
                continue

            if query.lower() in {
                "exit",
                "quit",
            }:

                print("Goodbye!")
                break

            answer = self.answer(
                query
            )

            print(
                f"AGENT: {answer}"
            )


def main():

    agent = QAAgent()

    agent.run()


if __name__ == "__main__":
    main()