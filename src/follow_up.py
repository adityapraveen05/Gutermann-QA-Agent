import re
from typing import List, Optional

from .memory import ConversationMemory


class FollowUpResolution:
    """
    Result of resolving a conversational follow-up.
    """

    def __init__(
        self,
        resolved_query: str,
        needs_clarification: bool = False,
        clarification_question: Optional[str] = None,
    ):
        self.resolved_query = resolved_query
        self.needs_clarification = needs_clarification
        self.clarification_question = clarification_question


class FollowUpResolver:
    """
    Resolves conversational references using relevant
    conversation history.

    Examples:

        What about the second one?
            -> What about AQUASCAN TM3?

        What about the first product?
            -> What about AQUASCAN 760T?

        Tell me more
            -> clarification when multiple products exist.
    """

    def resolve(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> FollowUpResolution:

        # ----------------------------------------------------
        # Find latest relevant product context
        # ----------------------------------------------------

        context_turn = (
            memory.get_latest_product_context()
        )

        if context_turn is None:

            return FollowUpResolution(
                resolved_query=message,
            )

        products = context_turn.retrieved_products

        if not products:

            return FollowUpResolution(
                resolved_query=message,
            )

        normalized = message.strip().lower()

        # ----------------------------------------------------
        # Explicit ordinal references
        # ----------------------------------------------------

        ordinal_patterns = [
            (r"\bfirst one\b", 0),
            (r"\bfirst product\b", 0),
            (r"\b1st one\b", 0),

            (r"\bsecond one\b", 1),
            (r"\bsecond product\b", 1),
            (r"\b2nd one\b", 1),

            (r"\bthird one\b", 2),
            (r"\bthird product\b", 2),
            (r"\b3rd one\b", 2),

            (r"\bfourth one\b", 3),
            (r"\bfourth product\b", 3),
            (r"\b4th one\b", 3),
        ]

        for pattern, index in ordinal_patterns:

            if re.search(
                pattern,
                normalized,
            ):

                if index < len(products):

                    product = products[index]

                    resolved = self._replace_reference(
                        message,
                        pattern,
                        product,
                    )

                    return FollowUpResolution(
                        resolved_query=resolved,
                    )

        # ----------------------------------------------------
        # Generic references
        # ----------------------------------------------------

        generic_references = [
            "that product",
            "this product",
            "that one",
            "this one",
            "the other one",
        ]

        for reference in generic_references:

            if reference in normalized:

                if len(products) > 1:

                    return FollowUpResolution(
                        resolved_query=message,
                        needs_clarification=True,
                        clarification_question=(
                            "Which product are you referring to?\n"
                            + self._format_product_options(
                                products
                            )
                        ),
                    )

                product = products[0]

                resolved = self._replace_reference(
                    message,
                    re.escape(reference),
                    product,
                )

                return FollowUpResolution(
                    resolved_query=resolved,
                )

        # ----------------------------------------------------
        # "Tell me more" style follow-up
        # ----------------------------------------------------

        more_patterns = [
            r"^tell me more[.!?]*$",
            r"^more details[.!?]*$",
            r"^can you explain that[.!?]*$",
            r"^can you explain more[.!?]*$",
        ]

        for pattern in more_patterns:

            if re.search(
                pattern,
                normalized,
            ):

                if len(products) > 1:

                    return FollowUpResolution(
                        resolved_query=message,
                        needs_clarification=True,
                        clarification_question=(
                            "Sure. Which product would you "
                            "like more details about?\n"
                            + self._format_product_options(
                                products
                            )
                        ),
                    )

                product = products[0]

                return FollowUpResolution(
                    resolved_query=(
                        f"Provide more details about "
                        f"{product}."
                    ),
                )

        # ----------------------------------------------------
        # No recognized follow-up reference
        # ----------------------------------------------------

        return FollowUpResolution(
            resolved_query=message,
        )

    @staticmethod
    def _replace_reference(
        message: str,
        pattern: str,
        product: str,
    ) -> str:
        """Replace a conversational reference with a product."""

        return re.sub(
            pattern,
            product,
            message,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _format_product_options(
        products: List[str],
    ) -> str:
        """Format product options for clarification."""

        options = products[:4]

        formatted = []

        for index, product in enumerate(
            options,
            start=1,
        ):
            formatted.append(
                f"{index}. {product}"
            )

        return "\n".join(formatted)