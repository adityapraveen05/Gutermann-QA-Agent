import re
from enum import Enum
from typing import Optional


class TriageCategory(str, Enum):
    GREETING = "GREETING"
    FOLLOW_UP = "FOLLOW_UP"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class TriageResult:
    """Represents the result of classifying a user message."""

    def __init__(
        self,
        category: TriageCategory,
        confidence: float,
        reason: str,
    ):
        self.category = category
        self.confidence = confidence
        self.reason = reason

    def __repr__(self):
        return (
            f"TriageResult("
            f"category={self.category.value}, "
            f"confidence={self.confidence:.2f}, "
            f"reason='{self.reason}')"
        )


class TriageService:
    """
    Classifies incoming messages into:

        GREETING
        FOLLOW_UP
        QUESTION_ANSWERING
        OUT_OF_SCOPE

    Obvious cases are handled with deterministic rules.
    Ambiguous cases can be handled later by the LLM.
    """

    GREETING_PATTERNS = [
        r"^hi$",
        r"^hello$",
        r"^hey$",
        r"^hi there$",
        r"^hello there$",
        r"^hey there$",
        r"^good morning$",
        r"^good afternoon$",
        r"^good evening$",
        r"^how are you$",
        r"^how are you doing$",
    ]

    FOLLOW_UP_PATTERNS = [
        r"^what about",
        r"^how about",
        r"^and what about",
        r"^what about the",
        r"^what about that",
        r"^what about this",
        r"^and the",
        r"^which one",
        r"^which product",
        r"^the second one",
        r"^the first one",
        r"^the other one",
        r"^that one",
        r"^this one",
        r"^it\??$",
        r"^why\??$",
        r"^how\??$",
        r"^can you explain that",
        r"^tell me more",
        r"^more details",
    ]

    OUT_OF_SCOPE_PATTERNS = [
        r"\bweather\b",
        r"\bfootball\b",
        r"\bcricket\b",
        r"\bmovie\b",
        r"\bmovies\b",
        r"\brecipe\b",
        r"\brestaurant\b",
        r"\bstock market\b",
        r"\bpolitics\b",
        r"\bpolitician\b",
        r"\bcapital of\b",
        r"\bwho is the president\b",
        r"\bwrite me a poem\b",
        r"\btranslate\b",
    ]

    def classify(
        self,
        message: str,
        has_conversation_history: bool = False,
    ) -> TriageResult:
        """
        Classify an incoming message.

        The history flag is important because a short message such as
        "what about that one?" is only a meaningful FOLLOW_UP when
        previous conversation exists.
        """

        if not message or not message.strip():
            return TriageResult(
                category=TriageCategory.OUT_OF_SCOPE,
                confidence=1.0,
                reason="Empty message.",
            )

        normalized = self._normalize(message)

        # ----------------------------------------------------
        # 1. Greeting
        # ----------------------------------------------------

        if self._matches(
            normalized,
            self.GREETING_PATTERNS,
        ):
            return TriageResult(
                category=TriageCategory.GREETING,
                confidence=0.99,
                reason="Message matches a greeting pattern.",
            )

        # ----------------------------------------------------
        # 2. Follow-up
        # ----------------------------------------------------

        if has_conversation_history and self._matches(
            normalized,
            self.FOLLOW_UP_PATTERNS,
        ):
            return TriageResult(
                category=TriageCategory.FOLLOW_UP,
                confidence=0.95,
                reason=(
                    "Message contains follow-up language and "
                    "conversation history is available."
                ),
            )

        # ----------------------------------------------------
        # 3. Obvious out-of-scope questions
        # ----------------------------------------------------

        if self._matches(
            normalized,
            self.OUT_OF_SCOPE_PATTERNS,
        ):
            return TriageResult(
                category=TriageCategory.OUT_OF_SCOPE,
                confidence=0.90,
                reason=(
                    "Message contains a topic outside the "
                    "Gutermann product knowledge domain."
                ),
            )

        # ----------------------------------------------------
        # 4. Default to question answering
        # ----------------------------------------------------

        return TriageResult(
            category=TriageCategory.QUESTION_ANSWERING,
            confidence=0.70,
            reason=(
                "Message appears to be a product-related "
                "question or information request."
            ),
        )

    @staticmethod
    def _normalize(message: str) -> str:
        """Normalize whitespace and punctuation."""

        message = message.strip().lower()

        message = re.sub(
            r"\s+",
            " ",
            message,
        )

        message = message.strip(" \t\n.!")

        return message

    @staticmethod
    def _matches(
        message: str,
        patterns,
    ) -> bool:
        """Check whether any regex pattern matches."""

        return any(
            re.search(
                pattern,
                message,
            )
            for pattern in patterns
        )


if __name__ == "__main__":

    service = TriageService()

    tests = [
        ("Hi", False),
        ("Hello there!", False),
        (
            "Which products use True Sound Sensors?",
            False,
        ),
        (
            "What about the second one?",
            True,
        ),
        (
            "Tell me more",
            True,
        ),
        (
            "What is the capital of France?",
            False,
        ),
    ]

    for message, has_history in tests:

        result = service.classify(
            message,
            has_conversation_history=has_history,
        )

        print(
            f"\nMessage: {message}"
        )

        print(
            f"Category: {result.category.value}"
        )

        print(
            f"Confidence: {result.confidence:.2f}"
        )

        print(
            f"Reason: {result.reason}"
        )