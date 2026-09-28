from dataclasses import dataclass
from typing import Any


@dataclass
class ProductChunk:
    text: str
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float
