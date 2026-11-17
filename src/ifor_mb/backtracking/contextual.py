from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ContextEvent:
    index: int
    node: str
    component: str
    event_type: str
    severity: str
    message: str
    timestamp: float


def _tokens(message: str):
    return set(str(message).lower().split())


def rank_context(query: ContextEvent, history: Iterable[ContextEvent]):
    """Rank prior evidence without using anomaly-category relevance labels."""
    query_tokens = _tokens(query.message)
    def key(event):
        tokens = _tokens(event.message)
        union = query_tokens | tokens
        jaccard = len(query_tokens & tokens) / len(union) if union else 0.0
        return (
            int(event.node == query.node),
            int(event.component == query.component),
            int(event.event_type == query.event_type),
            int(event.severity == query.severity),
            jaccard,
            -max(query.timestamp - event.timestamp, 0.0),
        )
    return sorted(history, key=key, reverse=True)
