"""Shared data types for :mod:`textrec`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

__all__ = ["Recommendation"]


@dataclass
class Recommendation:
    """A single recommendation for one user.

    Attributes
    ----------
    user_id:
        The user receiving the recommendation.
    doc_history:
        The document in the user's history that produced this recommendation
        (``None`` for cold-start recommendations).
    recommendation:
        The recommended document id.
    jsd:
        Jensen-Shannon divergence between ``doc_history`` and
        ``recommendation`` (``None`` for cold-start recommendations).
    votes:
        For cold-start recommendations, how many neighbours were recommended
        this document (``None`` for model-based recommendations).
    type:
        Either ``"LDA_JSD"`` (model-based) or ``"ColdStart"``.
    """

    user_id: Hashable
    recommendation: Hashable
    type: str
    doc_history: Optional[Hashable] = None
    jsd: Optional[float] = None
    votes: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "doc_history": self.doc_history,
            "recommendation": self.recommendation,
            "jsd": self.jsd,
            "votes": self.votes,
            "type": self.type,
        }
