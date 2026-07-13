"""textRec -- novel document recommendations with LDA + Jensen-Shannon divergence.

Python port of the R package. Typical use::

    from textrec import TextRec

    model = TextRec(n_topics=4, jsd_max=0.1).fit(doc_ids, documents)
    recs = model.recommend(
        interactions,                 # iterable of (user_id, doc_id)
        user_features=user_features,  # {user_id: [age, account_age, ...]}
        enable_coldstart=True,
    )
    for r in recs:
        print(r.as_dict())
"""

from __future__ import annotations

from .coldstart import cold_start
from .divergence import jensen_shannon_divergence, jsd_matrix
from .recommend import TextRec, select_optimal_k
from .types import Recommendation

__version__ = "0.1.0"

__all__ = [
    "TextRec",
    "Recommendation",
    "cold_start",
    "jensen_shannon_divergence",
    "jsd_matrix",
    "select_optimal_k",
    "__version__",
]
