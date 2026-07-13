"""K-nearest-neighbour cold-start engine.

Users without interaction history cannot receive model-based recommendations.
For each such user we find the ``k`` most similar users by their numeric profile
features, collect the recommendations those neighbours received, and recommend
the documents recommended most often across the neighbourhood. This mirrors
``R/ColdStart.R``.
"""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

from .types import Recommendation

__all__ = ["cold_start"]


def cold_start(
    user_features: Mapping[Hashable, Sequence[float]],
    warm_recommendations: Iterable[Recommendation],
    k: int = 5,
    max_recs: int = 1,
) -> list[Recommendation]:
    """Generate cold-start recommendations by neighbourhood voting.

    Parameters
    ----------
    user_features:
        Mapping of ``user_id -> numeric feature vector`` for *all* users
        (warm and cold). Features are standardised internally so columns on
        different scales contribute equally.
    warm_recommendations:
        Recommendations already produced for the warm users (those with
        history). Their ``user_id`` set defines who is "warm"; everyone else is
        "cold".
    k:
        Number of nearest neighbours to poll.
    max_recs:
        Maximum number of (top-voted) recommendations to return per cold user.

    Returns
    -------
    list[Recommendation]
        One or more ``Recommendation`` per cold user, with ``type="ColdStart"``,
        ``votes`` set to the neighbour vote count and ``doc_history``/``jsd``
        left as ``None``. Empty when there are no cold users or no neighbour
        recommendations.
    """
    warm = list(warm_recommendations)
    recs_by_user: dict[Hashable, list[Hashable]] = {}
    for r in warm:
        recs_by_user.setdefault(r.user_id, []).append(r.recommendation)

    warm_ids = list(recs_by_user.keys())
    all_ids = list(user_features.keys())
    cold_ids = [u for u in all_ids if u not in recs_by_user]

    if not cold_ids or not warm_ids:
        return []

    # standardise features (z-score per column), guarding against zero variance
    ids = all_ids
    X = np.asarray([np.asarray(user_features[u], dtype=float) for u in ids])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    pos = {u: i for i, u in enumerate(ids)}

    warm_idx = np.asarray([pos[u] for u in warm_ids])
    k = min(k, len(warm_idx))

    out: list[Recommendation] = []
    for cu in cold_ids:
        diffs = Z[warm_idx] - Z[pos[cu]]
        dist = np.sqrt(np.sum(diffs ** 2, axis=1))
        nearest = warm_idx[np.argsort(dist, kind="stable")[:k]]
        neighbour_ids = [ids[i] for i in nearest]

        votes: Counter = Counter()
        for nid in neighbour_ids:
            votes.update(recs_by_user.get(nid, []))
        if not votes:
            continue

        for doc, count in votes.most_common(max_recs):
            out.append(
                Recommendation(
                    user_id=cu,
                    doc_history=None,
                    recommendation=doc,
                    jsd=None,
                    votes=int(count),
                    type="ColdStart",
                )
            )
    return out
