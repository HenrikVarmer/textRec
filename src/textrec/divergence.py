"""Jensen-Shannon divergence utilities.

This mirrors the R implementation in ``R/JensenShannonDivergence.R`` /
``R/JSD_matrix.R``: divergences use natural logarithms and are therefore bounded
in ``[0, ln 2]`` -- ``0`` for identical distributions and ``ln 2`` for
distributions with disjoint support.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["jensen_shannon_divergence", "jsd_matrix"]


def _shannon_entropy(p: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    """Shannon entropy in nats, treating ``0 * log 0`` as ``0``."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -np.sum(terms, axis=axis)


def jensen_shannon_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    Parameters
    ----------
    p, q:
        Non-negative vectors of the same length. They are treated as
        probability distributions; zero-probability components are handled
        without producing ``NaN``.

    Returns
    -------
    float
        A value in ``[0, ln 2]``.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("`p` and `q` must have the same shape.")
    m = 0.5 * (p + q)
    # JSD = H(m) - 0.5 H(p) - 0.5 H(q)
    return float(_shannon_entropy(m) - 0.5 * _shannon_entropy(p) - 0.5 * _shannon_entropy(q))


def jsd_matrix(prob_distrib: ArrayLike, col_prob_distrib: ArrayLike | None = None) -> NDArray[np.float64]:
    """Pairwise Jensen-Shannon divergence between rows of two matrices.

    When ``col_prob_distrib`` is ``None`` the divergence of ``prob_distrib``
    against itself is returned -- a symmetric matrix with a zero diagonal, as
    used for document-to-document similarity.
    """
    P = np.atleast_2d(np.asarray(prob_distrib, dtype=float))
    Q = P if col_prob_distrib is None else np.atleast_2d(np.asarray(col_prob_distrib, dtype=float))
    if P.shape[1] != Q.shape[1]:
        raise ValueError("Row distributions must share the same dimensionality.")

    h_p = _shannon_entropy(P, axis=1)  # (n,)
    h_q = _shannon_entropy(Q, axis=1)  # (m,)

    out = np.empty((P.shape[0], Q.shape[0]), dtype=float)
    for i in range(P.shape[0]):
        m = 0.5 * (P[i] + Q)                  # (m, t)
        out[i] = _shannon_entropy(m, axis=1) - 0.5 * h_p[i] - 0.5 * h_q
    # numerical noise can produce tiny negatives; clamp to the valid range
    return np.clip(out, 0.0, np.log(2.0))
