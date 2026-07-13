"""The textRec recommendation pipeline (Python port).

Fits a Latent Dirichlet Allocation topic model to a corpus, measures the
Jensen-Shannon divergence between every pair of per-document topic
distributions, and recommends, for each document in a user's history, every
other (not-yet-seen) document whose divergence falls below ``jsd_max``. Users
without history
are optionally served by the k-nearest-neighbour cold-start engine.

This is the Python counterpart of the R package's ``textRec()`` entry point.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .coldstart import cold_start
from .divergence import jsd_matrix
from .types import Recommendation

__all__ = ["TextRec", "select_optimal_k"]


class TextRec:
    """LDA + Jensen-Shannon document recommender.

    Parameters
    ----------
    n_topics:
        Number of LDA topics (``k``).
    ngram_range:
        ``(min_n, max_n)`` word n-gram range for the vectoriser.
    stopwords:
        Either ``"english"``, a list of stop words, or ``None``.
    jsd_max:
        Maximum Jensen-Shannon divergence for a document to be recommended.
    lda_alpha:
        LDA Dirichlet document-topic prior (``doc_topic_prior``).
    max_features:
        Optional cap on vocabulary size.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_topics: int = 10,
        ngram_range: tuple[int, int] = (1, 2),
        stopwords: Optional[Sequence[str] | str] = "english",
        jsd_max: float = 0.1,
        lda_alpha: Optional[float] = None,
        max_features: Optional[int] = None,
        random_state: int = 123,
    ) -> None:
        self.n_topics = n_topics
        self.ngram_range = ngram_range
        self.stopwords = stopwords
        self.jsd_max = jsd_max
        self.lda_alpha = lda_alpha
        self.max_features = max_features
        self.random_state = random_state

        self.vectorizer_: Optional[CountVectorizer] = None
        self.lda_: Optional[LatentDirichletAllocation] = None
        self.doc_ids_: list[Hashable] = []
        self.doc_topics_: Optional[np.ndarray] = None
        self._divergence: Optional[np.ndarray] = None

    # -- fitting ------------------------------------------------------------
    def fit(self, doc_ids: Sequence[Hashable], documents: Sequence[str]) -> "TextRec":
        """Fit the vectoriser + LDA model and store per-document topic mixes."""
        if len(doc_ids) != len(documents):
            raise ValueError("`doc_ids` and `documents` must be the same length.")

        self.vectorizer_ = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=self.stopwords,
            max_features=self.max_features,
            lowercase=True,
        )
        dtm = self.vectorizer_.fit_transform(documents)

        self.lda_ = LatentDirichletAllocation(
            n_components=self.n_topics,
            doc_topic_prior=self.lda_alpha,
            learning_method="batch",
            random_state=self.random_state,
        )
        gamma = self.lda_.fit_transform(dtm)
        # rows sum to 1 -> per-document topic probability distributions
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        self.doc_ids_ = list(doc_ids)
        self.doc_topics_ = gamma
        self._divergence = None
        return self

    # -- divergence ---------------------------------------------------------
    def divergence_matrix(self) -> np.ndarray:
        """Symmetric document-to-document Jensen-Shannon divergence matrix."""
        self._require_fitted()
        if self._divergence is None:
            self._divergence = jsd_matrix(self.doc_topics_)
        return self._divergence

    # -- recommendation -----------------------------------------------------
    def recommend(
        self,
        interactions: Iterable[tuple[Hashable, Hashable]],
        user_features: Optional[Mapping[Hashable, Sequence[float]]] = None,
        enable_coldstart: bool = True,
        coldstart_k: int = 5,
        max_recs: int = 1,
    ) -> list[Recommendation]:
        """Recommend novel documents for each user.

        Parameters
        ----------
        interactions:
            Iterable of ``(user_id, doc_id)`` pairs.
        user_features:
            Optional mapping ``user_id -> numeric feature vector`` for the
            cold-start engine. Required if ``enable_coldstart`` is true and any
            users lack history.
        enable_coldstart:
            Whether to serve users without history via :func:`cold_start`.
        coldstart_k:
            Neighbours to poll in the cold-start engine.
        max_recs:
            Maximum recommendations per cold user.
        """
        self._require_fitted()
        D = self.divergence_matrix()
        index = {doc: i for i, doc in enumerate(self.doc_ids_)}

        # every document each user has already interacted with -- these are
        # never "novel", so they must not be recommended back to that user, not
        # even from a different item in their history.
        interactions = list(interactions)
        seen: dict[Hashable, set[Hashable]] = {}
        for user_id, doc_id in interactions:
            seen.setdefault(user_id, set()).add(doc_id)

        recs: list[Recommendation] = []
        for user_id, doc_id in interactions:
            if doc_id not in index:
                continue
            row = D[index[doc_id]]
            for j, div in enumerate(row):
                cand = self.doc_ids_[j]
                if cand == doc_id or cand in seen[user_id]:
                    continue
                if div < self.jsd_max:
                    recs.append(
                        Recommendation(
                            user_id=user_id,
                            doc_history=doc_id,
                            recommendation=cand,
                            jsd=float(div),
                            votes=None,
                            type="LDA_JSD",
                        )
                    )

        if enable_coldstart and user_features is not None:
            recs.extend(
                cold_start(user_features, recs, k=coldstart_k, max_recs=max_recs)
            )
        return recs

    # -- helpers ------------------------------------------------------------
    def _require_fitted(self) -> None:
        if self.doc_topics_ is None:
            raise RuntimeError("TextRec must be fit() before use.")


def select_optimal_k(
    documents: Sequence[str],
    candidate_topics: Iterable[int],
    ngram_range: tuple[int, int] = (1, 2),
    stopwords: Optional[Sequence[str] | str] = "english",
    random_state: int = 123,
) -> int:
    """Pick the topic count with the lowest held-out perplexity.

    A Python analogue of the R ``select_optimal_k()`` / ``automate_topics``
    behaviour: each candidate ``k`` is scored by LDA perplexity on the corpus
    and the best (lowest) is returned.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords, lowercase=True)
    dtm = vectorizer.fit_transform(documents)

    best_k, best_score = None, np.inf
    for k in candidate_topics:
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="batch", random_state=random_state
        )
        lda.fit(dtm)
        score = lda.perplexity(dtm)
        if score < best_score:
            best_k, best_score = k, score
    if best_k is None:
        raise ValueError("`candidate_topics` was empty.")
    return int(best_k)
