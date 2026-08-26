# textrec

> Recommend **unique and novel** documents with Latent Dirichlet Allocation
> (LDA) and Jensen–Shannon divergence.

<p>
<a href="https://github.com/HenrikVarmer/textrec/actions/workflows/tests.yaml"><img src="https://github.com/HenrikVarmer/textrec/actions/workflows/tests.yaml/badge.svg" alt="tests"></a>
<img src="https://img.shields.io/badge/python-%E2%89%A5%203.9-3776AB?logo=python&logoColor=white" alt="Python ≥ 3.9">
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
</p>

`textrec` models every document as a probability distribution over latent topics,
then measures how *close but distinct* two documents are using the
Jensen–Shannon divergence. Give it your documents and the interactions users had
with them, and it recommends documents that are **relevant but genuinely new** —
never something the user has already seen. Users with no history are served by a
k-nearest-neighbour **cold-start** engine.

```python
from textrec import TextRec

model = TextRec(n_topics=4, jsd_max=0.1).fit(doc_ids, documents)
recs  = model.recommend(
    interactions,                 # iterable of (user_id, doc_id)
    user_features=user_features,  # {user_id: [age, account_age, ...]}
    enable_coldstart=True,
)
for r in recs:
    print(r.as_dict())
```

- **LDA** via [scikit-learn](https://scikit-learn.org) (`CountVectorizer` + `LatentDirichletAllocation`)
- **Jensen–Shannon divergence** via NumPy — natural-log convention, bounded by `ln 2`, zero-safe
- **Novelty guaranteed** — documents already in a user's history are never recommended back
- **Cold-start** via k-nearest-neighbour voting over user profile features

## Install

```bash
pip install -e ".[test]"      # from a checkout; PyPI release to follow
```

Requires Python ≥ 3.9, NumPy and scikit-learn.

## Usage

```python
from textrec import TextRec

doc_ids   = [101, 102, 201, 202]
documents = [
    "rocket launch orbit satellite space mission",
    "space telescope orbit galaxy astronaut rocket",
    "recipe oven bake flour sugar cooking",
    "kitchen recipe bake bread flour cooking",
]

model = TextRec(n_topics=2, ngram_range=(1, 1), jsd_max=0.2).fit(doc_ids, documents)

interactions  = [("alice", 101), ("bob", 201)]
user_features = {"alice": [30, 5], "bob": [31, 5], "carol": [30, 5]}  # carol is cold

recs = model.recommend(interactions, user_features=user_features, enable_coldstart=True)
for r in recs:
    print(r.as_dict())
```

Each result is a `Recommendation` dataclass with `user_id`, `doc_history`,
`recommendation`, `jsd`, `votes` and `type` (`"LDA_JSD"` or `"ColdStart"`).

### How it works

1. **Topic model** — LDA assigns every document a probability distribution over `k` latent topics.
2. **Divergence** — the Jensen–Shannon divergence between every pair of per-document topic distributions is computed. Low divergence = topically similar but distinct.
3. **Recommendation** — for each document in a user's history, every *not-yet-seen* document below `jsd_max` is recommended.
4. **Cold-start** — users with no history inherit the most-voted recommendations of their `k` nearest neighbours (by profile features).

### Automatic topic selection

```python
from textrec import select_optimal_k

k = select_optimal_k(documents, candidate_topics=[2, 4, 8, 16])  # lowest perplexity
```

## API

| Object | Purpose |
|--------|---------|
| `TextRec` | Fit LDA, build the divergence matrix, produce recommendations |
| `jensen_shannon_divergence(p, q)` | JSD between two distributions |
| `jsd_matrix(P, Q=None)` | Pairwise JSD matrix |
| `cold_start(user_features, warm_recommendations, ...)` | KNN cold-start engine |
| `select_optimal_k(documents, candidate_topics, ...)` | Choose `k` by perplexity |
| `Recommendation` | Result dataclass |

## Tests

```bash
pip install -e ".[test]"
pytest
```

## R version

This is the primary, actively developed implementation. The original R package
lives at **[HenrikVarmer/textRec-R](https://github.com/HenrikVarmer/textRec-R)**.

## License

MIT © Henrik Varmer
