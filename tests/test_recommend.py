import numpy as np

from textrec import TextRec, select_optimal_k

# Two clearly separated topical clusters: "space" docs and "cooking" docs.
DOCS = {
    101: "rocket launch orbit satellite space mission astronaut",
    102: "space telescope orbit galaxy astronaut rocket cosmos",
    103: "satellite orbit space station rocket launch mission",
    201: "recipe oven bake flour sugar cooking kitchen",
    202: "kitchen recipe bake bread flour cooking oven",
    203: "cooking recipe sugar flour bake dessert kitchen",
}


def _model():
    ids = list(DOCS.keys())
    texts = list(DOCS.values())
    return ids, TextRec(n_topics=2, ngram_range=(1, 1), jsd_max=0.2,
                        random_state=0).fit(ids, texts)


def test_divergence_matrix_is_well_formed():
    ids, model = _model()
    D = model.divergence_matrix()
    assert D.shape == (len(ids), len(ids))
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)
    assert np.allclose(D, D.T, atol=1e-9)


def test_recommendations_stay_within_topic_cluster():
    ids, model = _model()
    # a user who read a space article should be recommended space articles
    recs = model.recommend([("u1", 101)], enable_coldstart=False)
    recommended = {r.recommendation for r in recs}
    assert recommended  # non-empty
    assert recommended <= {102, 103}          # only other space docs
    assert all(r.type == "LDA_JSD" for r in recs)
    assert all(r.recommendation != 101 for r in recs)  # never recommend self


def test_full_pipeline_with_coldstart():
    ids, model = _model()
    interactions = [("warm_space", 101), ("warm_cook", 201)]
    user_features = {
        "warm_space": [30, 5],
        "warm_cook": [31, 5],
        "cold_user": [30, 5],  # looks like warm_space -> should inherit its recs
    }
    recs = model.recommend(interactions, user_features=user_features,
                           enable_coldstart=True, coldstart_k=1)
    cold = [r for r in recs if r.type == "ColdStart"]
    assert any(r.user_id == "cold_user" for r in cold)


def test_never_recommends_already_seen_documents():
    ids, model = _model()
    # user has read one doc from each cluster; neither may be recommended back,
    # not even via the other document in their own history.
    recs = model.recommend([("u1", 101), ("u1", 102)], enable_coldstart=False)
    recommended = {r.recommendation for r in recs}
    assert 101 not in recommended
    assert 102 not in recommended


def test_select_optimal_k_runs_and_returns_candidate():
    texts = list(DOCS.values())
    k = select_optimal_k(texts, candidate_topics=[2, 3, 4], ngram_range=(1, 1))
    assert k in (2, 3, 4)
