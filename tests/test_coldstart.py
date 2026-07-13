from textrec import cold_start
from textrec.types import Recommendation


def _warm(user_id, doc):
    return Recommendation(user_id=user_id, recommendation=doc, type="LDA_JSD",
                          doc_history=1, jsd=0.01)


def test_cold_users_served_by_neighbour_votes():
    # users 5 & 6 are cold (mid-age), nearest warm neighbours are 5..6's peers
    user_features = {
        1: [20, 1], 2: [21, 1],   # young, warm
        3: [60, 9], 4: [61, 9],   # old, warm
        5: [40, 5], 6: [41, 5],   # mid, COLD
    }
    warm = [_warm(1, "777"), _warm(2, "777"), _warm(3, "888"), _warm(4, "999")]

    out = cold_start(user_features, warm, k=2, max_recs=1)
    served = {r.user_id for r in out}
    assert {5, 6} <= served
    assert all(r.type == "ColdStart" for r in out)
    assert all(r.votes is not None and r.votes >= 1 for r in out)
    assert all(r.doc_history is None and r.jsd is None for r in out)


def test_no_cold_users_returns_empty():
    user_features = {1: [20, 1], 2: [40, 5]}
    warm = [_warm(1, "9"), _warm(2, "9")]
    assert cold_start(user_features, warm) == []


def test_max_recs_caps_output_per_user():
    user_features = {1: [0, 0], 2: [0, 0], 3: [10, 10]}  # 3 is cold
    warm = [_warm(1, "a"), _warm(1, "b"), _warm(2, "c")]
    out = cold_start(user_features, warm, k=2, max_recs=2)
    assert len([r for r in out if r.user_id == 3]) <= 2
