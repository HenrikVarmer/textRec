import numpy as np
import pytest

from textrec import jensen_shannon_divergence, jsd_matrix


def test_identical_distributions_are_zero():
    assert jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)


def test_disjoint_support_is_ln2():
    assert jensen_shannon_divergence([1, 0], [0, 1]) == pytest.approx(np.log(2))


def test_symmetry():
    a = jensen_shannon_divergence([0.7, 0.3], [0.2, 0.8])
    b = jensen_shannon_divergence([0.2, 0.8], [0.7, 0.3])
    assert a == pytest.approx(b)


def test_handles_zeros_without_nan():
    val = jensen_shannon_divergence([0.5, 0.5, 0.0], [0.0, 0.5, 0.5])
    assert np.isfinite(val)


def test_bounded_by_ln2():
    val = jensen_shannon_divergence([0.9, 0.1], [0.1, 0.9])
    assert 0.0 <= val <= np.log(2) + 1e-9


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        jensen_shannon_divergence([0.5, 0.5], [1.0])


def test_matrix_symmetric_zero_diagonal():
    m = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
    D = jsd_matrix(m)
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0, atol=1e-12)
    assert np.allclose(D, D.T, atol=1e-12)


def test_matrix_matches_scalar():
    m = np.array([[0.6, 0.4], [0.1, 0.9]])
    D = jsd_matrix(m)
    assert D[0, 1] == pytest.approx(jensen_shannon_divergence(m[0], m[1]))
