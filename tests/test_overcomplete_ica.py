"""Tests for OvercompleteICA (faithful implementation) and the helper modules."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocica import (  # noqa: E402
    OvercompleteICA,
    align_sources,
    make_mixing_matrix,
    make_overcomplete_data,
    match_components,
    mixing_angle_error,
    source_correlation,
    source_snr,
)
from ocica._overcomplete_ica import _e_step, _update_v  # noqa: E402

C = np.sqrt(2.0)


def fixed_model(cls, A, noise_var):
    """Build a fitted-looking model with the given parameters, without fitting."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m = cls(A.shape[1], transform_max_iter=2000, transform_tol=1e-12)
    m.mixing_ = A
    m.noise_covariance_ = noise_var * np.eye(A.shape[0])
    m.mean_ = np.zeros(A.shape[0])
    m.n_features_in_ = A.shape[0]
    m._n_features_out = A.shape[1]
    return m


def quadrature_loglik(A, noise_var, x):
    """Exact log p(x) for M=1, L=2 by numerical integration on a grid."""
    g = np.linspace(-10, 10, 1201)
    dg = g[1] - g[0]
    G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    log_prior = (np.log(C / 2) - C * np.abs(G)).sum(1)
    P = G @ A[0]
    return np.array([
        logsumexp(-0.5 * (xi - P) ** 2 / noise_var + log_prior) + 2 * np.log(dg)
        - 0.5 * np.log(2 * np.pi * noise_var)
        for xi in x[:, 0]
    ])


# --------------------------------------------------------------- Verification of formulas
def test_e_step_matches_direct_formula():
    """Posterior moments equal the direct formula (Eqs. 3.5, 3.6)."""
    rng = np.random.default_rng(0)
    M, L, N = 3, 5, 7
    X = rng.standard_normal((N, M))
    A = rng.standard_normal((M, L))
    Sig = np.diag(rng.uniform(0.1, 0.5, M))
    V = rng.uniform(0.2, 2.0, (N, L))
    mean, post_cov, _ = _e_step(X, A, Sig, V)
    Si = np.linalg.inv(Sig)
    for n in range(N):
        P = np.linalg.inv(A.T @ Si @ A + np.diag(1.0 / V[n]))
        np.testing.assert_allclose(post_cov[n], P, atol=1e-10)
        np.testing.assert_allclose(mean[n], P @ A.T @ Si @ X[n], atol=1e-10)


def test_bound_formula_against_gaussian_marginal():
    """Bound = sum log phi(v) + log N(x; 0, A Lambda A^T + Sigma) (Eq. 3.3)."""
    rng = np.random.default_rng(1)
    M, L, N = 2, 3, 4
    X = rng.standard_normal((N, M))
    A = rng.standard_normal((M, L))
    Sig = 0.3 * np.eye(M)
    V = rng.uniform(0.2, 2.0, (N, L))
    _, _, bound = _e_step(X, A, Sig, V)
    for n in range(N):
        log_phi = np.sum(np.log(C / 2) - C**2 * V[n] / 2 + 0.5 * np.log(2 * np.pi * V[n]))
        ref = log_phi + multivariate_normal(np.zeros(M), A @ np.diag(V[n]) @ A.T + Sig).logpdf(X[n])
        assert bound[n] == pytest.approx(ref, abs=1e-10)


def test_bound_is_tight_for_complete_noise_free_case():
    """In the complete noise-free case the optimized bound equals the exact log-likelihood."""
    a = 1.7
    x = np.array([[-2.0], [-0.3], [0.1], [1.5]])
    m = fixed_model(OvercompleteICA, [[a]], noise_var=1e-12)
    exact = np.log(C / 2) - C * np.abs(x[:, 0] / a) - np.log(a)
    np.testing.assert_allclose(m.score_samples(x), exact, atol=1e-4)


def test_bound_is_below_exact_loglik():
    """For M=1, L=2 the bound never exceeds the exact log-likelihood."""
    A = np.array([[1.0, 0.6]])
    x = np.array([[-1.2], [0.05], [0.8], [2.5]])
    m = fixed_model(OvercompleteICA, A, noise_var=0.05)
    assert np.all(m.score_samples(x) <= quadrature_loglik(A, 0.05, x) + 1e-6)


def test_update_v_is_posterior_rms():
    """Variational update v = sqrt(E{s^2}) / c (Eq. 3.9)."""
    mean = np.array([[1.0, -2.0]])
    post_cov = np.array([[[0.5, 0.1], [0.1, 0.25]]])
    np.testing.assert_allclose(_update_v(mean, post_cov), np.sqrt([[1.5, 4.25]]) / C)


@pytest.mark.parametrize("n_xi_updates", [1, 2])
def test_lower_bound_monotonically_increases(n_xi_updates):
    """EM never decreases the variational lower bound."""
    X, _, _ = make_overcomplete_data(300, 3, 4, random_state=0)
    m = OvercompleteICA(4, n_xi_updates=n_xi_updates, max_iter=100, tol=0,
                        random_state=0).fit(X)
    assert np.all(np.diff(m.lower_bounds_) > -1e-9)


# --------------------------------------------------------------- Estimation performance
def test_stays_near_truth_from_true_initialization():
    """Started from the true mixing matrix, EM stays close to it."""
    X, S, A = make_overcomplete_data(2000, 2, 3, noise_std=0.05, min_angle=30, random_state=0)
    m = OvercompleteICA(3, w_init=A, random_state=0).fit(X)
    perm, signs, angles = match_components(A, m.mixing_)
    assert angles.max() < 10.0
    S_hat = align_sources(m.transform(X), perm, signs)
    assert source_correlation(S, S_hat).min() > 0.7


def test_complete_case():
    """With n_components=None the model is complete ICA."""
    X, _, A = make_overcomplete_data(2000, 2, 2, noise_std=0.05, min_angle=30, random_state=1)
    m = OvercompleteICA(n_init=3, random_state=0).fit(X)
    assert m.mixing_.shape == (2, 2)
    assert mixing_angle_error(A, m.mixing_).max() < 5.0


# --------------------------------------------------------------- API
def test_api_shapes_and_attributes():
    X, _, _ = make_overcomplete_data(200, 3, 5, random_state=0)
    m = OvercompleteICA(5, n_init=2, max_iter=50, random_state=0)
    assert m.fit(X) is m
    assert m.mixing_.shape == (3, 5)
    assert m.components_.shape == (5, 3)
    assert m.noise_covariance_.shape == (3, 3)
    assert m.mean_.shape == (3,)
    assert m.init_lower_bounds_.shape == (2,)
    assert m.best_init_ == int(np.argmax(m.init_lower_bounds_))
    assert m.lower_bound_ == m.init_lower_bounds_[m.best_init_]
    assert m.n_features_in_ == 3
    assert m.transform(X).shape == (200, 5)
    assert m.inverse_transform(m.transform(X)).shape == (200, 3)
    assert m.score_samples(X).shape == (200,)
    assert np.isfinite(m.score(X))
    assert list(m.get_feature_names_out()) == [f"overcompleteica{i}" for i in range(5)]


def test_fit_transform_consistent_with_transform():
    X, _, _ = make_overcomplete_data(500, 2, 3, random_state=0)
    m = OvercompleteICA(3, random_state=0)
    np.testing.assert_allclose(m.fit_transform(X), m.transform(X), atol=0.05)


def test_reproducible_with_random_state():
    X, _, _ = make_overcomplete_data(300, 2, 3, random_state=0)
    m1 = OvercompleteICA(3, n_init=3, max_iter=50, random_state=42).fit(X)
    m2 = OvercompleteICA(3, n_init=3, max_iter=50, random_state=42).fit(X)
    np.testing.assert_array_equal(m1.mixing_, m2.mixing_)


def test_w_init_shape_is_checked():
    X, _, A = make_overcomplete_data(100, 2, 3, random_state=0)
    with pytest.raises(ValueError):
        OvercompleteICA(3, w_init=A[:, :2]).fit(X)


def test_dataframe_feature_names():
    pd = pytest.importorskip("pandas")
    X, _, _ = make_overcomplete_data(200, 2, 3, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b"])
    m = OvercompleteICA(3, max_iter=20, random_state=0).fit(df)
    assert list(m.feature_names_in_) == ["a", "b"]
    assert m.transform(df).shape == (200, 3)


@pytest.mark.parametrize("kwargs", [
    dict(n_components=0), dict(n_init=0), dict(max_iter=0), dict(n_xi_updates=0),
    dict(tol=-1.0),
])
def test_invalid_params(kwargs):
    X, _, _ = make_overcomplete_data(50, 2, 3, random_state=0)
    with pytest.raises(ValueError):
        OvercompleteICA(**kwargs).fit(X)


def test_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        OvercompleteICA(3).transform(np.zeros((2, 2)))


def test_transform_wrong_n_features_raises():
    X, _, _ = make_overcomplete_data(100, 2, 3, random_state=0)
    m = OvercompleteICA(3, max_iter=10, random_state=0).fit(X)
    with pytest.raises(ValueError):
        m.transform(np.zeros((5, 3)))


def test_sklearn_estimator_checks():
    """The estimator passes all scikit-learn estimator checks."""
    from sklearn.utils.estimator_checks import check_estimator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = check_estimator(OvercompleteICA(max_iter=20, random_state=0), on_fail=None)
    assert [r["check_name"] for r in results if r["status"] == "failed"] == []


# --------------------------------------------------------------- Helper functions
def test_match_components_with_permutation_and_sign():
    A = make_mixing_matrix(3, 5, random_state=0)
    perm_true = np.array([2, 0, 4, 1, 3])
    signs_true = np.array([1, -1, -1, 1, 1])
    A_est = np.zeros_like(A)
    A_est[:, perm_true] = A * signs_true * 2.0
    perm, signs, angles = match_components(A, A_est)
    np.testing.assert_array_equal(perm, perm_true)
    np.testing.assert_array_equal(signs, signs_true)
    np.testing.assert_allclose(angles, 0.0, atol=1e-5)


def test_source_metrics():
    rng = np.random.default_rng(0)
    S = rng.laplace(size=(1000, 3))
    assert np.all(source_snr(S, 0.5 * S) > 100)
    np.testing.assert_allclose(source_correlation(S, 2 * S), 1.0)
    snr = source_snr(S, S + 0.1 * rng.standard_normal(S.shape))
    assert np.all((snr > 15) & (snr < 30))


def test_make_overcomplete_data():
    X, S, A = make_overcomplete_data(5000, 3, 5, noise_std=0.0, min_angle=20, random_state=0)
    assert X.shape == (5000, 3) and S.shape == (5000, 5) and A.shape == (3, 5)
    np.testing.assert_allclose(np.linalg.norm(A, axis=0), 1.0)
    G = np.abs(A.T @ A)
    np.fill_diagonal(G, 0)
    assert np.rad2deg(np.arccos(G.max())) >= 20
    np.testing.assert_allclose(S.var(axis=0), 1.0, atol=0.15)
    np.testing.assert_allclose(X, S @ A.T)
