"""Overcomplete ICA by the variational EM algorithm of Girolami (2001)."""

import numbers

import numpy as np
from scipy import linalg
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

# Scale parameter of the unit-variance Laplace distribution p(s) = (c/2) exp(-c|s|)
_C = np.sqrt(2.0)
_LOG_2PI = np.log(2.0 * np.pi)
# Lower bound of the variational parameters (prior variances). At 0 the posterior variance collapses and becomes numerically unstable.
_V_FLOOR = 1e-12
# Lower bound of the noise variance (relative to the mean data variance), for numerical stability.
_MIN_NOISE_VAR = 1e-6
# Ratio of the initial noise covariance to the data covariance
_INIT_NOISE_RATIO = 0.1


def _e_step(X, A, noise_cov, V, compute_bound=True):
    """Compute the posterior moments under the variational bound (Eqs. 3.5, 3.6).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered data.
    A : ndarray of shape (n_features, n_components)
        Mixing matrix.
    noise_cov : ndarray of shape (n_features, n_features)
        Noise covariance Sigma.
    V : ndarray of shape (n_samples, n_components)
        Variational parameters (diagonal of Lambda_n, i.e. the variances of the
        variational Gaussian prior).
    compute_bound : bool, default=True
        Whether to also compute the lower bound on the log-likelihood.

    Returns
    -------
    mean : ndarray of shape (n_samples, n_components)
        Posterior means E{s | x_n}.
    post_cov : ndarray of shape (n_samples, n_components, n_components)
        Posterior covariances (A^T Sigma^-1 A + Lambda_n^-1)^-1.
    bound : ndarray of shape (n_samples,) or None
        Lower bound on log p(x_n) (logarithm of Eq. 3.3).
    """
    n_features = X.shape[1]
    AV = A[None, :, :] * V[:, None, :]  # A Λ_n
    C = AV @ A.T + noise_cov[None]  # C_n = A Λ_n A^T + Σ
    L_chol = np.linalg.cholesky(C)
    # K_n = C_n^-1 A Λ_n.  Woodbury: (A^T Σ^-1 A + Λ^-1)^-1 = Λ - Λ A^T C^-1 A Λ
    K = np.linalg.solve(C, AV)
    mean = np.einsum("nml,nm->nl", K, X)
    post_cov = V[:, :, None] * np.eye(V.shape[1])[None] - np.swapaxes(AV, 1, 2) @ K

    bound = None
    if compute_bound:
        # log N(x_n; 0, C_n)
        z = np.linalg.solve(L_chol, X[:, :, None])[..., 0]
        logdet = 2.0 * np.log(np.diagonal(L_chol, axis1=1, axis2=2)).sum(axis=1)
        log_gauss = -0.5 * (n_features * _LOG_2PI + logdet + (z**2).sum(axis=1))
        # Σ_l log[(c/2) exp(-c^2 v / 2) sqrt(2π v)]  (corresponds to the coefficient φ(ξ) of the variational lower bound)
        log_phi = (
            np.log(_C / 2.0) - 0.5 * _C**2 * V + 0.5 * (_LOG_2PI + np.log(V))
        ).sum(axis=1)
        bound = log_gauss + log_phi
    return mean, post_cov, bound


def _update_v(mean, post_cov):
    """Update the variational parameters (Eq. 3.9 / B.2).

    Parameters
    ----------
    mean : ndarray of shape (n_samples, n_components)
        Posterior means.
    post_cov : ndarray of shape (n_samples, n_components, n_components)
        Posterior covariances.

    Returns
    -------
    V : ndarray of shape (n_samples, n_components)
        v = sqrt(E{s^2 | x}) / c.
    """
    second = np.diagonal(post_cov, axis1=1, axis2=2) + mean**2
    return np.maximum(np.sqrt(second) / _C, _V_FLOOR)


def _m_step(X, XtX, mean, post_cov):
    """Unconstrained M-step for the mixing matrix and the noise covariance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Centered data.
    XtX : ndarray of shape (n_features, n_features)
        X^T X.
    mean : ndarray of shape (n_samples, n_components)
        Posterior means.
    post_cov : ndarray of shape (n_samples, n_components, n_components)
        Posterior covariances.

    Returns
    -------
    A : ndarray of shape (n_features, n_components)
        Updated mixing matrix (Eq. 3.7).
    resid_cov : ndarray of shape (n_features, n_features)
        Updated full noise covariance before flooring (Eq. 3.8).
    """
    Ess = post_cov.sum(axis=0) + mean.T @ mean  # Σ_n E{ss^T | x_n}
    XS = X.T @ mean  # Σ_n x_n E{s | x_n}^T
    A = linalg.solve(Ess, XS.T, assume_a="pos").T
    resid_cov = (XtX - A @ XS.T) / X.shape[0]
    return A, 0.5 * (resid_cov + resid_cov.T)


def _floor_eigenvalues(S, min_var):
    """Clip the eigenvalues of a symmetric matrix from below.

    Parameters
    ----------
    S : ndarray of shape (n_features, n_features)
        Symmetric matrix.
    min_var : float
        Lower bound on the eigenvalues.

    Returns
    -------
    S_floored : ndarray of shape (n_features, n_features)
    """
    w, U = linalg.eigh(S)
    return (U * np.maximum(w, min_var)) @ U.T


def _random_mixing(cov, n_components, random_state):
    """Draw a random mixing matrix consistent with the data covariance.

    Returns A0 = sqrt(1 - r) Cov^{1/2} R, where R has orthonormal rows when
    L >= M (so that A0 A0^T = (1 - r) Cov) and orthonormal columns otherwise.

    Parameters
    ----------
    cov : ndarray of shape (n_features, n_features)
        Data covariance.
    n_components : int
        Number of sources L.
    random_state : RandomState
        Random number generator.

    Returns
    -------
    A : ndarray of shape (n_features, n_components)
    """
    n_features = cov.shape[0]
    # Random matrix R (M x L) with orthonormal rows (R R^T = I) if L >= M, orthonormal columns if L < M
    Q, _ = np.linalg.qr(random_state.standard_normal(
        (max(n_components, n_features), min(n_components, n_features))))
    R = Q.T if n_components >= n_features else Q
    w, U = linalg.eigh(cov)
    cov_sqrt = (U * np.sqrt(np.maximum(w, 0.0))) @ U.T
    return np.sqrt(1.0 - _INIT_NOISE_RATIO) * cov_sqrt @ R


class OvercompleteICA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Overcomplete ICA by variational EM (Girolami, 2001).

    Estimates L >= M independent Laplacian sources from M-dimensional
    observations under the linear model ``x = A s + eps`` with additive
    Gaussian noise of full covariance. The algorithm follows the paper:
    random initialization, the E-step of Eqs. 3.5-3.6 and the M-step of
    Eqs. 3.7-3.9. The API follows :class:`sklearn.decomposition.FastICA`.

    Parameters
    ----------
    n_components : int, default=None
        Number of sources L. If None, ``n_features`` is used (complete ICA).
    n_init : int, default=1
        Number of random initializations. The run with the highest
        variational lower bound is kept.
    max_iter : int, default=500
        Maximum number of EM iterations per initialization.
    tol : float, default=1e-6
        Convergence threshold on the increase of the per-sample lower bound.
    n_xi_updates : int, default=1
        Number of E-steps (each followed by an update of the variational
        parameters) per EM iteration. The simulation in the paper uses 2.
    w_init : array-like of shape (n_features, n_components), default=None
        Initial mixing matrix. If given, it is used for every initialization.
    transform_max_iter : int, default=200
        Maximum number of variational updates in :meth:`transform`.
    transform_tol : float, default=1e-6
        Convergence threshold (maximum relative change of the variational
        parameters) in :meth:`transform`.
    random_state : int, RandomState instance or None, default=None
        Seed or generator for the random initialization.
    verbose : int, default=0
        Print the bound per initialization (>= 1) or per iteration (>= 2).

    Attributes
    ----------
    mixing_ : ndarray of shape (n_features, n_components)
        Estimated mixing matrix A.
    components_ : ndarray of shape (n_components, n_features)
        Moore-Penrose pseudo-inverse of ``mixing_`` (for reference only).
        In the overcomplete case the source estimate is not a linear map
        of x, so :meth:`transform` does not use it.
    noise_covariance_ : ndarray of shape (n_features, n_features)
        Estimated noise covariance Sigma.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean of the training data.
    n_iter_ : int
        Number of EM iterations of the selected run.
    converged_ : bool
        Whether the selected run converged within ``tol``.
    lower_bound_ : float
        Final per-sample lower bound of the selected run.
    lower_bounds_ : ndarray of shape (n_iter_ + 1,)
        History of the per-sample lower bound of the selected run.
    init_lower_bounds_ : ndarray of shape (n_init,)
        Final per-sample lower bound of every initialization.
    best_init_ : int
        Index of the selected initialization.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit`. Defined only when X has
        feature names that are all strings.

    References
    ----------
    M. Girolami, "A Variational Method for Learning Sparse and Overcomplete
    Representations", Neural Computation 13(11), 2517-2532, 2001.

    Examples
    --------
    >>> from ocica import OvercompleteICA, make_overcomplete_data
    >>> X, S, A = make_overcomplete_data(1000, 2, 3, random_state=0)
    >>> ica = OvercompleteICA(n_components=3, random_state=0)
    >>> S_hat = ica.fit_transform(X)
    >>> S_hat.shape
    (1000, 3)
    """

    def __init__(
        self,
        n_components=None,
        *,
        n_init=1,
        max_iter=500,
        tol=1e-6,
        n_xi_updates=1,
        w_init=None,
        transform_max_iter=200,
        transform_tol=1e-6,
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_xi_updates = n_xi_updates
        self.w_init = w_init
        self.transform_max_iter = transform_max_iter
        self.transform_tol = transform_tol
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------     Validation
    _positive_int_params = ("max_iter", "n_init", "n_xi_updates", "transform_max_iter")
    _non_negative_params = ("tol", "transform_tol")

    def _check_params(self, n_features):
        """Validate the hyper-parameters and return the number of sources.

        Parameters
        ----------
        n_features : int
            Number of features.

        Returns
        -------
        n_components : int
        """
        n_components = self.n_components if self.n_components is not None else n_features
        if not isinstance(n_components, numbers.Integral) or n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {self.n_components!r}")
        for name in self._positive_int_params:
            val = getattr(self, name)
            if not isinstance(val, numbers.Integral) or val < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {val!r}")
        for name in self._non_negative_params:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)!r}")
        return n_components

    def _check_w_init(self, n_features, n_components):
        """Return ``w_init`` as an array, or None.

        Parameters
        ----------
        n_features : int
            Number of features.
        n_components : int
            Number of sources.

        Returns
        -------
        w_init : ndarray of shape (n_features, n_components) or None
        """
        if self.w_init is None:
            return None
        w_init = np.array(self.w_init, dtype=float)
        if w_init.shape != (n_features, n_components):
            raise ValueError(
                f"w_init must have shape {(n_features, n_components)}, got {w_init.shape}"
            )
        return w_init

    # ------------------------------------------------------------------     Training
    def _initialize(self, X, n_components, rng):
        """Initialize the mixing matrix, noise covariance and variational parameters.

        A is random (or ``w_init``), Sigma is a fraction of the data covariance and
        the variational parameters are drawn as |xi| with xi ~ N(0, 1), as in the
        simulation of the paper (Fig. 2).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Centered data.
        n_components : int
            Number of sources.
        rng : RandomState
            Random number generator.

        Returns
        -------
        A : ndarray of shape (n_features, n_components)
        noise_cov : ndarray of shape (n_features, n_features)
        V : ndarray of shape (n_samples, n_components)
        """
        cov = X.T @ X / X.shape[0]
        w_init = self._check_w_init(X.shape[1], n_components)
        A = w_init if w_init is not None else _random_mixing(cov, n_components, rng)
        noise_cov = _INIT_NOISE_RATIO * cov
        V = np.maximum(np.abs(rng.standard_normal((X.shape[0], n_components))), _V_FLOOR)
        return A, noise_cov, V

    def _update_noise(self, resid_cov, noise_cov, min_var):
        """Return the noise covariance after the M-step (full covariance, Eq. 3.8).

        Parameters
        ----------
        resid_cov : ndarray of shape (n_features, n_features)
            Unconstrained M-step estimate.
        noise_cov : ndarray of shape (n_features, n_features)
            Current noise covariance (unused here).
        min_var : float
            Lower bound on the eigenvalues.

        Returns
        -------
        noise_cov : ndarray of shape (n_features, n_features)
        """
        return _floor_eigenvalues(resid_cov, min_var)

    def _run_em(self, X, A, noise_cov, V, min_var):
        """Run variational EM from the given initial values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Centered data.
        A : ndarray of shape (n_features, n_components)
            Initial mixing matrix.
        noise_cov : ndarray of shape (n_features, n_features)
            Initial noise covariance.
        V : ndarray of shape (n_samples, n_components)
            Initial variational parameters.
        min_var : float
            Absolute lower bound on the noise variance.

        Returns
        -------
        result : dict
            Keys ``A``, ``noise_cov``, ``mean``, ``post_cov``, ``bounds``,
            ``n_iter`` and ``converged``.
        """
        XtX = X.T @ X
        bounds = []
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # E-step. If n_xi_updates > 1, update the variational parameters first with A and Σ fixed
            for k in range(self.n_xi_updates):
                mean, post_cov, bound = _e_step(X, A, noise_cov, V, compute_bound=(k == 0))
                if k == 0:
                    bounds.append(bound.mean())
                if k < self.n_xi_updates - 1:
                    V = _update_v(mean, post_cov)

            # M-step (Eqs. 3.7, 3.8, 3.9 of the paper)
            A, resid_cov = _m_step(X, XtX, mean, post_cov)
            noise_cov = self._update_noise(resid_cov, noise_cov, min_var)
            V = _update_v(mean, post_cov)

            if self.verbose > 1:
                print(f"  iter {n_iter}: lower bound = {bounds[-1]:.6f}")
            if n_iter > 1 and abs(bounds[-1] - bounds[-2]) < self.tol:
                converged = True
                break

        mean, post_cov, bound = _e_step(X, A, noise_cov, V)
        bounds.append(bound.mean())
        return dict(A=A, noise_cov=noise_cov, mean=mean, post_cov=post_cov,
                    bounds=np.asarray(bounds), n_iter=n_iter, converged=converged)

    def _score_run(self, X, result, rng):
        """Return the criterion used to select among initializations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Centered data.
        result : dict
            Output of :meth:`_run_em`.
        rng : RandomState
            Random number generator.

        Returns
        -------
        score : float
            Final per-sample lower bound.
        """
        return result["bounds"][-1]

    def _fit(self, X):
        """Fit the model and return the posterior means of the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
        """
        X = validate_data(self, X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        X = np.asarray(X, dtype=np.float64)
        n_components = self._check_params(X.shape[1])

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        min_var = _MIN_NOISE_VAR * max(Xc.var(axis=0).mean(), np.finfo(float).tiny)

        rng = check_random_state(self.random_state)
        best, best_score = None, -np.inf
        init_bounds, init_scores = [], []
        for i in range(self.n_init):
            A, noise_cov, V = self._initialize(Xc, n_components, rng)
            res = self._run_em(Xc, A, noise_cov, V, min_var)
            score = self._score_run(Xc, res, rng)
            init_bounds.append(res["bounds"][-1])
            init_scores.append(score)
            if self.verbose:
                print(f"init {i + 1}/{self.n_init}: lower bound = {res['bounds'][-1]:.6f}, "
                      f"score = {score:.6f}, n_iter = {res['n_iter']}, "
                      f"converged = {res['converged']}")
            if best is None or score > best_score:
                best, best_score = res, score
                self.best_init_ = i

        self.mixing_ = best["A"]
        self.components_ = linalg.pinv(best["A"])
        self.noise_covariance_ = best["noise_cov"]
        self.n_iter_ = best["n_iter"]
        self.converged_ = best["converged"]
        self.lower_bounds_ = best["bounds"]
        self.lower_bound_ = float(best["bounds"][-1])
        self.init_lower_bounds_ = np.asarray(init_bounds)
        self._init_scores = np.asarray(init_scores)
        self._n_features_out = n_components
        return best["mean"]

    def fit(self, X, y=None):
        """Estimate the mixing matrix A and the noise covariance Sigma from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model and return the posterior means of the training sources.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
            Posterior means E{s | x}.
        """
        return self._fit(X)

    # ------------------------------------------------------------    ------ Inference
    def _infer(self, X):
        """Iterate the variational parameters with A and Sigma fixed.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Uncentered data.

        Returns
        -------
        mean : ndarray of shape (n_samples, n_components)
            Posterior means.
        post_cov : ndarray of shape (n_samples, n_components, n_components)
            Posterior covariances.
        bound : ndarray of shape (n_samples,)
            Lower bound on log p(x_n).
        """
        Xc = X - self.mean_
        # Start from v = 1/c, which corresponds to prior variance 1 (to make inference deterministic)
        V = np.full((Xc.shape[0], self.mixing_.shape[1]), 1.0 / _C)
        for _ in range(self.transform_max_iter):
            mean, post_cov, _ = _e_step(Xc, self.mixing_, self.noise_covariance_, V,
                                        compute_bound=False)
            V_new = _update_v(mean, post_cov)
            change = np.max(np.abs(V_new - V) / np.maximum(V, 1e-8))
            V = V_new
            if change < self.transform_tol:
                break
        return _e_step(Xc, self.mixing_, self.noise_covariance_, V)

    def transform(self, X, copy=True):
        """Return the posterior means E{s | x} of the sources.

        Under the variational bound the posterior is Gaussian, so the posterior
        mean coincides with the MAP estimate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.
        copy : bool, default=True
            Present for API consistency; the input is never modified.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
            Estimated sources.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False, copy=copy)
        mean, _, _ = self._infer(np.asarray(X, dtype=np.float64))
        return mean

    def inverse_transform(self, X, copy=True):
        """Reconstruct the observations, ``X = S A^T + mean_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Sources.
        copy : bool, default=True
            Whether to copy the input.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Reconstructed observations.
        """
        check_is_fitted(self)
        S = np.array(X, dtype=np.float64, copy=copy)
        if S.ndim != 2 or S.shape[1] != self.mixing_.shape[1]:
            raise ValueError(
                f"X must be a 2D array with n_components={self.mixing_.shape[1]} columns"
            )
        return S @ self.mixing_.T + self.mean_

    def score_samples(self, X):
        """Return the variational lower bound on the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        Returns
        -------
        bound : ndarray of shape (n_samples,)
            Lower bound on log p(x_n | A, Sigma).
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        _, _, bound = self._infer(np.asarray(X, dtype=np.float64))
        return bound

    def score(self, X, y=None):
        """Return the mean variational lower bound over the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        score : float
        """
        return float(np.mean(self.score_samples(X)))
