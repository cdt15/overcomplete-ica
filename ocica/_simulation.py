"""Synthetic data for validating overcomplete ICA."""

import numpy as np
from sklearn.utils import check_random_state


def make_mixing_matrix(n_features, n_sources, min_angle=15.0, max_tries=10000,
                       random_state=None):
    """Draw a random mixing matrix with unit-norm, well-separated columns.

    Columns are drawn until every pair makes an angle of at least
    ``min_angle`` degrees. Signs are ignored, so angles lie in [0, 90].

    Parameters
    ----------
    n_features : int
        Number of observed variables M.
    n_sources : int
        Number of sources L.
    min_angle : float, default=15.0
        Minimum angle between any two columns, in degrees.
    max_tries : int, default=10000
        Maximum number of rejection-sampling attempts.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator.

    Returns
    -------
    A : ndarray of shape (n_features, n_sources)
        Mixing matrix with unit-norm columns.

    Raises
    ------
    RuntimeError
        If no matrix satisfying ``min_angle`` is found within ``max_tries``.
    """
    rng = check_random_state(random_state)
    cos_max = np.cos(np.deg2rad(min_angle))
    for _ in range(max_tries):
        A = rng.standard_normal((n_features, n_sources))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        G = np.abs(A.T @ A)
        np.fill_diagonal(G, 0.0)
        if G.max() < cos_max:
            return A
    raise RuntimeError(
        f"no mixing matrix with min_angle={min_angle} found in {max_tries} tries"
    )


def make_overcomplete_data(n_samples, n_features, n_sources, noise_std=0.1,
                           min_angle=15.0, random_state=None):
    """Generate data from the model ``x = A s + eps``.

    The sources are unit-variance Laplacian, A is drawn by
    :func:`make_mixing_matrix` and eps ~ N(0, noise_std^2 I).

    Parameters
    ----------
    n_samples : int
        Number of samples N.
    n_features : int
        Number of observed variables M.
    n_sources : int
        Number of sources L.
    noise_std : float, default=0.1
        Standard deviation of the isotropic Gaussian noise.
    min_angle : float, default=15.0
        Minimum angle between columns of A, in degrees.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Observations.
    S : ndarray of shape (n_samples, n_sources)
        True sources.
    A : ndarray of shape (n_features, n_sources)
        True mixing matrix.
    """
    rng = check_random_state(random_state)
    A = make_mixing_matrix(n_features, n_sources, min_angle=min_angle, random_state=rng)
    S = rng.laplace(scale=1.0 / np.sqrt(2.0), size=(n_samples, n_sources))
    X = S @ A.T + noise_std * rng.standard_normal((n_samples, n_features))
    return X, S, A
