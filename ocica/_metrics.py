"""Evaluation metrics that account for the permutation and sign ambiguity."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_components(A_true, A_est):
    """Match the estimated columns of a mixing matrix to the true columns.

    Columns are normalized and the assignment maximizing the sum of |cos| is
    found with the Hungarian algorithm.

    Parameters
    ----------
    A_true : ndarray of shape (n_features, n_sources)
        True mixing matrix.
    A_est : ndarray of shape (n_features, n_sources)
        Estimated mixing matrix.

    Returns
    -------
    perm : ndarray of shape (n_sources,)
        ``perm[l]`` is the index of the estimated column matched to true column l.
    signs : ndarray of shape (n_sources,)
        Sign (+1 or -1) of each matched estimated column.
    angles : ndarray of shape (n_sources,)
        Angle in degrees between each true column and its matched estimate
        after sign alignment.
    """
    At = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
    Ae = A_est / np.linalg.norm(A_est, axis=0, keepdims=True)
    cos = At.T @ Ae
    row, col = linear_sum_assignment(-np.abs(cos))
    perm = col[np.argsort(row)]
    c = cos[np.arange(At.shape[1]), perm]
    signs = np.where(c >= 0, 1.0, -1.0)
    angles = np.rad2deg(np.arccos(np.clip(np.abs(c), 0.0, 1.0)))
    return perm, signs, angles


def mixing_angle_error(A_true, A_est):
    """Return the angle in degrees between matched columns.

    Parameters
    ----------
    A_true : ndarray of shape (n_features, n_sources)
        True mixing matrix.
    A_est : ndarray of shape (n_features, n_sources)
        Estimated mixing matrix.

    Returns
    -------
    angles : ndarray of shape (n_sources,)
        Same as the ``angles`` output of :func:`match_components`.
    """
    return match_components(A_true, A_est)[2]


def align_sources(S_est, perm, signs):
    """Reorder and flip estimated sources to match the true sources.

    Parameters
    ----------
    S_est : ndarray of shape (n_samples, n_sources)
        Estimated sources.
    perm : ndarray of shape (n_sources,)
        Permutation from :func:`match_components`.
    signs : ndarray of shape (n_sources,)
        Signs from :func:`match_components`.

    Returns
    -------
    S_aligned : ndarray of shape (n_samples, n_sources)
    """
    return S_est[:, perm] * signs


def source_snr(S_true, S_est):
    """Return the signal-to-noise ratio of each source in dB.

    SNR = 10 log10(||s||^2 / ||s - alpha s_hat||^2), where alpha is the
    least-squares scale (it compensates for the shrinkage of the posterior
    mean).

    Parameters
    ----------
    S_true : ndarray of shape (n_samples, n_sources)
        True sources.
    S_est : ndarray of shape (n_samples, n_sources)
        Estimated sources aligned with :func:`align_sources`.

    Returns
    -------
    snr : ndarray of shape (n_sources,)
    """
    num = (S_true**2).sum(axis=0)
    alpha = (S_true * S_est).sum(axis=0) / np.maximum((S_est**2).sum(axis=0), 1e-300)
    err = ((S_true - alpha * S_est) ** 2).sum(axis=0)
    return 10.0 * np.log10(num / np.maximum(err, 1e-300))


def source_correlation(S_true, S_est):
    """Return the correlation coefficient of each source.

    Parameters
    ----------
    S_true : ndarray of shape (n_samples, n_sources)
        True sources.
    S_est : ndarray of shape (n_samples, n_sources)
        Estimated sources aligned with :func:`align_sources`.

    Returns
    -------
    corr : ndarray of shape (n_sources,)
    """
    St = S_true - S_true.mean(axis=0)
    Se = S_est - S_est.mean(axis=0)
    return (St * Se).sum(axis=0) / np.sqrt((St**2).sum(axis=0) * (Se**2).sum(axis=0))
