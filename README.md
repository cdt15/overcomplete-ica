# Overcomplete ICA

Overcomplete ICA by the variational EM algorithm of Girolami (2001).

The package estimates `L >= M` independent Laplacian sources from `M`-dimensional observations under the linear model `x = A s + eps`, where `eps` is additive Gaussian noise with full covariance. The estimator follows the scikit-learn API (similar to `sklearn.decomposition.FastICA`).

## Requirements

- Python >= 3.9
- numpy
- scipy
- scikit-learn >= 1.6

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/cdt15/overcomplete-ica.git
```

For development, clone the repository and use an editable install together with the test dependency:

```bash
git clone https://github.com/cdt15/overcomplete-ica.git
cd overcomplete-ica
pip install -e .
pip install pytest
```

## Usage

```python
from ocica import (
    OvercompleteICA,
    make_overcomplete_data,
    match_components,
    align_sources,
    mixing_angle_error,
    source_snr,
)

# Generate 3 Laplacian sources mixed into 2 observed variables (overcomplete)
X, S, A = make_overcomplete_data(n_samples=1000, n_features=2, n_sources=3, random_state=0)

# Fit the model and estimate the sources
ica = OvercompleteICA(n_components=3, n_init=5, random_state=0)
S_hat = ica.fit_transform(X)      # shape (1000, 3)

ica.mixing_                       # estimated mixing matrix A, shape (2, 3)
ica.noise_covariance_             # estimated noise covariance
ica.inverse_transform(S_hat)      # reconstruction of X

# Evaluate the result against the ground truth
perm, signs, angles = match_components(A, ica.mixing_)
S_aligned = align_sources(S_hat, perm, signs)
print(mixing_angle_error(A, ica.mixing_))  # angle error per column (degrees)
print(source_snr(S, S_aligned))            # SNR per source (dB)
```

### `OvercompleteICA` parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `n_components` | `None` | Number of sources `L`. If `None`, the number of features is used (complete ICA). |
| `n_init` | `1` | Number of random initializations; the run with the highest lower bound is kept. |
| `max_iter` | `10` | Maximum number of EM iterations per initialization. |
| `tol` | `1e-6` | Convergence threshold on the per-sample lower bound. |
| `n_xi_updates` | `1` | Number of E-steps per EM iteration. |
| `w_init` | `None` | Initial mixing matrix of shape `(n_features, n_components)`. |
| `random_state` | `None` | Seed or generator for the random initialization. |

See the docstring of `OvercompleteICA` for all parameters and fitted attributes.

## Reference

M. Girolami, "A Variational Method for Learning Sparse and Overcomplete Representations", *Neural Computation* 13(11), 2517-2532, 2001.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
