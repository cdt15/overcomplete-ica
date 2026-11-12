"""Overcomplete ICA by the variational EM algorithm of Girolami (2001)."""

from ._overcomplete_ica import OvercompleteICA
from ._metrics import (
    align_sources,
    match_components,
    mixing_angle_error,
    source_correlation,
    source_snr,
)
from ._simulation import make_mixing_matrix, make_overcomplete_data

__all__ = [
    "OvercompleteICA",
    "align_sources",
    "match_components",
    "mixing_angle_error",
    "source_correlation",
    "source_snr",
    "make_mixing_matrix",
    "make_overcomplete_data",
]
