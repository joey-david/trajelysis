"""Provide shared run-record I/O, dimensionality reduction, and sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from reasoning_trajectory.artifacts import read_generation_rows, read_sample_records


T = TypeVar("T")


def project_3d(
    x: np.ndarray,
    *,
    random_state: int | None = None,
    tsne_perplexity: int = 30,
) -> dict[str, np.ndarray]:
    """Project a feature matrix into PCA and t-SNE three-dimensional spaces."""
    return {
        "pca": PCA(n_components=3, random_state=random_state).fit_transform(x),
        "tsne": TSNE(
            n_components=3,
            perplexity=min(tsne_perplexity, len(x) - 1),
            init="random",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(x),
    }


def evenly_capped(items: list[T], max_items: int) -> list[T]:
    """Downsample an ordered list at evenly spaced positions."""
    if max_items <= 0 or len(items) <= max_items:
        return items
    keep = np.linspace(0, len(items) - 1, max_items, dtype=int)
    return [items[int(i)] for i in keep]


__all__ = [
    "evenly_capped",
    "project_3d",
    "read_generation_rows",
    "read_sample_records",
]
