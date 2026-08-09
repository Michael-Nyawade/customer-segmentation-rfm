"""Tests for src.visualization.plots."""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids needing a display
import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.plots import (
    plot_clusters_pca,
    plot_elbow,
    plot_rfm_distribution_by_segment,
    plot_segment_counts,
    plot_silhouette,
)


def _make_segmented_rfm():
    return pd.DataFrame({
        "Recency": [5, 40, 300, 10, 60, 250],
        "Frequency": [20, 5, 1, 15, 4, 2],
        "Monetary": [5000, 1200, 200, 4000, 900, 300],
        "Segment": ["Champion", "Others", "At Risk", "Champion", "Others", "At Risk"],
    })


def test_plot_segment_counts_creates_file():
    rfm = _make_segmented_rfm()
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_segment_counts(rfm, tmp_dir, filename="test_segments.png")
        assert (Path(tmp_dir) / "test_segments.png").exists()
        assert fig is not None
        plt.close(fig)


def test_plot_rfm_distribution_by_segment_creates_file():
    rfm = _make_segmented_rfm()
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_rfm_distribution_by_segment(rfm, tmp_dir, filename="test_dist.png")
        assert (Path(tmp_dir) / "test_dist.png").exists()
        plt.close(fig)


def test_plot_elbow_creates_file():
    inertia = {1: 1000, 2: 600, 3: 400, 4: 300}
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_elbow(inertia, tmp_dir, filename="test_elbow.png")
        assert (Path(tmp_dir) / "test_elbow.png").exists()
        plt.close(fig)


def test_plot_silhouette_creates_file():
    scores = {2: 0.5, 3: 0.55, 4: 0.6, 5: 0.58}
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_silhouette(scores, tmp_dir, filename="test_silhouette.png")
        assert (Path(tmp_dir) / "test_silhouette.png").exists()
        plt.close(fig)


def test_plot_clusters_pca_creates_file():
    import numpy as np
    scaled_features = np.random.rand(6, 3)
    cluster_labels = pd.Series(["A", "A", "B", "B", "C", "C"])
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plot_clusters_pca(scaled_features, cluster_labels, tmp_dir, filename="test_pca.png")
        assert (Path(tmp_dir) / "test_pca.png").exists()
        plt.close(fig)