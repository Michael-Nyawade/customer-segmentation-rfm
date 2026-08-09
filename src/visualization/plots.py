"""Plotting functions for RFM segmentation and clustering results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_segment_counts(rfm, figures_dir: str, filename: str = "customer_segments_chart.png"):
    """Bar chart of customer counts per rule-based segment.

    Parameters
    ----------
    rfm : pd.DataFrame
        Must contain a Segment column.
    figures_dir : str
        Directory to save the figure into.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figures_path = _ensure_dir(figures_dir)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.countplot(
        data=rfm, x="Segment", order=rfm["Segment"].value_counts().index,
        hue="Segment", palette="Set2", legend=False, ax=ax,
    )
    ax.set_title("Number of Customers per Segment")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Customer Count")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(figures_path / filename, dpi=300)

    return fig


def plot_rfm_distribution_by_segment(
    rfm, figures_dir: str, filename: str = "rfm_distribution_by_segment.png"
):
    """Boxplots of Recency, Frequency, Monetary distributions by segment.

    Parameters
    ----------
    rfm : pd.DataFrame
        Must contain Recency, Frequency, Monetary, Segment columns.
    figures_dir : str
        Directory to save the figure into.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figures_path = _ensure_dir(figures_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.suptitle("Distribution of Recency, Frequency and Monetary Values by Segment\n")

    sns.boxplot(data=rfm, x="Segment", y="Recency", hue="Segment", palette="Set3", legend=False, ax=axes[0])
    axes[0].set_title("Recency by Segment")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(data=rfm, x="Segment", y="Frequency", hue="Segment", palette="Set2", legend=False, ax=axes[1])
    axes[1].set_title("Frequency by Segment")
    axes[1].tick_params(axis="x", rotation=45)

    sns.boxplot(data=rfm, x="Segment", y="Monetary", hue="Segment", palette="Set1", legend=False, ax=axes[2])
    axes[2].set_title("Monetary by Segment")
    axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(figures_path / filename, dpi=300)

    return fig


def plot_elbow(inertia: dict, figures_dir: str, filename: str = "optimal_no_of_clusters_elbow.png"):
    """Elbow plot from a k -> inertia mapping.

    Parameters
    ----------
    inertia : dict
        Output of src.models.train.compute_elbow_inertia.
    figures_dir : str
        Directory to save the figure into.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figures_path = _ensure_dir(figures_dir)

    ks = sorted(inertia)
    values = [inertia[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, values, marker="o")
    ax.set_title("Elbow Method - Optimal k\n")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.grid(True)
    fig.savefig(figures_path / filename)

    return fig


def plot_silhouette(
    scores: dict, figures_dir: str, filename: str = "optimal_no_of_clusters_silhouette.png"
):
    """Silhouette score plot from a k -> score mapping.

    Parameters
    ----------
    scores : dict
        Output of src.models.train.compute_silhouette_scores.
    figures_dir : str
        Directory to save the figure into.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figures_path = _ensure_dir(figures_dir)

    ks = sorted(scores)
    values = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, values, marker="o", color="green")
    ax.set_title("Silhouette Scores for k\n")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True)
    fig.savefig(figures_path / filename)

    return fig


def plot_clusters_pca(
    scaled_features, cluster_labels, figures_dir: str, filename: str = "cluster_plot.png"
):
    """2D PCA scatter plot of clusters.

    Parameters
    ----------
    scaled_features : np.ndarray
        Scaled RFM features, same rows/order as cluster_labels.
    cluster_labels : pd.Series or array-like
        Descriptive cluster label per row (e.g. rfm["Cluster_Label"]).
    figures_dir : str
        Directory to save the figure into.
    filename : str
        Output filename.

    Returns
    -------
    matplotlib.figure.Figure
    """

    figures_path = _ensure_dir(figures_dir)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features)

    plot_df = pd.DataFrame({
        "PCA1": components[:, 0],
        "PCA2": components[:, 1],
        "Segment": pd.Series(cluster_labels).values,
    })

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(data=plot_df, x="PCA1", y="PCA2", hue="Segment", palette="Set2", s=60, ax=ax)
    ax.set_title("Customer Segments by K-Means Clustering (PCA Reduced)\n")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(title="Segment")
    fig.tight_layout()
    fig.savefig(figures_path / filename, dpi=300)

    return fig
