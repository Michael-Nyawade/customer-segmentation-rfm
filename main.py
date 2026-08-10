"""End-to-end pipeline: load, clean, engineer features, train, predict, visualize, export."""

from src.data.load_data import load_config, load_raw_data
from src.data.preprocess import clean_transactions
from src.features.feature_engineering import assign_segments, compute_rfm, score_rfm
from src.models.train import (
    compute_elbow_inertia,
    compute_silhouette_scores,
    fit_kmeans,
    label_clusters_by_value,
    profile_clusters,
    save_model_artifacts,
    scale_rfm_features,
)
from src.visualization.plots import (
    plot_clusters_pca,
    plot_elbow,
    plot_rfm_distribution_by_segment,
    plot_segment_counts,
    plot_silhouette,
)


def run_pipeline(config_path: str = "config.yaml") -> None:
    """Run the full RFM segmentation and clustering pipeline end-to-end."""
    config = load_config(config_path)

    print("Loading raw data...")
    raw = load_raw_data(config)

    print("Cleaning transactions...")
    clean = clean_transactions(raw)

    print("Computing RFM metrics...")
    rfm = compute_rfm(clean, config["rfm"]["reference_date_offset_days"])
    rfm = score_rfm(rfm, config["rfm"]["n_bins"])
    rfm = assign_segments(rfm)

    figures_dir = config["reports"]["figures_dir"]

    print("Plotting rule-based segment charts...")
    plot_segment_counts(rfm, figures_dir)
    plot_rfm_distribution_by_segment(rfm, figures_dir)

    print("Scaling features for clustering...")
    scaled, scaler = scale_rfm_features(rfm)

    print("Running k-selection diagnostics...")
    inertia = compute_elbow_inertia(scaled, random_state=config["model"]["random_state"])
    plot_elbow(inertia, figures_dir)

    silhouette = compute_silhouette_scores(scaled, random_state=config["model"]["random_state"])
    plot_silhouette(silhouette, figures_dir)

    print(f"Training final KMeans model (k={config['model']['n_clusters']})...")
    kmeans = fit_kmeans(
        scaled,
        n_clusters=config["model"]["n_clusters"],
        random_state=config["model"]["random_state"],
    )
    rfm["Cluster"] = kmeans.predict(scaled)

    print("Profiling and labeling clusters...")
    cluster_summary = profile_clusters(rfm)
    cluster_labels = label_clusters_by_value(cluster_summary)
    rfm["Cluster_Label"] = rfm["Cluster"].map(cluster_labels)

    plot_clusters_pca(scaled, rfm["Cluster_Label"], figures_dir)

    print("Saving model artifacts...")
    save_model_artifacts(kmeans, scaler, cluster_labels, config["models"]["dir"])

    print("Exporting enriched customer table...")
    output_path = f"{config['data']['processed']}/{config['data']['output_filename']}"
    rfm.to_csv(output_path)

    print(f"Pipeline complete. Output saved to {output_path}")
    print(f"Cluster summary:\n{cluster_summary}")
    print(f"Cluster labels: {cluster_labels}")


if __name__ == "__main__":
    run_pipeline()