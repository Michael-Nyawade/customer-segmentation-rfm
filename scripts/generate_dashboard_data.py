"""Generate a static JSON summary for the frontend dashboard.

Run locally after main.py produces data/processed/rfm_segments_with_clusters.csv.
Output is committed to app/data/dashboard.json for the static frontend to fetch.
"""

import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/rfm_segments_with_clusters.csv")
OUTPUT_PATH = Path("app/data/dashboard.json")


def main():
    rfm = pd.read_csv(INPUT_PATH, index_col="CustomerID")

    cluster_summary = (
        rfm.groupby("Cluster_Label")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(1)
    )
    cluster_summary["Count"] = rfm["Cluster_Label"].value_counts()

    dashboard_data = {
        "total_customers": len(rfm),
        "cluster_counts": rfm["Cluster_Label"].value_counts().to_dict(),
        "segment_counts": rfm["Segment"].value_counts().to_dict(),
        "cluster_summary": [
            {
                "label": label,
                "recency": row["Recency"],
                "frequency": row["Frequency"],
                "monetary": row["Monetary"],
                "count": int(row["Count"]),
            }
            for label, row in cluster_summary.iterrows()
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Dashboard data written to {OUTPUT_PATH}")
    print(json.dumps(dashboard_data, indent=2))


if __name__ == "__main__":
    main()