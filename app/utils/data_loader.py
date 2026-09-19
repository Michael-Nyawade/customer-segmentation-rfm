from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "rfm_segments_with_clusters.csv"
)

def load_dashboard_data():

    return pd.read_csv(
        DATA_PATH,
        index_col="CustomerID"
    )