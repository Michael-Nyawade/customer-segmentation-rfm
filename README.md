# Customer Segmentation Using RFM Analysis and K-Means Clustering

## Project Overview

This project segments customers based on purchasing behavior using RFM
(Recency, Frequency, Monetary) analysis combined with K-Means clustering.
It applies both a rule-based scoring approach and unsupervised clustering
to the UCI Online Retail dataset, producing customer segments that can
inform targeted marketing, retention, and re-engagement strategies.

## Dataset

- **Source:** [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Size:** ~540,000 transactions from a UK-based online retailer (Dec 2010 - Dec 2011)
- **Columns:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

Place the raw CSV at `data/raw/rfm_customer_segments.csv` before running
the pipeline (this path is git-ignored and not included in the repo).

## Project Structure

```bash
customer-segmentation-rfm/
├── config.yaml       # Central configuration (paths, RFM params, model params)
├── main.py           # End-to-end pipeline entry point
├── data/
│ ├── raw/            # Raw input CSV (git-ignored)
│ └── processed/      # Final enriched output CSV (git-ignored)
├── models/           # Trained model, scaler, cluster labels (git-ignored)
├── reports/
│ └── figures/        # Generated plots (git-ignored)
├── notebooks/        # Exploratory notebooks
├── src/
│ ├── data/           # Loading and cleaning
│ ├── features/       # RFM computation, scoring, rule-based segmentation
│ ├── models/         # Scaling, k-selection, training, persistence, prediction
│ └── visualization/  # Plotting functions
└── tests/            # Unit tests for each layer
```

## Setup

```bash
git clone <repo-url>
cd customer-segmentation-rfm

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the raw dataset at `data/raw/rfm_customer_segments.csv`.

## Running the Pipeline

```bash
python main.py
```

This runs the full pipeline: loads and cleans the raw data, computes RFM
metrics, applies rule-based scoring and segmentation, runs k-selection
diagnostics, trains the K-Means model, labels clusters, generates all
plots, saves model artifacts, and exports the final enriched customer
table to `data/processed/rfm_segments_with_clusters.csv`.

The model always retrains from scratch on each run, so results are
reproducible from raw data every time `main.py` is executed.

## Methodology

### 1. Data Cleaning
- Remove rows with missing `CustomerID`
- Remove cancelled orders (`InvoiceNo` starting with `C`)
- Remove non-positive `Quantity` or `UnitPrice`
- Derive `TotalPrice = Quantity * UnitPrice`

### 2. RFM Feature Engineering
- **Recency:** days since last purchase
- **Frequency:** number of unique invoices per customer
- **Monetary:** total amount spent

### 3. Rule-Based Segmentation
Each RFM metric is scored 1-5 via quantile binning (rank-transformed to
avoid duplicate bin-edge errors on skewed data). Combined scores are
mapped to segments: Champion, Loyal Customer, Big Spender, At Risk,
Lost, Others.

### 4. K-Means Clustering
- RFM values are scaled with `StandardScaler`
- Optimal `k` explored via Elbow and Silhouette methods (see
  `reports/figures/optimal_no_of_clusters_*.png`)
- Final model trained with `k=4` (configurable in `config.yaml`)
- Clusters are labeled by ranking mean Monetary value per cluster
  (VIP, Loyal High-Spender, Mid-Value, At Risk), rather than a hardcoded
  index mapping — this makes labeling robust to KMeans' arbitrary
  cluster index assignment across runs

## Predicting on New Data

Trained model artifacts (`models/kmeans_model.joblib`,
`models/scaler.joblib`, `models/cluster_labels.joblib`) can be used to
score new, unseen customer data without retraining:

```python
from src.models.predict import predict_from_rfm, predict_from_transactions

# If you already have Recency/Frequency/Monetary values:
result = predict_from_rfm(new_rfm_df, models_dir="models")

# Or starting from raw transaction rows:
result = predict_from_transactions(new_transactions_df, models_dir="models")
```

## Tools & Technologies

- **Data manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine learning:** scikit-learn (KMeans, StandardScaler, PCA, silhouette_score)
- **Testing:** pytest

## Tests

```bash
pytest tests/ -v
```