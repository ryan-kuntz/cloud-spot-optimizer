# Cloud Spot Instance Price Optimizer

> Helping ML teams answer one question: **Should I launch my training job now, or wait for prices to drop?**

---

## What is this?

Cloud computing lets you rent GPU servers by the hour — but the price fluctuates constantly based on supply and demand. "Spot instances" are discounted (up to 90% off) spare capacity that providers sell when their servers would otherwise sit idle. The catch: prices change every few minutes, and they vary by server type, region, and time of day.

This project builds a forecasting tool that predicts future spot prices and recommends the cheapest time and location to launch a GPU training job — so ML teams can cut compute costs without guessing.


---

## Key Findings
*[To be completed]*

---

## Approach

1. **Collected** historical spot price data across GPU instance types and AWS regions
2. **Explored** price patterns: time-of-day cycles, regional differences, volatility by instance type
3. **Engineered features** including lag prices, rolling averages, and time-based signals
4. **Trained a forecasting model** to predict prices 1–24 hours ahead
5. **Translated forecasts into recommendations**: given a job's runtime and GPU requirements, 
   output the cheapest available launch window

---

## Project Structure

| Path | What it does |
|----------|-------------|
| `src/ingest.py` | Loads raw AWS spot price data from Zenodo `.tsv.zst` files |
| `src/preprocess.py` | Filters to relevant regions, GPU instance types, and OS |
| `src/collect.py` | Merges external macro signals (natural gas prices via FRED) |
| `src/features.py` | Creates time-based (and, soon, lag) features for modelling |
| `src/train.py` | Trains, evaluates, and saves the forecasting model |
| `src/evaluate.py` | Evaluates model performance and reports feature importances |
| `notebooks/01_eda.ipynb` | Exploratory analysis and pattern identification |
| `notebooks/02_feature_engineering.ipynb` | Documents feature engineering decisions |
| `app/app.py` | Streamlit dashboard (in progress — see open issues) |

---

## Data Sources

- **AWS Spot Price History** via the [Zenodo dataset](https://zenodo.org/records/18821638) (raw `.tsv.zst` monthly snapshots)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/ryan-kuntz/cloud-job-scheduler.git
cd cloud-job-scheduler

# Create and activate virtual environment
python -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard (in progress — see open issues)
streamlit run app/app.py
```

Data is not included in this repo due to size.

## Running the pipeline

Raw data must be downloaded from Zenodo before running the pipeline:
- [AWS Spot Price History Dataset](https://zenodo.org/records/18821638)
- Download the desired monthly `.tsv.zst` files into `data/raw/`

```bash
python src/ingest.py       # loads .tsv.zst files → spot_prices_raw.parquet
python src/preprocess.py   # filters → spot_prices_filtered.parquet
python src/collect.py      # merges FRED data → base_dataset.parquet
python src/features.py     # Feature engineering — transforms raw price data into model inputs
python src/train.py         # Train and save the forecasting model
python src/evaluate.py      # Evaluate model performance
```

## Exploring the analysis

To walk through the EDA and feature engineering notebooks:

```bash
jupyter notebook
```

Open `notebooks/01_eda.ipynb` to start.

---

## Known Limitations 

- Currently scoped to a subset of GPU instance types and AWS regions
- Model does not yet account for interruption probability (only price)
