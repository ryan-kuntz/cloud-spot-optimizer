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

| Notebook | What it does |
|----------|-------------|
| 01_data_ingestion | Loading and filtering raw AWS spot price data |
| 02_data_collection | Merging external data sources |
| 03_eda | Exploratory analysis and pattern identification |
| 04_feature_engineering | Creating time-based and lag features for modelling |
| 05_modelling | Training, evaluating, and tuning the forecasting model |
| 06_conclusions | Recommendations and business implications |

---

## Data Sources

- **AWS Spot Price History** via the Boto3 API / [Zenodo dataset](Add link)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/ryan-kuntz/cloud-spot-optimizer.git
cd cloud-spot-optimizer

# Create and activate virtual environment
python -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
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
