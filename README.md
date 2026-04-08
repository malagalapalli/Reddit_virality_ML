# Predicting Social Media Content Virality Using Distributed Machine Learning on Reddit Big Data

## Overview
This project applies **distributed machine learning** using **Apache Spark (PySpark MLlib)** to predict whether Reddit posts will go viral, trained on the **webis/tldr-17** dataset (~18GB, 3M+ posts). It compares PySpark with scikit-learn, performs hyperparameter tuning via CrossValidator, and includes comprehensive evaluation with bootstrap confidence intervals and scalability analysis.

## Dataset
- **Source**: [webis/tldr-17](https://huggingface.co/datasets/webis/tldr-17) (HuggingFace)
- **Size**: ~18GB CSV, 3M+ Reddit posts
- **Columns**: author, body, normalizedBody, subreddit, subreddit_id, id, content, summary
- **Virality Definition**: Posts from top 20% most active subreddits → `is_viral = 1`

## How It Works (End-to-End)
1. **Ingest**: Load the Reddit CSV into Spark DataFrames.
2. **Explore (EDA)**: Measure missing values, subreddit distribution, class balance, and text-length behavior.
3. **Engineer Data**: Convert CSV to Parquet and prepare a stratified sample for faster experimentation.
4. **Preprocess**: Build a PySpark pipeline with text cleaning, TF-IDF, numeric features, and subreddit indexing.
5. **Train Models**: Fit multiple PySpark classifiers and compare with scikit-learn baselines.
6. **Tune**: Run hyperparameter search (CrossValidator + ParamGrid) on top models.
7. **Evaluate**: Compute classification metrics, confusion matrix, ROC, and bootstrap confidence intervals.
8. **Analyze Scalability**: Run strong/weak scaling experiments.
9. **Export**: Save analytics tables/figures to `tableau/data/` for dashboarding.

## Project Structure
```
reddit-virality-project/
├── config/
│   ├── spark_config.yaml          # Spark session & data path config
│   └── project_config.yaml        # Project metadata & model params
├── data/
│   ├── raw/reddit_posts.csv       # Original dataset (~18GB)
│   ├── parquet/                   # Converted Parquet files
│   ├── partitioned/               # Partitioned by is_viral
│   ├── sample/                    # 10% stratified sample
│   └── models/                    # Saved ML models
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_data_engineering.ipynb  # CSV→Parquet, cleaning, sampling
│   ├── 03_preprocessing.ipynb     # Feature engineering & pipeline
│   ├── 04_pyspark_modeling.ipynb  # PySpark MLlib (4 models)
│   ├── 05_sklearn_comparison.ipynb# scikit-learn comparison
│   ├── 06_hyperparameter_tuning.ipynb # CrossValidator + ParamGrid
│   ├── 07_evaluation.ipynb        # ROC, confusion matrix, bootstrap CI
│   ├── 08_scalability_analysis.ipynb  # Strong/weak scaling
│   └── 09_tableau_export.ipynb    # Tableau dashboard data export
├── scripts/
│   ├── __init__.py
│   ├── spark_utils.py             # Spark session & config utilities
│   ├── custom_transformer.py      # Custom PySpark Transformers
│   └── feature_engineering.py     # ML pipeline builder
├── tableau/
│   └── data/                      # CSV exports for Tableau dashboards
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Unit tests (structure, config, imports)
│   └── test_feature_engineering.py# Integration tests (PySpark transforms)
├── download_data.py               # Dataset download script
├── environment.yml                # Conda environment spec
└── README.md
```

## Models
| Model | Framework | Description |
|---|---|---|
| Logistic Regression | PySpark MLlib | Baseline linear classifier |
| Random Forest | PySpark MLlib | Ensemble of decision trees |
| Decision Tree | PySpark MLlib | Single tree classifier |
| Gradient Boosted Trees | PySpark MLlib | Sequential boosted ensemble |

All four models are also trained with **scikit-learn** on a 200K-row sample for comparison.

## Feature Engineering
- **13 numeric features**: char_count, word_count, sentence_count, avg_word_length, unique_word_ratio, question_count, exclamation_count, uppercase_ratio, has_url, paragraph_count, summary_length, summary_word_count, content_density
- **5000 TF-IDF features**: Term frequency–inverse document frequency from post body text
- **1 categorical**: subreddit (StringIndexer)
- **Total**: 5014 feature dimensions

## Execution Order
1. `download_data.py` — Download dataset (already done)
2. `01_eda.ipynb` — Explore data distributions
3. `02_data_engineering.ipynb` — Convert to Parquet, create sample
4. `03_preprocessing.ipynb` — Build & fit preprocessing pipeline
5. `04_pyspark_modeling.ipynb` — Train 4 PySpark models
6. `05_sklearn_comparison.ipynb` — Train 4 sklearn models for comparison
7. `06_hyperparameter_tuning.ipynb` — CrossValidator tuning
8. `07_evaluation.ipynb` — Full evaluation with bootstrap CI
9. `08_scalability_analysis.ipynb` — Scaling experiments
10. `09_tableau_export.ipynb` — Export data for Tableau dashboards

## Setup and Run
### 1) Create environment
```bash
conda env create -f environment.yml
conda activate reddit-virality
```

### 2) Place dataset
Put `reddit_posts.csv` at:
`data/raw/reddit_posts.csv`

### 3) Run complete notebook pipeline
Open notebooks in the listed execution order and run all cells.

### 4) Fast execution path
If you need a faster pass for validation:
- Run `run_metrics_fast.py`
- Use sampled data notebooks (`02_data_engineering.ipynb` onward) before full-scale runs

## Running Tests
```bash
python -m pytest tests/ -v
```

## Requirements
- Python 3.12+
- PySpark 3.5+
- scikit-learn, pandas, numpy, matplotlib, seaborn, pyarrow

## Tableau Dashboards
1. **Data Quality Overview** — Dataset stats, missing values, class distribution
2. **Model Performance Comparison** — Metrics across all models and frameworks
3. **Business Insights** — What makes posts go viral
4. **Scalability & Cost Analysis** — Strong/weak scaling, cost projections
