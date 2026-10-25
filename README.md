# Laptop Resale Price Prediction Using Machine Learning and Streamlit

## Practical Aim
Train and deploy a machine-learning regression pipeline that estimates a used laptop's seller asking price from Indian second-hand marketplace listing data.

## Problem Statement
Laptop prices depend on model series, processor and graphics configuration, memory, storage, display size, and warranty. The original demonstration model used synthetic formula-generated targets, so it could not support real-world valuations. This version uses a documented C2C listing dataset and evaluates against its provided holdout split.

## Important Limitation
The dataset target is a seller's **asking price**, not a confirmed completed sale price. Predictions estimate listing prices and are not guaranteed sale values or professional valuations. Dataset coverage and feature quality also constrain accuracy.

## Dataset Source and License
The source is [Indian Laptop Valuation & Arbitrage Dataset](https://www.kaggle.com/datasets/pagarani/indian-laptop-valuation-and-arbitrage-dataset), created by Pagarani. Kaggle identifies its license as [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). It describes 7,801 Indian peer-to-peer laptop listings and identifies `price` as the seller's INR asking price.

This project distributes a reduced, feature-only version of that dataset in `dataset.csv`. Seller descriptions and target-derived columns have been omitted. The included dataset remains subject to CC BY-NC-SA 4.0: attribution is required, adaptations must use the same license, and commercial use is not permitted. Review the source terms and underlying data rights before reuse or redistribution.

## Model Features and Target
The model predicts `price` (INR asking price) from:
- Brand, normalized laptop model series, CPU configuration, and GPU configuration.
- RAM, SSD capacity, display diagonal, and warranty status.

The app uses dropdown and numeric input controls for model features; location and sliders are not required. Listing description, location, condition score, price-normalized columns, and target-encoded columns are excluded so the inputs stay user-enterable and target leakage is prevented.

Dashboard selections are searchable and populated from the training data: 28 brands, all 31 model series (filtered to matching brand and generic listings), model-specific processor choices drawn from the 535 observed processor categories, and all 110 graphics configurations.

## Training and Evaluation
- Preprocessing is contained in the saved scikit-learn pipeline: numeric median imputation and categorical imputation/one-hot encoding.
- The estimator is an `ExtraTreesRegressor` with log-transformed target prices.
- The source's `is_global_train` split is preserved. The model trains only on the source training rows; the source holdout rows are used for evaluation.
- Reported metrics are MAE, RMSE, R², and a training-median MAE baseline, evaluated on that holdout.
- Metrics describe performance on asking-price listings from this dataset, not accuracy against actual completed sales or future markets.

The training script prints the latest metrics on the source-provided holdout. The RMSE may be much higher than MAE because some asking prices are difficult to estimate. These figures are not a guarantee for an individual laptop.

Latest run using the eight user-enterable features (1,939 holdout listings):
- Median-baseline MAE: ₹16,307.68
- Model MAE: ₹7,460.25
- Model RMSE: ₹15,692.28
- Model R²: 0.6692

Run training:
```bash
python train_model.py
```

To reproduce `dataset.csv`, download the source dataset archive from Kaggle, extract `Kaggle_Master_Engineered_State.csv`, then run:
```bash
python prepare_dataset.py path\to\Kaggle_Master_Engineered_State.csv
```
The preparation script retains only needed feature/target/split columns, drops rows without a usable target or split label, and leaves missing feature values for pipeline imputation.

## Technology Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Project Structure
- `app.py` — Streamlit listing-price estimator.
- `prepare_dataset.py` — creates the privacy-conscious feature-only dataset from the Kaggle source CSV.
- `train_model.py` — validates data, trains the preprocessing/model pipeline, evaluates on the fixed holdout, and saves the pipeline.
- `dataset.csv` — prepared listing data, excluding free-text seller descriptions, location, condition score, and target-derived features.
- `dataset_synthetic_demo.csv` — the previous synthetic dataset, retained as an archived example only; it is not used for current training or prediction.
- `laptop_price_model.joblib` — serialized fitted pipeline.
- `test_train_model.py` — focused tests for dataset validation and model inference.
- `requirements.txt` — Python dependencies.

## Run the Application
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Disclaimer
Predictions are estimates of seller asking prices from a historical dataset. Actual transaction prices may differ substantially. This project is for educational use and non-commercial reuse of the dataset.
