# Laptop Resale Price Prediction Using Machine Learning and Streamlit

## Practical Aim
This project demonstrates a machine learning deployment workflow for predicting the resale value of used laptops. The trained model is packaged into a local Streamlit web app so users can enter laptop specifications and receive a resale-price estimate in Indian Rupees.

## Problem Statement
Used laptop resale prices vary based on brand, processor, memory, storage, display size, condition, battery health, and warranty status. Estimating a fair resale value manually can be time-consuming and inconsistent. This project uses a supervised machine learning model to create a simple and repeatable resale-price prediction system.

## Objectives
- Build a clean dataset of laptop specifications and resale prices.
- Train a regression model to predict resale value.
- Save the full preprocessing and model pipeline for reuse.
- Deploy the trained pipeline in a Streamlit interface.
- Evaluate the model using standard regression metrics.

## Technology Stack
- Python
- Streamlit
- Scikit-learn
- pandas
- NumPy
- joblib

## Dataset Source and License Limitations
This project initially uses a synthetic laptop resale dataset created for educational and demonstration purposes because a suitable public used-laptop resale dataset was not available in the local environment. The dataset is not sourced from a licensed public resale marketplace and should not be treated as real-world market evidence. The model is therefore presented as a demonstration pipeline rather than a production-grade valuation model.

## Input Features and Target Variable
The model uses the following features:
- brand
- processor
- ram_gb
- storage_gb
- screen_size
- gpu
- condition
- battery_health
- warranty_months

Target variable:
- resale_price_inr

## Model Training Procedure
1. Load the dataset from `dataset.csv`.
2. Validate required columns and target values.
3. Separate numeric and categorical features.
4. Use a `ColumnTransformer` to apply missing-value handling and one-hot encoding.
5. Train a `RandomForestRegressor` with a fixed random state.
6. Split the dataset into train and test sets using `train_test_split`.
7. Evaluate performance on the held-out set.
8. Save the full fitted pipeline as `laptop_price_model.joblib`.

## Evaluation Metrics
Current evaluation metrics from the trained demonstration model:
- Dataset size: 900 rows × 10 columns
- MAE: ₹11,722.70
- RMSE: ₹15,616.97
- R²: 0.7943

These values were produced by running `python train_model.py` on the synthetic dataset included in this project.

## Project Structure
- `app.py` — Streamlit web interface for price prediction.
- `train_model.py` — dataset validation, model training, evaluation, and model export.
- `dataset.csv` — laptop feature dataset used for training.
- `laptop_price_model.joblib` — saved preprocessing + model pipeline.
- `requirements.txt` — Python dependencies for the project.
- `README.md` — project documentation.

## Running the Project
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Disclaimer
This project is a learning-oriented demonstration. The prediction is an estimate and may not reflect live resale market conditions.
