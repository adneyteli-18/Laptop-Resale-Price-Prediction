import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "laptop_price_model.joblib"
TARGET_COLUMN = "price"
SPLIT_COLUMN = "is_global_train"

CATEGORICAL_FEATURES = [
    "clean_brand",
    "clean_model_series",
    "cpu_composite_clean",
    "clean_gpu_composite",
]
NUMERIC_FEATURES = [
    "ram_size_gb",
    "storage_size_gb",
    "clean_display_inches",
    "warranty",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = [TARGET_COLUMN, SPLIT_COLUMN, *FEATURES]


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[SPLIT_COLUMN] = df[SPLIT_COLUMN].astype("boolean")

    df = df.dropna(subset=[TARGET_COLUMN, SPLIT_COLUMN])
    df = df[df[TARGET_COLUMN] > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset has no rows with a positive asking price and valid split.")

    if df[FEATURES].isna().all(axis=None):
        raise ValueError("Dataset has no usable feature values.")

    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=3),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    estimator = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
    )
    price_model = TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", price_model),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        df = load_dataset()
        training_data = df[df[SPLIT_COLUMN]].copy()
        holdout_data = df[~df[SPLIT_COLUMN]].copy()
        if training_data.empty or holdout_data.empty:
            raise ValueError(
                "Dataset must contain rows in both the provided training and holdout splits."
            )

        pipeline = build_pipeline()
        pipeline.fit(training_data[FEATURES], training_data[TARGET_COLUMN])

        predictions = pipeline.predict(holdout_data[FEATURES])
        mae = mean_absolute_error(holdout_data[TARGET_COLUMN], predictions)
        rmse = np.sqrt(
            mean_squared_error(holdout_data[TARGET_COLUMN], predictions)
        )
        r2 = r2_score(holdout_data[TARGET_COLUMN], predictions)
        baseline = np.full(
            len(holdout_data), training_data[TARGET_COLUMN].median()
        )
        baseline_mae = mean_absolute_error(
            holdout_data[TARGET_COLUMN], baseline
        )

        joblib.dump(pipeline, MODEL_PATH, compress=3)

        print(f"Dataset rows: {len(df):,}")
        print(f"Training rows: {len(training_data):,}")
        print(f"Source holdout rows (not fitted): {len(holdout_data):,}")
        print(f"Target: seller asking price (INR), column '{TARGET_COLUMN}'")
        print(f"Features: {FEATURES}")
        print(f"Holdout median-baseline MAE: ₹{baseline_mae:,.2f}")
        print(f"Holdout MAE: ₹{mae:,.2f}")
        print(f"Holdout RMSE: ₹{rmse:,.2f}")
        print(f"Holdout R²: {r2:.4f}")
        print(f"Model saved to: {MODEL_PATH}")
    except Exception as exc:
        logging.error("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
