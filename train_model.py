import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "laptop_price_model.joblib"
TARGET_COLUMN = "resale_price_inr"

REQUIRED_COLUMNS = [
    "brand",
    "processor",
    "ram_gb",
    "storage_gb",
    "screen_size",
    "gpu",
    "condition",
    "battery_health",
    "warranty_months",
    TARGET_COLUMN,
]

NUMERIC_FEATURES = ["ram_gb", "storage_gb", "screen_size", "battery_health", "warranty_months"]
CATEGORICAL_FEATURES = ["brand", "processor", "gpu", "condition"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.empty:
        raise ValueError("Dataset is empty after loading.")

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset has no valid target values after filtering.")

    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        df = load_dataset(DATASET_PATH)
        X = df[FEATURES]
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        joblib.dump(pipeline, MODEL_PATH)

        print("Dataset size:", df.shape)
        print("Selected features:", FEATURES)
        print(f"MAE: ₹{mae:,.2f}")
        print(f"RMSE: ₹{rmse:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"Model saved to: {MODEL_PATH}")

    except Exception as exc:  # pragma: no cover - explicit CLI error path
        logging.error("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
