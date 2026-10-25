import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "dataset.csv"

SOURCE_COLUMNS = [
    "price",
    "is_discrete",
    "storage_size_gb",
    "ram_size_gb",
    "warranty",
    "clean_model_series",
    "cpu_composite_clean",
    "clean_brand",
    "clean_display_inches",
    "clean_gpu_composite",
    "is_global_train",
]


def prepare_dataset(source_path: Path, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    source = pd.read_csv(source_path)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing_columns:
        raise ValueError(f"Source dataset is missing required columns: {missing_columns}")

    dataset = source[SOURCE_COLUMNS].copy()
    dataset["is_global_train"] = (
        dataset["is_global_train"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    for column in [
        "price",
        "is_discrete",
        "storage_size_gb",
        "ram_size_gb",
        "warranty",
        "clean_display_inches",
    ]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset = dataset.dropna(subset=["price", "is_global_train"]).copy()
    dataset = dataset[dataset["price"] > 0]
    dataset = dataset[dataset["is_discrete"].isin([0, 1])]
    dataset["is_discrete"] = dataset["is_discrete"].astype(int)
    dataset["warranty"] = dataset["warranty"].astype(int)
    dataset = dataset.reset_index(drop=True)
    if dataset.empty or dataset["is_global_train"].nunique() != 2:
        raise ValueError("Prepared data must have valid prices and both source split groups.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the laptop listing dataset while omitting seller descriptions and leakage columns."
    )
    parser.add_argument(
        "source_csv",
        type=Path,
        help="Path to Kaggle_Master_Engineered_State.csv from the dataset archive.",
    )
    args = parser.parse_args()

    dataset = prepare_dataset(args.source_csv)
    print(f"Prepared {len(dataset):,} rows with {len(dataset.columns)} columns.")
    print(
        "Split sizes:",
        dataset["is_global_train"].value_counts().sort_index().to_dict(),
    )
    print(f"Saved feature-only dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
