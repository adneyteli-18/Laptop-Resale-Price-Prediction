from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import FEATURES, INTEGRATED_GRAPHICS, normalize_graphics

MODEL_PATH = Path(__file__).resolve().parent / "laptop_price_model.joblib"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.csv"

BRAND_MODEL_PREFIXES = {
    "ACER": ("Acer ", "TravelMate"),
    "APPLE": ("Apple ",),
    "ASUS": ("Asus ",),
    "DELL": ("Dell ",),
    "HP": ("HP ",),
    "INFINIX": ("Infinix ",),
    "LENOVO": ("Lenovo ",),
    "SAMSUNG": ("Samsung ",),
}
GENERIC_MODEL_SERIES = {
    "Generic OEM Chassis Series",
    "Other / Unclassified Hardware",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH)
    training_rows = data[data["is_global_train"].astype(bool)]
    if training_rows.empty:
        raise ValueError("The dataset does not contain rows in its training split.")
    return training_rows


def _options(data: pd.DataFrame, column: str) -> list:
    return sorted(data[column].dropna().unique().tolist())


def _default_index(options: list, preferred_value) -> int:
    return options.index(preferred_value) if preferred_value in options else 0


def _format_hardware_option(value: str) -> str:
    label = value.replace("_", " ").title()
    for abbreviation, display in {
        "Amd": "AMD",
        "Dgpu": "dGPU",
        "Gb": "GB",
        "Gtx": "GTX",
        "I3": "i3",
        "I5": "i5",
        "I7": "i7",
        "I9": "i9",
        "Igpu": "iGPU",
        "Nvidia": "NVIDIA",
        "Rtx": "RTX",
        "Sku": "SKU",
        "Sysram": "Shared RAM",
        "Th": "TH",
    }.items():
        label = label.replace(abbreviation, display)
    return label


def _model_series_options(data: pd.DataFrame, brand: str) -> list:
    brand_data = data[data["clean_brand"] == brand]
    options = _options(brand_data, "clean_model_series")
    prefixes = BRAND_MODEL_PREFIXES.get(brand)
    if prefixes is None:
        return [option for option in options if option in GENERIC_MODEL_SERIES]
    return [
        option
        for option in options
        if option.startswith(prefixes) or option in GENERIC_MODEL_SERIES
    ]


def _dedicated_gpu_options(data: pd.DataFrame) -> list:
    discrete_rates = data.groupby("clean_gpu_composite")["is_discrete"].mean()
    return sorted(discrete_rates[discrete_rates > 0.5].index.tolist())


def build_input_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    brand_options = _options(data, "clean_brand")
    gpu_options = _dedicated_gpu_options(data)
    st.subheader("Laptop inputs")
    st.caption(
        f"Enter the specifications shown on the laptop listing or device. "
        f"Catalog: {len(brand_options)} brands, "
        f"{data['clean_model_series'].nunique()} model series, "
        f"{data['cpu_composite_clean'].nunique()} processors, and "
        f"{len(gpu_options)} dedicated graphics configurations."
    )

    left, right = st.columns(2, gap="large")
    with left:
        brand = st.selectbox(
            "Brand",
            brand_options,
            index=_default_index(brand_options, "DELL"),
        )
        brand_data = data[data["clean_brand"] == brand]
        model_options = _model_series_options(data, brand)
        model_series = st.selectbox(
            "Model series",
            model_options,
            index=_default_index(model_options, "Dell Latitude"),
        )
        model_data = brand_data[brand_data["clean_model_series"] == model_series]
        processor_options = _options(model_data, "cpu_composite_clean")
        processor = st.selectbox(
            "Processor",
            processor_options,
            index=_default_index(
                processor_options,
                "INTEL_CORE_I5_SKU11_TH",
            ),
            format_func=_format_hardware_option,
            help=f"{len(processor_options)} processor options found for this model series.",
        )
        ram_gb = st.number_input(
            "RAM (GB)",
            min_value=2,
            max_value=128,
            value=8,
            step=2,
            help="Enter memory capacity in gigabytes.",
        )
        storage_gb = st.number_input(
            "Storage (GB)",
            min_value=32,
            max_value=8000,
            value=256,
            step=128,
            help="Enter SSD capacity in gigabytes.",
        )
    with right:
        has_dedicated_gpu = st.checkbox("Dedicated graphics card", value=False)
        if has_dedicated_gpu:
            dedicated_gpu_rows = data[
                data["is_discrete"].eq(1)
                & data["clean_gpu_composite"].isin(gpu_options)
            ]
            default_gpu = dedicated_gpu_rows["clean_gpu_composite"].value_counts().idxmax()
            gpu = st.selectbox(
                "Graphics card",
                gpu_options,
                index=_default_index(gpu_options, default_gpu),
                format_func=_format_hardware_option,
                help=(
                    f"Choose from {len(gpu_options)} dedicated graphics "
                    "configurations found in the training data."
                ),
            )
        else:
            gpu = INTEGRATED_GRAPHICS
            st.caption("Integrated graphics selected; no dedicated graphics card.")
        screen_size = st.number_input(
            "Screen size (inches)",
            min_value=10.0,
            max_value=18.5,
            value=float(data["clean_display_inches"].median()),
            step=0.5,
            help="Enter diagonal display size in inches.",
        )
        warranty = st.checkbox("Active warranty", value=False)

    submitted = st.button("Estimate resale value", type="primary", use_container_width=True)

    input_frame = pd.DataFrame(
        [
            {
                "clean_brand": brand,
                "clean_model_series": model_series,
                "cpu_composite_clean": processor,
                "clean_gpu_composite": gpu,
                "is_discrete": int(has_dedicated_gpu),
                "ram_size_gb": ram_gb,
                "storage_size_gb": storage_gb,
                "clean_display_inches": screen_size,
                "warranty": int(warranty),
            }
        ],
        columns=FEATURES,
    )
    return normalize_graphics(input_frame), submitted


def main() -> None:
    st.set_page_config(page_title="Laptop Resale Price Predictor", page_icon="💻", layout="centered")
    st.title("Laptop Resale Price Predictor")
    st.caption("Estimate a used laptop's seller asking price in Indian Rupees.")

    try:
        model = load_model()
        reference_data = load_reference_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Required model or training data is unavailable. Train the model and check the project files. Details: {exc}")
        st.stop()

    input_frame, submitted = build_input_frame(reference_data)

    if submitted:
        try:
            prediction = model.predict(input_frame)[0]
            if not np.isfinite(prediction) or prediction < 0:
                raise ValueError("The model returned an invalid asking price.")
            st.metric("Estimated listing price", f"₹{prediction:,.0f}")
            st.caption("Estimate based on seller asking prices. Actual sale prices may differ.")
        except ValueError as exc:
            st.error(f"Could not estimate this laptop: {exc}")
        except Exception as exc:  # pragma: no cover - user-facing error path
            st.error(f"Prediction failed: {exc}")
    else:
        st.info("Enter the laptop specifications and select **Estimate resale value**.")

    st.caption("For educational use. This estimate is not a guaranteed offer or confirmed transaction price.")


if __name__ == "__main__":
    main()
