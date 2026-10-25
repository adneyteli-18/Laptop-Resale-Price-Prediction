from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "laptop_price_model.joblib"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.csv"

FEATURES = [
    "clean_brand",
    "clean_model_series",
    "cpu_composite_clean",
    "clean_gpu_composite",
    "state",
    "ram_size_gb",
    "storage_size_gb",
    "clean_display_inches",
    "warranty",
    "condition_degradation_index",
]


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


def build_input_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    brand_options = _options(data, "clean_brand")
    model_options = _options(data, "clean_model_series")
    processor_options = _options(data, "cpu_composite_clean")
    gpu_options = _options(data, "clean_gpu_composite")
    state_options = _options(data, "state")
    ram_options = sorted(data["ram_size_gb"].dropna().unique().tolist())
    storage_options = sorted(data["storage_size_gb"].dropna().unique().tolist())

    with st.form("laptop_specifications"):
        st.markdown("### Device specifications")
        st.caption("Match the listed specifications as closely as you can.")

        left, right = st.columns(2, gap="large")
        with left:
            brand = st.selectbox(
                "Brand",
                brand_options,
                index=_default_index(brand_options, "DELL"),
            )
            model_series = st.selectbox(
                "Model series",
                model_options,
                index=_default_index(model_options, "Dell Latitude"),
            )
            processor = st.selectbox(
                "Processor",
                processor_options,
                index=_default_index(processor_options, "INTEL_CORE_I5_SKU11_TH"),
                format_func=lambda value: value.replace("_", " ").title(),
            )
            ram_gb = st.select_slider(
                "RAM",
                options=ram_options,
                value=min(ram_options, key=lambda value: abs(value - 8)),
                format_func=lambda value: f"{value} GB",
            )
            storage_gb = st.select_slider(
                "Storage",
                options=storage_options,
                value=min(storage_options, key=lambda value: abs(value - 256)),
                format_func=lambda value: f"{value // 1024} TB" if value >= 1024 else f"{value} GB",
            )
        with right:
            gpu = st.selectbox(
                "Graphics",
                gpu_options,
                index=_default_index(gpu_options, "INTEL_INFERRED_IGPU_SYSRAM"),
                format_func=lambda value: value.replace("_", " ").title(),
            )
            state = st.selectbox(
                "State",
                state_options,
                index=_default_index(state_options, "Gujarat"),
            )
            screen_size = st.slider(
                "Screen size",
                min_value=10.0,
                max_value=18.5,
                value=float(data["clean_display_inches"].median()),
                step=0.1,
                format="%.1f in",
            )
            warranty = st.checkbox("Active warranty", value=False)
            degradation = st.slider(
                "Wear / degradation",
                min_value=0.0,
                max_value=0.4,
                value=0.0,
                step=0.01,
                help="Dataset score derived from listing condition details; 0 means no degradation was recorded.",
            )

        submitted = st.form_submit_button("Estimate resale value", type="primary", use_container_width=True)

    input_frame = pd.DataFrame(
        [
            {
                "clean_brand": brand,
                "clean_model_series": model_series,
                "cpu_composite_clean": processor,
                "clean_gpu_composite": gpu,
                "state": state,
                "ram_size_gb": ram_gb,
                "storage_size_gb": storage_gb,
                "clean_display_inches": screen_size,
                "warranty": int(warranty),
                "condition_degradation_index": degradation,
            }
        ],
        columns=FEATURES,
    )
    return input_frame, submitted


def main() -> None:
    st.set_page_config(page_title="Laptop Resale Price Predictor", page_icon="💻", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f7fb;
            color: #193247;
            --primary-color: #087eaa;
        }
        [data-testid="stHeader"] {
            background: rgba(244, 247, 251, 0.9);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 2rem 2.2rem;
            margin-bottom: 1.5rem;
            border-radius: 22px;
            color: white;
            background: linear-gradient(120deg, #102a43 0%, #145b83 58%, #168aad 100%);
            box-shadow: 0 16px 36px rgba(16, 42, 67, 0.16);
        }
        .hero-eyebrow {
            margin: 0 0 0.65rem;
            color: #a9e4f5;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .hero h1 {
            margin: 0;
            color: white;
            font-size: clamp(2rem, 4vw, 2.8rem);
            letter-spacing: -0.04em;
        }
        .hero-copy {
            max-width: 700px;
            margin: 0.65rem 0 0;
            color: #deeff7;
            font-size: 1rem;
            line-height: 1.6;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e2eaf2;
            border-radius: 18px;
            background: white;
            box-shadow: 0 8px 24px rgba(25, 50, 75, 0.05);
        }
        .result-label {
            color: #547084;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stCaptionContainer"] {
            color: #193247;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-color: #cedbe6;
            background-color: white;
            color: #193247;
        }
        .result-placeholder {
            margin-top: 2rem;
            color: #557084;
            font-size: 1rem;
            line-height: 1.6;
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            border: 0;
            border-radius: 10px;
            background: #087eaa;
            font-weight: 700;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            background: #06698f;
            border: 0;
        }
        div[data-testid="stSlider"] label,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectSlider"] label {
            color: #223b50;
            font-weight: 600;
        }
        </style>
        <section class="hero">
            <p class="hero-eyebrow">Laptop value estimator</p>
            <h1>Estimate a used laptop listing price.</h1>
            <p class="hero-copy">Compare your laptop’s specifications with thousands of Indian second-hand listings to estimate an asking price in Indian Rupees.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model()
        reference_data = load_reference_data()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Required model or training data is unavailable. Train the model and check the project files. Details: {exc}")
        st.stop()

    form_column, result_column = st.columns([1.55, 1], gap="large")
    with form_column, st.container(border=True):
        input_frame, submitted = build_input_frame(reference_data)

    with result_column, st.container(border=True):
        st.markdown('<p class="result-label">Your estimate</p>', unsafe_allow_html=True)
        if submitted:
            try:
                prediction = model.predict(input_frame)[0]
                if not np.isfinite(prediction) or prediction < 0:
                    raise ValueError("The model returned an invalid asking price.")
                st.metric("Estimated listing price", f"₹{prediction:,.0f}")
                st.caption("Estimated from comparable second-hand listing prices.")
            except ValueError as exc:
                st.error(f"Could not estimate this laptop: {exc}")
            except Exception as exc:  # pragma: no cover - user-facing error path
                st.error(f"Prediction failed: {exc}")
        else:
            st.markdown(
                '<p class="result-placeholder">Your estimated price will appear here after you enter the laptop details and select <b>Estimate resale value</b>.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.caption("The estimate reflects seller asking prices, not confirmed transaction prices.")
    st.caption(
        "This estimate is not a guaranteed offer. Actual sale prices vary with device age, seller negotiation, local demand, and details not included in the dataset."
    )


if __name__ == "__main__":
    main()
