from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "laptop_price_model.joblib"

FEATURES = [
    "brand",
    "processor",
    "ram_gb",
    "storage_gb",
    "screen_size",
    "gpu",
    "condition",
    "battery_health",
    "warranty_months",
]

BRANDS = ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI"]
PROCESSORS = [
    "Intel Core i3",
    "Intel Core i5",
    "Intel Core i7",
    "AMD Ryzen 5",
    "AMD Ryzen 7",
    "Apple M1",
    "Apple M2",
    "Apple M3",
]
GPUS = [
    "Integrated",
    "Intel Iris Xe",
    "NVIDIA MX450",
    "NVIDIA RTX 3050",
    "NVIDIA RTX 3060",
    "NVIDIA RTX 4050",
    "Apple GPU",
]
CONDITIONS = ["Excellent", "Good", "Fair"]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_input_frame() -> pd.DataFrame:
    brand = st.selectbox("Brand", BRANDS)
    processor = st.selectbox("Processor", PROCESSORS)
    ram_gb = st.slider("RAM (GB)", min_value=4, max_value=64, value=16, step=4)
    storage_gb = st.slider("Storage (GB)", min_value=128, max_value=2048, value=512, step=128)
    screen_size = st.slider("Screen size (inches)", min_value=11.0, max_value=17.5, value=14.0, step=0.1)
    gpu = st.selectbox("GPU", GPUS)
    condition = st.selectbox("Condition", CONDITIONS)
    battery_health = st.slider("Battery health (%)", min_value=30, max_value=100, value=85, step=1)
    warranty_months = st.slider("Warranty remaining (months)", min_value=0, max_value=36, value=6, step=1)

    return pd.DataFrame(
        [{
            "brand": brand,
            "processor": processor,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "screen_size": screen_size,
            "gpu": gpu,
            "condition": condition,
            "battery_health": battery_health,
            "warranty_months": warranty_months,
        }],
        columns=FEATURES,
    )


def main() -> None:
    st.set_page_config(page_title="Laptop Resale Price Predictor", layout="centered")
    st.title("Laptop Resale Price Prediction")
    st.caption("Estimate the resale value of a used laptop in Indian Rupees based on key specifications.")

    st.markdown(
        "This application predicts a resale price estimate. It is intended for demonstration use and should not replace a formal market valuation."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(f"Model not found. Please train the model first and ensure laptop_price_model.joblib exists. Details: {exc}")
        st.stop()

    input_frame = build_input_frame()

    if st.button("Predict Resale Price", type="primary"):
        try:
            prediction = model.predict(input_frame)[0]
            st.success(f"Estimated resale price: ₹{prediction:,.0f}")
            st.info("Disclaimer: This value is an estimate derived from the trained model and may vary from actual resale market prices.")
        except ValueError as exc:
            st.error(f"Invalid input provided: {exc}")
        except Exception as exc:  # pragma: no cover - user-facing error path
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
