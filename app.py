
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from datetime import datetime


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HemoScan - Hb Prediction",
    page_icon="🩸",
    layout="wide"
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ml" / "models_saved" / "best_model.keras"
RESULTS_PATH = BASE_DIR / "data" / "prediction_results.csv"
HISTORY_PATH = BASE_DIR / "data" / "prediction_history.csv"


# --------------------------------------------------
# INITIALIZE PERMANENT PREDICTION HISTORY
# --------------------------------------------------

if "prediction_history" not in st.session_state:

    if HISTORY_PATH.exists():

        try:
            history_df = pd.read_csv(HISTORY_PATH)

            st.session_state.prediction_history = (
                history_df.to_dict("records")
            )

        except Exception as e:
            st.error(f"Unable to load prediction history: {e}")
            st.session_state.prediction_history = []

    else:
        st.session_state.prediction_history = []


# --------------------------------------------------
# SAVE PREDICTION HISTORY
# --------------------------------------------------

def save_prediction_history():

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(uploaded_file):

    image_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to read the uploaded image."
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(np.float32)

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_hemoglobin(model, image):

    prediction = model.predict(
        image,
        verbose=0
    )

    hb_value = float(prediction[0][0])

    if not np.isfinite(hb_value):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    return hb_value


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🩸 HemoScan")

st.subheader(
    "Experimental Hemoglobin Estimation Using Fingernail Images"
)

st.write(
    """
    HemoScan is an experimental machine-learning application
    that estimates hemoglobin levels from fingernail images.
    """
)

st.warning(
    "Upload a cropped fingernail image. This application does not "
    "automatically detect or crop fingernails from full-hand images."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.header("📷 Upload Fingernail Image")

uploaded_file = st.file_uploader(
    "Choose a fingernail image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Fingernail Image",
            width=350
        )

        if st.button(
            "Predict Hemoglobin Level",
            type="primary"
        ):

            with st.spinner(
                "Analyzing fingernail image..."
            ):

                model = load_model()

                processed_image = preprocess_image(
                    uploaded_file
                )

                hb_value = predict_hemoglobin(
                    model,
                    processed_image
                )

            st.success(
                "Prediction completed successfully!"
            )

            st.metric(
                label="Experimental Hb Estimate",
                value=f"{hb_value:.2f} g/dL"
            )

            # Add prediction to history
            new_record = {
                "Date & Time": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),
                "Image": uploaded_file.name,
                "Predicted Hb (g/dL)": round(
                    hb_value,
                    2
                )
            }

            st.session_state.prediction_history.append(
                new_record
            )

            # Save history permanently
            save_prediction_history()

            st.info(
                "This is an experimental estimate, not a clinical "
                "hemoglobin measurement."
            )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )


# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()

st.header("📋 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Download history
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="hemoscan_prediction_history.csv",
        mime="text/csv"
    )

    # Clear history
    if st.button(
        "Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()

        st.success(
            "Prediction history cleared."
        )

        st.rerun()

else:

    st.info(
        "No predictions have been recorded yet."
    )


# --------------------------------------------------
# MODEL PERFORMANCE DASHBOARD
# --------------------------------------------------

st.divider()

st.header("📊 Model Performance Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "MAE",
    "1.718 g/dL"
)

col2.metric(
    "RMSE",
    "2.3624 g/dL"
)

col3.metric(
    "R² Score",
    "0.239"
)

col4.metric(
    "Test Patients",
    "38"
)


# --------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

st.divider()

st.subheader(
    "Actual vs Predicted Hemoglobin Levels"
)

if RESULTS_PATH.exists():

    try:

        results_df = pd.read_csv(
            RESULTS_PATH
        )

        # Correct CSV column names
        required_columns = [
            "HB_g_dL",
            "Predicted_Hb"
        ]

        if all(
            column in results_df.columns
            for column in required_columns
        ):

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.scatter(
                results_df["HB_g_dL"],
                results_df["Predicted_Hb"],
                alpha=0.7
            )

            minimum = min(
                results_df["HB_g_dL"].min(),
                results_df["Predicted_Hb"].min()
            )

            maximum = max(
                results_df["HB_g_dL"].max(),
                results_df["Predicted_Hb"].max()
            )

            ax.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--"
            )

            ax.set_xlabel(
                "Actual Hb (g/dL)"
            )

            ax.set_ylabel(
                "Predicted Hb (g/dL)"
            )

            ax.set_title(
                "Actual vs Predicted Hb Levels"
            )

            ax.grid(True)

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.warning(
                "Required columns HB_g_dL and Predicted_Hb "
                "were not found in prediction_results.csv."
            )

    except Exception as e:

        st.error(
            f"Unable to display prediction graph: {e}"
        )

else:

    st.warning(
        "Prediction results CSV file was not found."
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.header("ℹ️ Model Information")

st.write(
    """
    - Model: EfficientNetB0
    - Framework: TensorFlow / Keras
    - Input image size: 224 × 224
    - Output: Experimental hemoglobin estimate in g/dL
    - Dataset: Fingernail images with laboratory hemoglobin measurements
    """
)


# --------------------------------------------------
# MEDICAL DISCLAIMER
# --------------------------------------------------

st.divider()

st.header("⚠️ Medical Disclaimer")

st.write(
    """
    HemoScan is an academic experimental prototype developed
    for educational and research purposes.

    Its predictions are not clinically validated and must not
    be used to diagnose anemia, make treatment decisions,
    or replace laboratory hemoglobin testing.

    Consult a qualified healthcare professional for actual
    hemoglobin assessment.
    """
)

