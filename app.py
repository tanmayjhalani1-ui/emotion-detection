from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import load_metadata, load_model, predict

st.set_page_config(page_title="Facial Expression Classifier", page_icon="🙂", layout="wide")
st.title("Facial Expression Classifier")
st.caption("A FER-2013 research/demo classifier. It predicts dataset labels, not a person's true internal emotional state.")

@st.cache_resource
def get_model():
    return load_model()

metadata = load_metadata()
upload_col, report_col = st.columns([1, 1], gap="large")

with upload_col:
    st.subheader("Classify an image")
    upload = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
    st.caption("Use a clear, front-facing, single-face image for the closest match to the training data.")
    if upload:
        image = Image.open(upload).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)
        try:
            result = predict(image, get_model())
            st.metric("Predicted dataset label", result["predicted_label"].title(), f"{result['confidence']:.1%} model confidence")
            st.dataframe(pd.DataFrame(result["top_three"]), hide_index=True, use_container_width=True)
            st.bar_chart(pd.Series(result["probabilities"], name="Model score"))
        except FileNotFoundError as error:
            st.warning(str(error))

with report_col:
    st.subheader("Model report")
    if metadata:
        st.metric("Selected architecture", metadata.get("selected_model", "—"))
        st.dataframe(pd.DataFrame(metadata.get("candidates", [])), hide_index=True, use_container_width=True)
        st.caption(metadata.get("limitations", ""))
    else:
        st.info("Train the model before running inference.")
        st.code("python -m src.train", language="bash")
    st.markdown("**Dataset:** FER-2013 · 48×48 grayscale face images")
    st.markdown("**Classes:** angry, disgust, fear, happy, neutral, sad, surprise")
