import streamlit as st
from infer import infer
from io import BytesIO
import tempfile
import os

st.title("💧 Drone Plant Monitor")
uploaded = st.file_uploader("Upload a drone image...", type=["jpg","png"])
model_path = "best_model.h5"

if uploaded:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded.getvalue())
    temp_file.flush()
    cls_name, prob = infer(model_path, temp_file.name)
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)
    st.success(f"🌱 Plant is **{cls_name.upper()}**, confidence {prob.max()*100:.2f}%")
    os.unlink(temp_file.name)
