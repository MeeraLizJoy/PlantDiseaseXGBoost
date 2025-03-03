# src/streamlit_app.py
import streamlit as st
import numpy as np
import cv2
from src import data_loading, feature_extraction, explainability, utils, model_evaluation
import tensorflow as tf

# Load models and data
data_dir = "data/PlantifyDr"
img_width, img_height = 224, 224
lgb_model = utils.load_pickle("data/lgb_model.pkl")
label_encoder = utils.load_pickle("data/label_encoder.pkl")
feature_extractor = feature_extraction.create_feature_extractor(img_width, img_height)

def predict_and_explain(image):
    image = cv2.resize(image, (img_width, img_height))
    image = np.expand_dims(image, axis=0)
    features = feature_extraction.extract_features(image, feature_extractor)
    prediction = lgb_model.predict(features)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    return predicted_label

# Streamlit UI
st.title("Plant Disease Prediction")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    if st.button("Predict"):
        predicted_label = predict_and_explain(image)
        st.write(f"Predicted Disease: {predicted_label}")