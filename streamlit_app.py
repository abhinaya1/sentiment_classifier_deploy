import streamlit as st
import requests

st.title("Customer Review Sentiment Classifier")

review = st.text_area("Enter Review")

if st.button("Predict"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": review}
    )

    result = response.json()

    st.write("Prediction:", result["prediction"])
    st.write("Confidence:", result["confidence"])