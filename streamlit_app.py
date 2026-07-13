import streamlit as st
import requests

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

API_URL = "https://sentiment-classifier-deploy.onrender.com"

st.set_page_config(
    page_title="Customer Review Sentiment Classifier",
    page_icon="💬",
    layout="centered"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:
    st.title("📊 Project Information")

    st.markdown("""
### Model
- Logistic Regression

### Text Representation
- TF-IDF Vectorizer

### Classes
- 😊 positive
- 😐 neutral
- ☹️ negative

### Backend
- FastAPI

### Deployment
- Render

### Frontend
- Streamlit
""")

    st.divider()

    st.subheader("API Status")

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)

        if response.status_code == 200:
            st.success("🟢 Online")
        else:
            st.warning("🟡 API Responded with Error")

    except Exception:
        st.error("🔴 Offline")

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("💬 Customer Review Sentiment Classifier")

st.write(
    """
Predict whether a customer review is **positive**, **neutral**, or **negative**
using a machine learning model deployed with FastAPI.
"""
)

# ---------------------------------------------------
# Example Reviews
# ---------------------------------------------------

st.subheader("Try an Example")

examples = [
    "",
    "Absolutely amazing product! I would buy it again.",
    "The product is okay. It works as expected.",
    "Worst purchase I've ever made."
]

selected_example = st.selectbox(
    "Choose an example review",
    examples
)

review = st.text_area(
    "Customer Review",
    value=selected_example,
    height=180,
    placeholder="Type a customer review..."
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🔍 Predict Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Predicting..."):

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json={"text": review},
                timeout=20
            )

            if response.status_code != 200:
                st.write("Status:", response.status_code)
                st.write("Response:", response.text)
                st.error("Prediction failed.")
                st.stop()

            result = response.json()

            prediction = result["prediction"]

            # If your API returns all probabilities
            probabilities = result.get("probabilities")

            st.divider()

            st.subheader("Prediction")

            if prediction.lower() == "positive":
                st.success("😊 positive")

            elif prediction.lower() == "neutral":
                st.warning("😐 neutral")

            else:
                st.error("☹️ negative")

            # --------------------------------------------
            # Show confidence
            # --------------------------------------------

            if probabilities:

                st.subheader("Confidence Scores")

                positive = probabilities["positive"]
                neutral = probabilities["neutral"]
                negative = probabilities["negative"]

                st.write(f"😊 positive: **{positive:.1%}**")
                st.progress(positive)

                st.write(f"😐 neutral: **{neutral:.1%}**")
                st.progress(neutral)

                st.write(f"☹️ negative: **{negative:.1%}**")
                st.progress(negative)

            else:

                confidence = result["confidence"]

                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%"
                )

                st.progress(confidence)

        except Exception as e:

            st.error("Unable to connect to the API.")
            st.exception(e)

# ---------------------------------------------------
# How it Works
# ---------------------------------------------------

st.divider()

with st.expander("⚙️ How It Works"):

    st.markdown("""
1. The user enters a customer review.

2. Streamlit sends the review to the deployed **FastAPI** REST API.

3. The API loads:

   - TF-IDF Vectorizer
   - Logistic Regression Model

4. TF-IDF converts the text into numerical features.

5. Logistic Regression predicts one of three classes:

   - positive
   - neutral
   - negative

6. The prediction and confidence scores are returned to Streamlit and displayed.
""")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit • FastAPI • Scikit-learn • Render"
)