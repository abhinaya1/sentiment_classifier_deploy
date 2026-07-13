# Customer Review Sentiment Classifier – Deployment

This repository contains the **deployment** of a Customer Review Sentiment Classifier using **FastAPI**, **Streamlit**, and **Render**. The application serves a trained machine learning model through a REST API and provides an interactive web interface for real-time sentiment prediction.

> **Sentiment Classes:** Positive • Neutral • Negative

---

## Overview

The application allows users to enter a customer review and receive:

* Predicted sentiment (Positive, Neutral, or Negative)
* Confidence score
* Probability for each sentiment class

This project focuses on taking a trained machine learning model from development to a production-style application that users can interact with through a web interface.

---

## Tech Stack

**Machine Learning**

* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression

**Backend**

* FastAPI
* Uvicorn

**Frontend**

* Streamlit

**Deployment**

* Render
* GitHub

---

## Project Structure

```
sentiment_classifier_deploy
├── __pycache__
│   └── app.cpython-312.pyc
├── app.py
├── Dockerfile
├── requirements.txt
├── runtime.txt
├── sentiment_model.pkl
├── streamlit_app.py
└── tfidf.pkl
```

---

## Running Locally

### Clone the repository

```bash
git clone https://github.com/abhinaya1/sentiment_classifier_deploy.git
cd sentiment_classifier_deploy
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the FastAPI server

```bash
uvicorn api:app --reload
```

API documentation:

```
http://127.0.0.1:8000/docs
```

### Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## API Example

### Request

```json
{
  "text": "This product exceeded my expectations!"
}
```

### Response

```json
{
  "prediction": "Positive",
  "confidence": 0.98,
  "probabilities": {
    "Negative": 0.01,
    "Neutral": 0.01,
    "Positive": 0.98
  }
}
```

---

## Related Project

This repository focuses on **deployment**.

The complete machine learning workflow—including data exploration, preprocessing, feature engineering, model training, evaluation, and model selection—is available in the original repository:

**Customer Review Sentiment Classifier**
https://github.com/abhinaya1/Customer-Review-Sentiment-Classifier

---

## Skills Demonstrated

* Machine Learning Deployment
* REST API Development
* FastAPI
* Streamlit
* Model Serialization
* Cloud Deployment
* End-to-End ML Applications

---

## Future Improvements

* Docker containerization
* CI/CD with GitHub Actions
* Model monitoring
* Automated testing
* Explainable AI (SHAP/LIME)
* Authentication and rate limiting

---

## Author

**Abhinaya Gyawali**

* GitHub: https://github.com/abhinaya1
* Portfolio: https://abhinaya1.github.io/
* LinkedIn: https://www.linkedin.com/in/abhinaya-gyawali/
