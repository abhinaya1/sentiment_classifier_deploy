# Customer Review Sentiment Classifier Deployment

An end-to-end Machine Learning application that classifies customer reviews into **Positive**, **Negative**, or **Neutral** sentiment using Natural Language Processing (NLP). The project includes a trained Logistic Regression model, a FastAPI backend for inference, and a Streamlit frontend for interactive predictions.

---

## Project Overview

This application analyzes customer review text and predicts one of three sentiment classes:

- 😊 Positive
- 😐 Neutral
- 😞 Negative

For each prediction, the application returns:

- Predicted sentiment
- Confidence score
- Probability for each sentiment class

The project demonstrates a complete machine learning deployment workflow—from model training and serialization to API development and a user-friendly web interface.

---

## Features

- Three-class sentiment classification
- NLP text preprocessing
- TF-IDF vectorization
- Logistic Regression classifier
- REST API built with FastAPI
- Interactive Streamlit web application
- Confidence score and class probabilities
- Ready for cloud deployment

---

## Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

### Deployment
- Render
- GitHub

---

## Model

### Algorithm
- Logistic Regression (Multiclass Classification)

### Feature Engineering
- Text cleaning
- Lowercasing
- TF-IDF Vectorization

### Output

Example response:

```json
{
  "prediction": "Neutral",
  "confidence": 0.82,
  "probabilities": {
    "Negative": 0.09,
    "Neutral": 0.82,
    "Positive": 0.09
  }
}
```

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

## Installation

### Clone the repository

```bash
git clone https://github.com/abhinaya1/sentiment_classifier_deploy.git

cd sentiment_classifier_deploy
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the FastAPI Backend

```bash
uvicorn api:app --reload
```

Open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Run the Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## API Endpoint

### POST `/predict`

### Request

```json
{
  "text": "The product works as expected."
}
```

### Response

```json
{
  "prediction": "Neutral",
  "confidence": 0.84,
  "probabilities": {
    "Negative": 0.07,
    "Neutral": 0.84,
    "Positive": 0.09
  }
}
```

---

## Sample Predictions

| Review | Prediction |
|---------|------------|
| This product exceeded my expectations! | Positive |
| The product is okay and works as expected. | Neutral |
| Very disappointed with the quality. | Negative |
| Excellent value for the price. | Positive |
| It's neither good nor bad. | Neutral |

---

## Skills Demonstrated

- Natural Language Processing (NLP)
- Text Classification
- TF-IDF Feature Engineering
- Logistic Regression
- Model Serialization
- REST API Development
- FastAPI
- Streamlit
- Machine Learning Deployment
- Cloud Deployment
- End-to-End ML Workflow

---

## Future Enhancements

- Docker containerization
- GitHub Actions for CI/CD
- Model monitoring and logging
- Batch prediction endpoint
- Explainable AI using SHAP or LIME
- Transformer-based models (BERT/RoBERTa)
- User authentication

---

## Author

**Abhinaya Gyawali**

- GitHub: https://github.com/abhinaya1
- Portfolio: https://abhinaya1.github.io/
- LinkedIn: https://www.linkedin.com/in/abhinaya-gyawali/
