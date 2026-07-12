from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

tfidf = joblib.load("tfidf.pkl")
model = joblib.load("sentiment_model.pkl")

class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment Classifier API"}

@app.post("/predict")
def predict(review: Review):

    X = tfidf.transform([review.text])

    prediction = model.predict(X)[0]

    probs = model.predict_proba(X)[0]

    classes = model.classes_

    return {
        "prediction": prediction,
        "probabilities": {
            cls: float(prob)
            for cls, prob in zip(classes, probs)
        }
    }