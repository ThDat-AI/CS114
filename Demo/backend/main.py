import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from newspaper import Article
import nltk
import requests
from config import AVAILABLE_MODELS, CORS_ORIGINS, HOST, PORT
from Machine_Learning import predict_ml
from Bert import predict_news_category
from Deep_Learning import predict_bilstm, predict_textcnn
from Clean import clean_text
import uvicorn

# Download nltk punkt tokenizers for newspaper3k
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="News Classifier API", description="API to classify news articles")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str
    model_id: str

class TextRequest(BaseModel):
    title: str
    body: str
    model_id: str


def predict(title: str, body: str, model_id: str):
    """Route prediction to the appropriate model module."""
    model_id = str(model_id or "").lower()
    
    # BERT Model
    if model_id == "bert":
        try:
            result = predict_news_category(title, body)
            return {
                "prediction": result["category"],
                "confidence": result.get("confidence")
            }
        except Exception as e:
            return {"error": f"BERT prediction failed: {e}"}
    
    # BiLSTM Model
    if model_id == "bilstm":
        try:
            label, confidence = predict_bilstm(title, body)
            return {
                "prediction": label,
                "confidence": confidence
            }
        except Exception as e:
            return {"error": f"BiLSTM prediction failed: {e}"}
    
    # TextCNN Model
    if model_id == "textcnn":
        try:
            label, confidence = predict_textcnn(title, body)
            return {
                "prediction": label,
                "confidence": confidence
            }
        except Exception as e:
            return {"error": f"TextCNN prediction failed: {e}"}
    
    # Machine Learning Models (Logistic Regression, SVM)
    if model_id in ["logistic_regression", "svm"]:
        return predict_ml(title, body, model_id)
    
    return {"error": f"Model '{model_id}' is not supported. Available models: {AVAILABLE_MODELS}"}


@app.get("/models")
def get_models():
    return {"models": AVAILABLE_MODELS}

@app.post("/predict_url")
def predict_from_url(req: URLRequest):
    try:
        article = Article(req.url)
        article.download()
        article.parse()
    except Exception:
        try:
            response = requests.get(
                req.url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=10
            )
            response.raise_for_status()
            article = Article(req.url)
            article.download(input_html=response.text)
            article.parse()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse URL: {str(e)}")

    title = article.title or ""
    body = article.text or ""

    if not title and not body:
        raise HTTPException(status_code=400, detail="Failed to extract content from the provided URL.")

    # Clean the article body (only body, not title)
    cleaned_body = clean_text(body)

    result = predict(title, cleaned_body, req.model_id)

    return {
        "title": title,
        "body": cleaned_body,
        "result": result
    }

@app.post("/predict_text")
def predict_from_text(req: TextRequest):
    # Clean the article body (only body, not title)
    cleaned_body = clean_text(req.body)
    
    result = predict(req.title, cleaned_body, req.model_id)
    return {
        "title": req.title,
        "body": cleaned_body,
        "result": result
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
