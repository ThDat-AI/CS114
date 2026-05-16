import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "Weights" / "Machine_Learning"

# Frontend Available Models
AVAILABLE_MODELS = [
    {"id": "logistic_regression", "name": "Logistic Regression"},
    {"id": "svm", "name": "Support Vector Machine"},
    {"id": "bert", "name": "BERT Transformer"},
    {"id": "bilstm", "name": "BiLSTM"},
    {"id": "textcnn", "name": "TextCNN"},
]

# Path to the current active Logistic Regression Model and Vectorizer
LOGISTIC_REGRESSION_MODEL_PATH = WEIGHTS_DIR / "logistic_model.pkl"
LOGISTIC_REGRESSION_VECTORIZER_PATH = WEIGHTS_DIR / "tfidf_vectorizer_lr.pkl"

# Path to the current active SVM Model and Vectorizer
SVM_MODEL_PATH = WEIGHTS_DIR / "svm_model.pkl"
SVM_VECTORIZER_PATH = WEIGHTS_DIR / "tfidf_vectorizer_svm.pkl"

# Server settings
HOST = "127.0.0.1"
PORT = 8080
CORS_ORIGINS = ["*"]
