import joblib
import numpy as np
from config import (
    LOGISTIC_REGRESSION_MODEL_PATH, LOGISTIC_REGRESSION_VECTORIZER_PATH,
    SVM_MODEL_PATH, SVM_VECTORIZER_PATH,
)

# Load the models and vectorizers
ml_models = {}

def load_model(model_path, vectorizer_path, model_id):
    """Load ML model and vectorizer from disk."""
    model = None
    vectorizer = None
    try:
        if model_path.exists() and vectorizer_path.exists():
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print(f"{model_id} Model and Vectorizer loaded successfully.")
        else:
            if not model_path.exists():
                print(f"Warning: Model not found at {model_path}.")
            if not vectorizer_path.exists():
                print(f"Warning: Vectorizer not found at {vectorizer_path}.")
            print(f"Using mock predictions for {model_id}.")
    except Exception as e:
        print(f"Error loading {model_id} model or vectorizer: {e}")
    
    return model, vectorizer

ml_models["logistic_regression"] = load_model(LOGISTIC_REGRESSION_MODEL_PATH, LOGISTIC_REGRESSION_VECTORIZER_PATH, "Logistic Regression")
ml_models["svm"] = load_model(SVM_MODEL_PATH, SVM_VECTORIZER_PATH, "SVM")


def predict_ml(title: str, body: str, model_id: str):
    """Predict using Machine Learning models (Logistic Regression or SVM) supporting 8 classes."""
    model_id = str(model_id or "").lower()
    text = (title + " . " + body).strip()

    if model_id not in ml_models:
        return {"error": f"ML Model '{model_id}' is not supported."}

    model, vectorizer = ml_models[model_id]
    
    if model is not None and vectorizer is not None:
        try:
            text_lower = text.lower()
            transformed_text = vectorizer.transform([text_lower])
            
            # 1. Dự đoán nhãn (Trả về nhãn có điểm cao nhất trong 8 lớp)
            prediction = model.predict(transformed_text)[0]
            if hasattr(prediction, "item"):
                prediction = prediction.item()
            
            # 2. Xử lý lấy độ tự tin (Confidence) an toàn cho mô hình 8 lớp
            confidence = None
            
            if hasattr(model, "predict_proba"):
                # Trả về mảng dạng [[p1, p2, p3, p4, p5, p6, p7, p8]]
                probabilities = model.predict_proba(transformed_text)
                if isinstance(probabilities, list):
                    probabilities = probabilities[0]
                
                # Lấy dòng đầu tiên [0] và tìm xác suất lớn nhất trong 8 lớp
                confidence = float(np.max(probabilities[0]))
                
            elif hasattr(model, "decision_function"):
                # Trả về mảng điểm số dạng [[s1, s2, s3, s4, s5, s6, s7, s8]]
                distance = model.decision_function(transformed_text)
                
                # Lấy dòng đầu tiên [0] và tìm điểm số cao nhất trong 8 lớp
                confidence = float(np.max(distance[0]))
            
            return {
                "prediction": prediction,
                "confidence": confidence
            }
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    else:
        return {
            "prediction": "Mock Class 1" if len(text) % 2 == 0 else "Mock Class 2",
            "warning": f"This is a mock prediction because the {model_id} model or vectorizer files were not found.",
            "confidence": None
        }