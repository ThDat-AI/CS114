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

    MOCK_CLASSES = [
        'business_finance', 
        'culture_lifestyle', 
        'film', 
        'food', 
        'music', 
        'sport', 
        'tech_science_games', 
        'world_environment'
    ]

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
            
            # 2. Xử lý lấy độ tự tin (Confidence) hoặc khoảng cách (Distance)
            confidence = None
            distance_val = None
            all_scores = {}
            
            if model_id == "svm":
                if hasattr(model, "decision_function"):
                    distances = model.decision_function(transformed_text)[0]
                    distance_val = float(np.max(distances))
                    all_scores = {str(model.classes_[i]): float(distances[i]) for i in range(len(distances))}
                else:
                    distance_val = 1.0  # Fallback
                    all_scores = {}
                
                return {
                    "prediction": prediction,
                    "distance": distance_val,
                    "scores": all_scores
                }
            else:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(transformed_text)[0]
                    confidence = float(np.max(probabilities))
                    all_scores = {str(model.classes_[i]): float(probabilities[i]) for i in range(len(probabilities))}
                elif hasattr(model, "decision_function"):
                    distances = model.decision_function(transformed_text)[0]
                    confidence = float(np.max(distances))
                    all_scores = {str(model.classes_[i]): float(distances[i]) for i in range(len(distances))}
                else:
                    confidence = 1.0
                    all_scores = {}
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "scores": all_scores
                }
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    else:
        pred_idx = len(text) % len(MOCK_CLASSES)
        prediction = MOCK_CLASSES[pred_idx]
        
        if model_id == "svm":
            distances = [0.1] * len(MOCK_CLASSES)
            distances[pred_idx] = 1.414
            all_scores = {MOCK_CLASSES[i]: distances[i] for i in range(len(MOCK_CLASSES))}
            return {
                "prediction": prediction,
                "warning": f"This is a mock prediction because the {model_id} model or vectorizer files were not found.",
                "distance": 1.414,
                "scores": all_scores
            }
        else:
            probs = [0.05] * len(MOCK_CLASSES)
            probs[pred_idx] = 0.65
            all_scores = {MOCK_CLASSES[i]: probs[i] for i in range(len(MOCK_CLASSES))}
            return {
                "prediction": prediction,
                "warning": f"This is a mock prediction because the {model_id} model or vectorizer files were not found.",
                "confidence": 0.65,
                "scores": all_scores
            }