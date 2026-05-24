# News Classifier Demo

This project provides a web demonstration for a News Classifier using Machine Learning models.

## Project Structure

- `backend/`: FastAPI backend to handle model predictions and newspaper parsing.
- `frontend/`: Vanilla HTML/JS/CSS frontend.
- `Weights/Machine_Learning/`: Directory where the model weights (e.g., `logistic_regression.joblib`) should be placed.

## Features

1. **Parse by URL**: Paste a news article URL. The backend uses `newspaper3k` to automatically extract the title and body, then predicts the class.
2. **Input Text**: Manually paste the title and body of an article to predict its class.
3. **Model Selection**: Currently, only Logistic Regression is enabled. Other models are placeholders for future expansion.

## Setup Instructions

### 1. Place Model Weights
Make sure you have your trained Logistic Regression model and its TF-IDF vectorizer saved as `logistic_model.pkl` and `tfidf_vectorizer_lr.pkl` inside the `Weights/Machine_Learning/` directory.

```bash
mkdir -p Weights/Machine_Learning
# Move your logistic_model.pkl and tfidf_vectorizer_lr.pkl files here
```

### 2. Start the Backend
The backend uses `uv` to manage dependencies. From the `Demo` directory:

```bash
cd backend
# Install dependencies
uv add fastapi uvicorn newspaper3k scikit-learn pydantic joblib lxml_html_clean
# Run the FastAPI server
uv run uvicorn main:app --host 127.0.0.1 --port 8080
```
> **Note**: The first run might take a moment to download the `nltk` tokenizer model (`punkt`).

### 3. Start the Frontend
Since it's a Vanilla HTML frontend, you can use the VS Code **Live Server** extension, or run a simple Python HTTP server:

```bash
cd frontend
python -m http.server 5500
```
Then navigate to `http://localhost:5500` in your web browser.

## Configuration
You can easily manage the settings in `backend/config.py`, such as the available models, model paths, and server configurations.
