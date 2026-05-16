import joblib

model = joblib.load(r"D:\CS114_Project\Demo\Weights\Machine_Learning\logistic_model.pkl")
print("Model and Vectorizer loaded successfully.")
print(model.classes_)
