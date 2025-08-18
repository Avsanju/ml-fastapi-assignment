# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model at startup
model_data = joblib.load("model.pkl")
model = model_data["model"]
class_names = model_data["class_names"]
features = model_data["features"]
accuracy = model_data["accuracy"]

app = FastAPI(title="Iris Classification API", description="Predict iris species using ML model")

# Input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features_arr = np.array([[input_data.sepal_length,
                                  input_data.sepal_width,
                                  input_data.petal_length,
                                  input_data.petal_width]])
        
        # Predict class
        pred_class = model.predict(features_arr)[0]
        pred_proba = model.predict_proba(features_arr).max()

        return PredictionOutput(
            prediction=class_names[pred_class],
            confidence=float(pred_proba)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "problem_type": "classification",
        "features": features,
        "classes": class_names,
        "accuracy": accuracy
    }
