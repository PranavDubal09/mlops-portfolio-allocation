from fastapi import FastAPI, HTTPException
from src.api.model_loader import load_model
from src.api.predict import run_inference

app = FastAPI(title="Portfolio Allocation API")

model = None


# 🔹 Load model at startup
@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        model = None


# 🔹 Root endpoint
@app.get("/")
def home():
    return {"message": "Portfolio API running"}


# 🔹 Health check (important for Docker/K8s)
@app.get("/health")
def health():
    if model is None:
        return {"status": "Model not loaded"}
    return {"status": "OK"}


# 🔹 Prediction endpoint
@app.get("/predict")
def predict():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        result = run_inference(model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 🔹 Reload model manually
@app.post("/reload")
def reload_model():
    global model
    try:
        model = load_model()
        return {"status": "Model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))