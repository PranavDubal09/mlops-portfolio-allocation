import mlflow.keras
import os

def load_model():
    if os.path.exists("/app/mlruns"):
        # Docker
        model_path = "/app/mlruns/569968458974769878/d8833beea85b4714bdc97c16429e5a25/artifacts/final_model"
    else:
        # Local
        model_path = "mlruns/569968458974769878/d8833beea85b4714bdc97c16429e5a25/artifacts/final_model"

    print(f"Loading model from: {model_path}")

    return mlflow.keras.load_model(model_path)