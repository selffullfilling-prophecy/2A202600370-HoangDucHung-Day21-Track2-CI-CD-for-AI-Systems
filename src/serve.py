from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import pandas as pd
from pathlib import Path

app = FastAPI(title="Wine Quality Inference API")

GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")
LOCAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    if not GCS_BUCKET:
        if LOCAL_MODEL_PATH.exists():
            return str(LOCAL_MODEL_PATH)
        raise RuntimeError(
            "GCS_BUCKET chua duoc dat va khong tim thay model cuc bo tai "
            f"{LOCAL_MODEL_PATH}"
        )

    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client()

    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(GCS_MODEL_KEY)

    blob.download_to_filename(MODEL_PATH)

    print("Model da duoc tai xuong tu GCS.")
    return MODEL_PATH



model = joblib.load(download_model())
MODEL_FEATURE_NAMES = list(getattr(model, "feature_names_in_", FEATURE_NAMES))


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)",
        )

    features = pd.DataFrame([req.features], columns=MODEL_FEATURE_NAMES)
    prediction = int(model.predict(features)[0])
    labels = {
        0: "thap",
        1: "trung_binh",
        2: "cao",
    }

    return {
        "prediction": prediction,
        "label": labels.get(prediction, "unknown"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
