"""
Crop prediction (inference) module + CLI.

CLI example:
    python predict.py --N 90 --P 42 --K 43 --temperature 21 --humidity 82 --ph 6.5 --rainfall 203

Library example:
    from predict import recommend
    recommend({"N": 90, "P": 42, "K": 43, "temperature": 21,
               "humidity": 82, "ph": 6.5, "rainfall": 203})
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
# Also support the original project layout, where the artifact lives at the root.
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "crop_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Input validation ranges (reject garbage before it reaches the model)
VALID_RANGES = {
    "N": (0, 300), "P": (0, 300), "K": (0, 300),
    "temperature": (0, 55), "humidity": (0, 100),
    "ph": (0, 14), "rainfall": (0, 500),
}


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def validate(inputs: dict) -> dict:
    missing = [f for f in FEATURES if f not in inputs]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    clean = {}
    for f in FEATURES:
        v = float(inputs[f])
        lo, hi = VALID_RANGES[f]
        if not lo <= v <= hi:
            raise ValueError(f"{f}={v} out of valid range [{lo}, {hi}]")
        clean[f] = v
    return clean


def recommend(inputs: dict, top_k: int = 3) -> dict:
    """Return the best crop plus top-k alternatives with confidence."""
    clean = validate(inputs)
    model = _load_model()
    X = pd.DataFrame([clean], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    ranked = sorted(zip(model.classes_, proba), key=lambda t: t[1], reverse=True)[:top_k]
    return {
        "recommended_crop": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 4),
        "top_predictions": [
            {"crop": c, "probability": round(float(p), 4)} for c, p in ranked
        ],
        "alternatives": [
            {"crop": c, "probability": round(float(p), 4)} for c, p in ranked[1:]
        ],
        "input": clean,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop recommendation")
    for f in FEATURES:
        parser.add_argument(f"--{f}", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(recommend(vars(args)), indent=2))
