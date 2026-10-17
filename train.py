"""
Train, evaluate and persist the crop recommendation model.

    python train.py

Outputs:
    models/crop_model.joblib   - sklearn Pipeline (scaler + classifier)
    models/metrics.json        - evaluation report (for tracking / CI gates)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "crop_data.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "Crop_recommendation.csv"
MODEL_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
MIN_ACCURACY = 0.90  # quality gate: refuse to save a weak model


def build_pipeline(clf) -> Pipeline:
    # Scaler + model are bundled so training and serving preprocessing can never drift
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, clf in candidates.items():
        cv_acc = cross_val_score(build_pipeline(clf), X_train, y_train, cv=cv, scoring="accuracy")
        scores[name] = cv_acc.mean()
        print(f"{name:20s} CV accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")

    best_name = max(scores, key=scores.get)
    print(f"\nBest model: {best_name}")

    model = build_pipeline(candidates[best_name]).fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"Test accuracy: {acc:.4f} | macro F1: {f1:.4f}\n")
    print(classification_report(y_test, preds))

    if acc < MIN_ACCURACY:
        raise SystemExit(f"Accuracy {acc:.3f} below gate {MIN_ACCURACY}; model not saved.")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "crop_model.joblib")

    importances = model.named_steps["clf"].feature_importances_
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": best_name,
        "cv_accuracy": {k: round(v, 4) for k, v in scores.items()},
        "test_accuracy": round(acc, 4),
        "test_macro_f1": round(f1, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": list(model.classes_),
        "feature_importance": {f: round(float(i), 4) for f, i in zip(FEATURES, importances)},
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/crop_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
