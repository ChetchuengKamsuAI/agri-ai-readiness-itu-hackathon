"""Crop recommendation model. RandomForest on Crop_recommendation.csv."""
from __future__ import annotations
import argparse, json, logging, pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import config

logger = logging.getLogger(__name__)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
VALID_RANGES = {"N": (0, 200), "P": (0, 150), "K": (0, 250), "temperature": (-5, 55), "humidity": (0, 100), "ph": (3.0, 10.0), "rainfall": (0, 500)}


def load_data(csv_path=None):
    csv_path = csv_path or config.UAE_USECASE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Crop CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df[FEATURES], df[TARGET]


def train_model(X, y, n_estimators=config.RF_N_ESTIMATORS, random_state=config.RF_RANDOM_STATE, test_size=config.TEST_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {"accuracy": round(acc, 4), "n_train": len(X_train), "n_test": len(X_test), "n_classes": len(clf.classes_),
        "classes": list(clf.classes_),
        "feature_importances": dict(zip(FEATURES, [round(float(v), 4) for v in clf.feature_importances_])),
        "macro_f1": round(report["macro avg"]["f1-score"], 4)}
    return clf, metrics


def save_model(clf, metrics):
    path = config.MODEL_DIR / "crop_rf.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": clf, "features": FEATURES, "metrics": metrics}, f)
    return path


def load_model():
    path = config.MODEL_DIR / "crop_rf.pkl"
    if not path.exists():
        raise FileNotFoundError("No trained model. Run: python crop_model.py")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["metrics"]


def predict(soil):
    clf, _ = load_model()
    row = pd.DataFrame([[soil[f] for f in FEATURES]], columns=FEATURES)
    proba = clf.predict_proba(row)[0]
    classes = clf.classes_
    order = np.argsort(proba)[::-1]
    top = [(classes[i], round(float(proba[i]), 4)) for i in order[:3]]
    anomaly = _detect_anomaly(soil)
    return {"recommended_crop": top[0][0], "confidence": top[0][1], "alternatives": top[1:],
            "anomaly_detected": bool(anomaly), "anomaly_details": anomaly, "input": soil}


def _detect_anomaly(soil):
    issues = []
    for feat, (lo, hi) in VALID_RANGES.items():
        val = soil.get(feat)
        if val is None:
            issues.append(f"{feat}: missing")
        elif not (lo <= val <= hi):
            issues.append(f"{feat}={val} outside valid range [{lo}, {hi}]")
    return issues


def parse_soil_args(args):
    soil = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            soil[k.strip()] = float(v.strip())
    defaults = {"N": 90, "P": 42, "K": 43, "temperature": 20.9, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
    for f in FEATURES:
        soil.setdefault(f, defaults[f])
    return soil


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", nargs="*")
    args = parser.parse_args()
    if args.predict is not None:
        print(json.dumps(predict(parse_soil_args(args.predict)), indent=2, ensure_ascii=False))
        return
    X, y = load_data()
    print(f"Loaded {len(X)} rows, {y.nunique()} crop classes")
    clf, metrics = train_model(X, y)
    save_model(clf, metrics)
    print(f"Accuracy: {metrics['accuracy']} | Macro F1: {metrics['macro_f1']}")
    print("Feature importances:")
    for feat, imp in sorted(metrics["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat:15s} {imp:.4f}")


if __name__ == "__main__":
    main()
