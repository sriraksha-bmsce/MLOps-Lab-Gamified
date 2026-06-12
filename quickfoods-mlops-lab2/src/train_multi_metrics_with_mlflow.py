import os
import json
import time
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y = df["delivery_time_min"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate_regression(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "r2": float(r2)}


def measure_inference_latency_ms(model, X_sample: pd.DataFrame, repeats: int = 200) -> float:
    # Warm up
    _ = model.predict(X_sample)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = model.predict(X_sample)
    end = time.perf_counter()

    avg_sec = (end - start) / repeats
    return float(avg_sec * 1000.0)


def train_and_log(model_name: str, model, params: dict, X_train, X_test, y_train, y_test) -> dict:
    with mlflow.start_run(run_name=model_name):
        # Tags help in filtering runs later
        mlflow.set_tag("project", "QuickFoods")
        mlflow.set_tag("problem_type", "regression")
        mlflow.set_tag("dataset", "delivery_times_v1")

        # Log parameters
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train
        model.fit(X_train, y_train)

        # Predict & metrics
        preds = model.predict(X_test)
        metrics = evaluate_regression(y_test, preds)

        # Log multiple metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Save model artifact locally
        ensure_dir(MODEL_DIR)
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        # Compute model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_kb = model_size_bytes / 1024.0
        mlflow.log_metric("model_size_kb", float(model_size_kb))

        # Measure basic inference latency on 1 sample (avg over repeats)
        X_one = X_test.iloc[[0]] if len(X_test) > 0 else X_train.iloc[[0]]
        latency_ms = measure_inference_latency_ms(model, X_one, repeats=200)
        mlflow.log_metric("avg_inference_latency_ms", latency_ms)

        # Log artifacts: model file + JSON report
        mlflow.log_artifact(model_path)

        report = {
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "model_size_kb": model_size_kb,
            "avg_inference_latency_ms": latency_ms,
        }
        report_path = os.path.join(MODEL_DIR, f"{model_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # Log model in MLflow format (useful later for registry/deploy)
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

        print(
            f"[OK] {model_name} | "
            f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f} | R2={metrics['r2']:.3f} | "
            f"Size={model_size_kb:.1f}KB | Latency={latency_ms:.3f}ms"
        )

        return {"model_name": model_name, **metrics, "model_size_kb": model_size_kb, "avg_inference_latency_ms": latency_ms}


def main():
    print("=== Exercise 03: MLflow Multi-Metric Tracking (QuickFoods) ===")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split(df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    # Model 1: Baseline
    results.append(
        train_and_log(
            model_name="LinearRegression",
            model=LinearRegression(),
            params={},
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
    )

    # Model 2: RandomForest (example parameters)
    results.append(
        train_and_log(
            model_name="RandomForest",
            model=RandomForestRegressor(n_estimators=150, random_state=42),
            params={"n_estimators": 150},
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
    )

    # Model 3: GradientBoosting
    results.append(
        train_and_log(
            model_name="GradientBoosting",
            model=GradientBoostingRegressor(random_state=42),
            params={},
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        )
    )

    # Choose "best" model by MAE (business-friendly metric)
    best = sorted(results, key=lambda x: x["mae"])[0]

    print("\n=== Best model by MAE (lower is better) ===")
    print(
        f"Best: {best['model_name']} | "
        f"MAE={best['mae']:.3f} | RMSE={best['rmse']:.3f} | R2={best['r2']:.3f} | "
        f"Size={best['model_size_kb']:.1f}KB | Latency={best['avg_inference_latency_ms']:.3f}ms"
    )

    print("\nNext:")
    print("1) Start MLflow UI: mlflow ui")
    print("2) Open: http://127.0.0.1:5000")
    print("3) Sort by MAE / RMSE and compare trade-offs.")


if __name__ == "__main__":
    main()
