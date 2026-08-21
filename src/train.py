import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Set default tracking URI if not already set by environment variable
if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # --- BONUS 5: Kiem tra phan phối du lieu (Data Drift / Imbalance Alert) ---
    total_train_samples = len(y_train)
    class_counts = y_train.value_counts().to_dict()
    label_distribution = {}
    print("=== BONUS 5: DATA DISTRIBUTION CHECK ===")
    for cls in [0, 1, 2]:
        count = class_counts.get(cls, 0)
        ratio = float(count / total_train_samples) if total_train_samples > 0 else 0.0
        label_distribution[str(cls)] = round(ratio, 4)
        if ratio < 0.10:
            print(f"[WARNING] Class {cls} ratio {ratio*100:.2f}% < 10% (Data Imbalance Detected!)")
        else:
            print(f"Class {cls} ratio: {ratio*100:.2f}% ({count}/{total_train_samples})")

    with mlflow.start_run():

        # 3. Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # 4. Khoi tao va huan luyen RandomForestClassifier
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # --- BONUS 3: Confusion Matrix & Detailed Performance Report ---
        cm = confusion_matrix(y_eval, preds, labels=[0, 1, 2])
        report_str = classification_report(
            y_eval,
            preds,
            labels=[0, 1, 2],
            target_names=["thap (0)", "trung_binh (1)", "cao (2)"],
            digits=4,
        )

        print("\n=== BONUS 3: CONFUSION MATRIX ===")
        print(cm)
        print("\n=== CLASSIFICATION REPORT (PRECISION & RECALL PER CLASS) ===")
        print(report_str)

        # Ghi file report.txt
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write("=== CONFUSION MATRIX ===\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write("=== CLASSIFICATION REPORT (PRECISION / RECALL PER CLASS) ===\n")
            f.write(report_str + "\n")

        # 6. Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # 7. In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 8. Luu metrics & label_distribution ra file outputs/metrics.json
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": label_distribution,
                },
                f,
                indent=2,
            )

        # 9. Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # 10. Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
