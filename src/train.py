import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


EVAL_THRESHOLD = 0.70

# Lấy thư mục gốc project, bất kể bạn chạy train.py từ đâu
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ép MLflow luôn dùng file này
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# Ép artifact luôn nằm trong thư mục này
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"

# Tạo experiment riêng để artifact_location chắc chắn là mlartifacts
MLFLOW_EXPERIMENT_NAME = "random-forest-wine-quality"


def resolve_project_path(path: str | Path) -> Path:
    """
    Nếu path là relative path, chuyển nó thành path tính từ project root.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def setup_mlflow() -> None:
    """
    Cấu hình MLflow cố định:
    - Tracking store: project-root/mlflow.db
    - Artifact store: project-root/mlartifacts/
    """

    MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    artifact_uri = MLFLOW_ARTIFACT_ROOT.resolve().as_uri()

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=artifact_uri,
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình RandomForestClassifier và ghi nhận kết quả vào MLflow.

    Tham số:
        params    : dict chứa các siêu tham số cho RandomForestClassifier.
        data_path : đường dẫn đến file dữ liệu huấn luyện.
        eval_path : đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """

    setup_mlflow()

    data_path = resolve_project_path(data_path)
    eval_path = resolve_project_path(eval_path)

    # 1. Đọc dữ liệu train và eval
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Kiểm tra cột target
    if "target" not in df_train.columns:
        raise ValueError(f"File train không có cột target: {data_path}")

    if "target" not in df_eval.columns:
        raise ValueError(f"File eval không có cột target: {eval_path}")

    # 3. Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # 4. Log siêu tham số
        mlflow.log_params(params)

        # 5. Khởi tạo model
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            min_samples_split=params.get("min_samples_split", 2),
            random_state=42,
        )

        # 6. Huấn luyện
        model.fit(X_train, y_train)

        # 7. Dự đoán
        preds = model.predict(X_eval)

        # 8. Tính metric
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # 9. Log metric vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 10. Log model vào MLflow artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if acc < EVAL_THRESHOLD:
            print(f"Warning: accuracy {acc:.4f} is below threshold {EVAL_THRESHOLD:.2f}")
        else:
            print(f"Model passed threshold {EVAL_THRESHOLD:.2f}")

        # 11. Lưu metrics.json vào project-root/outputs
        outputs_dir = PROJECT_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            "accuracy": acc,
            "f1_score": f1,
        }

        with open(outputs_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # 12. Lưu model.pkl vào project-root/models
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, models_dir / "model.pkl")

    return acc


if __name__ == "__main__":
    params_path = PROJECT_ROOT / "params.yaml"

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train(params)