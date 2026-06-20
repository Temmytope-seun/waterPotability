import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
from urllib.parse import urlparse
from dotenv import load_dotenv
from src.dscProject.entity.config_entity import ModelEvaluationConfig
from src.dscProject.utils.common import save_json
from pathlib import Path

load_dotenv()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred, pred_proba):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred_proba)
        return accuracy, f1, roc_auc

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy, f1, roc_auc = self.eval_metrics(y_test, y_pred, y_pred_proba)

            save_json(path=Path(self.config.metrics_file_path), data={
                "Accuracy": accuracy,
                "F1_Score": f1,
                "ROC_AUC": roc_auc,
            })

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("Accuracy", float(accuracy))
            mlflow.log_metric("F1_Score", float(f1))
            mlflow.log_metric("ROC_AUC", float(roc_auc))

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", # type: ignore
                                         registered_model_name="RandomForestWaterPotability")
            else:
                mlflow.sklearn.log_model(model, "model") # type: ignore
