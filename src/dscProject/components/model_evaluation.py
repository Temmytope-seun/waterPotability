import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
from urllib.parse import urlparse
import numpy as np
from src.dscProject.entity.config_entity import ModelEvaluationConfig
from src.dscProject.utils.common import save_json
from pathlib import Path

os.environ["MLFLOW_TRACKING_URL"] = "https://dagshub.com/Temmytope-seun/dscProject.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Temmytope-seun"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ffd4eb226da5bd1070fe9492741f8ef8728c0623"

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        # Load the test data
        test_data = pd.read_csv(self.config.test_data_path)
        # Load the trained model
        model = joblib.load(self.config.model_path)
        # Separate features and target variable
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_test = test_data[self.config.target_column]
        
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print('trackinggggg',tracking_url_type_store)
        print(self.config.mlflow_uri)
        print(mlflow.__version__)
        with mlflow.start_run():
            # Make predictions
            y_pred = model.predict(X_test)
            (rmse, mae, r2) = self.eval_metrics(y_test, y_pred)

            # Save metrics to a JSON file
            metrics = {
                "RMSE": rmse,
                "MAE": mae,
                "R2_Score": r2
            }
            save_json(path=Path(self.config.metrics_file_path), data=metrics)
                    
            # Log metrics to MLflow
            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2_Score", r2)
            
            # mlflow.sklearn.log_model(model, "model")
            # run_id = run.info.run_id

        # mlflow.register_model(
        #     f"runs:/{run_id}/model",
        #     "ElasticNetModel"
        # )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNetModel") # type: ignore
            else:
                mlflow.sklearn.log_model(model, "model") # type: ignore

        