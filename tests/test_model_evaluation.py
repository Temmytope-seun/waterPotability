import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import pandas as pd

from src.waterPotability.entity.config_entity import ModelEvaluationConfig
from src.waterPotability.components.model_evaluation import ModelEvaluation


def _make_config():
    return ModelEvaluationConfig(
        root_dir=Path("artifacts/model_evaluation"),
        test_data_path=Path("artifacts/data_transformation/test.csv"),
        model_path=Path("artifacts/model_trainer/model/model.joblib"),
        all_params={"n_estimators": 100, "max_depth": 6, "min_samples_split": 2},
        metrics_file_path=Path("artifacts/model_evaluation/metrics.json"),
        target_column="Potability",
        mlflow_uri="https://dagshub.com/test/test.mlflow",
    )


def _make_test_df(n=30):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "ph":               rng.uniform(0, 14, n),
        "Hardness":         rng.uniform(50, 300, n),
        "Solids":           rng.uniform(1000, 50000, n),
        "Chloramines":      rng.uniform(1, 12, n),
        "Sulfate":          rng.uniform(100, 500, n),
        "Conductivity":     rng.uniform(200, 800, n),
        "Organic_carbon":   rng.uniform(2, 25, n),
        "Trihalomethanes":  rng.uniform(10, 120, n),
        "Turbidity":        rng.uniform(1, 7, n),
        "Potability":       rng.integers(0, 2, n),
    })


class TestEvalMetrics(unittest.TestCase):

    def setUp(self):
        self.evaluator = ModelEvaluation(config=_make_config())

    def test_perfect_predictions_give_accuracy_one(self):
        actual = np.array([0, 1, 0, 1, 1, 0])
        proba  = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        accuracy, f1, roc_auc = self.evaluator.eval_metrics(actual, actual.copy(), proba)
        self.assertAlmostEqual(accuracy, 1.0)
        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(roc_auc, 1.0)

    def test_imperfect_predictions_give_accuracy_below_one(self):
        actual = np.array([0, 1, 0, 1, 1, 0])
        pred   = np.array([1, 0, 1, 0, 0, 1])
        proba  = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.7])
        accuracy, f1, roc_auc = self.evaluator.eval_metrics(actual, pred, proba)
        self.assertLess(accuracy, 1.0)
        self.assertGreaterEqual(accuracy, 0.0)

    def test_metrics_are_bounded_between_zero_and_one(self):
        rng    = np.random.default_rng(42)
        actual = rng.integers(0, 2, 50)
        pred   = rng.integers(0, 2, 50)
        proba  = rng.uniform(0, 1, 50)
        accuracy, f1, roc_auc = self.evaluator.eval_metrics(actual, pred, proba)
        for metric in (accuracy, f1, roc_auc):
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)


class TestLogIntoMlflow(unittest.TestCase):

    @mock.patch("src.waterPotability.components.model_evaluation.mlflow")
    @mock.patch("src.waterPotability.components.model_evaluation.save_json")
    @mock.patch("src.waterPotability.components.model_evaluation.joblib.load")
    @mock.patch("src.waterPotability.components.model_evaluation.pd.read_csv")
    def test_sets_tracking_uri(self, mock_read_csv, mock_load, _mock_save_json, mock_mlflow):
        df = _make_test_df()
        mock_read_csv.return_value = df
        mock_load.return_value.predict.return_value = df["Potability"].values
        mock_load.return_value.predict_proba.return_value = np.column_stack([
            1 - df["Potability"].values, df["Potability"].values
        ]).astype(float)
        mock_mlflow.get_tracking_uri.return_value = "https://dagshub.com/test/test.mlflow"

        config = _make_config()
        ModelEvaluation(config=config).log_into_mlflow()
        mock_mlflow.set_tracking_uri.assert_called_once_with(config.mlflow_uri)

    @mock.patch("src.waterPotability.components.model_evaluation.mlflow")
    @mock.patch("src.waterPotability.components.model_evaluation.save_json")
    @mock.patch("src.waterPotability.components.model_evaluation.joblib.load")
    @mock.patch("src.waterPotability.components.model_evaluation.pd.read_csv")
    def test_logs_all_three_metrics(self, mock_read_csv, mock_load, _mock_save_json, mock_mlflow):
        df = _make_test_df()
        mock_read_csv.return_value = df
        mock_load.return_value.predict.return_value = df["Potability"].values
        mock_load.return_value.predict_proba.return_value = np.column_stack([
            1 - df["Potability"].values, df["Potability"].values
        ]).astype(float)
        mock_mlflow.get_tracking_uri.return_value = "https://dagshub.com/test/test.mlflow"

        ModelEvaluation(config=_make_config()).log_into_mlflow()

        logged = {call.args[0] for call in mock_mlflow.log_metric.call_args_list}
        self.assertSetEqual(logged, {"Accuracy", "F1_Score", "ROC_AUC"})

    @mock.patch("src.waterPotability.components.model_evaluation.mlflow")
    @mock.patch("src.waterPotability.components.model_evaluation.save_json")
    @mock.patch("src.waterPotability.components.model_evaluation.joblib.load")
    @mock.patch("src.waterPotability.components.model_evaluation.pd.read_csv")
    def test_saves_metrics_json_with_correct_keys(self, mock_read_csv, mock_load,
                                                   mock_save_json, mock_mlflow):
        df = _make_test_df()
        mock_read_csv.return_value = df
        mock_load.return_value.predict.return_value = df["Potability"].values
        mock_load.return_value.predict_proba.return_value = np.column_stack([
            1 - df["Potability"].values, df["Potability"].values
        ]).astype(float)
        mock_mlflow.get_tracking_uri.return_value = "https://dagshub.com/test/test.mlflow"

        ModelEvaluation(config=_make_config()).log_into_mlflow()

        mock_save_json.assert_called_once()
        saved_data = mock_save_json.call_args.kwargs["data"]
        self.assertIn("Accuracy", saved_data)
        self.assertIn("F1_Score", saved_data)
        self.assertIn("ROC_AUC", saved_data)


if __name__ == "__main__":
    unittest.main()
