import unittest
from unittest import mock
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.dscProject.entity.config_entity import ModelTrainerConfig
from src.dscProject.components.model_trainer import ModelTrainer


def _make_config():
    return ModelTrainerConfig(
        root_dir=Path("artifacts/model_trainer"),
        train_data_path=Path("artifacts/data_transformation/train.csv"),
        test_data_path=Path("artifacts/data_transformation/test.csv"),
        model_name="model.joblib",
        n_estimators=10,
        max_depth=3,
        min_samples_split=2,
        target_column="Potability",
    )


def _sample_df(n=60):
    rng = np.random.default_rng(0)
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


class TestModelTrainer(unittest.TestCase):

    @mock.patch("src.dscProject.components.model_trainer.joblib.dump")
    @mock.patch("src.dscProject.components.model_trainer.create_directories")
    @mock.patch("src.dscProject.components.model_trainer.pd.read_csv")
    def test_model_is_saved(self, mock_read_csv, _mock_dirs, mock_dump):
        mock_read_csv.return_value = _sample_df()
        ModelTrainer(config=_make_config()).train_model()
        mock_dump.assert_called_once()

    @mock.patch("src.dscProject.components.model_trainer.joblib.dump")
    @mock.patch("src.dscProject.components.model_trainer.create_directories")
    @mock.patch("src.dscProject.components.model_trainer.pd.read_csv")
    def test_saved_model_is_random_forest(self, mock_read_csv, _mock_dirs, mock_dump):
        mock_read_csv.return_value = _sample_df()
        ModelTrainer(config=_make_config()).train_model()
        saved_model = mock_dump.call_args[0][0]
        self.assertIsInstance(saved_model, RandomForestClassifier)

    @mock.patch("src.dscProject.components.model_trainer.joblib.dump")
    @mock.patch("src.dscProject.components.model_trainer.create_directories")
    @mock.patch("src.dscProject.components.model_trainer.pd.read_csv")
    def test_model_path_contains_model_name(self, mock_read_csv, _mock_dirs, mock_dump):
        mock_read_csv.return_value = _sample_df()
        config = _make_config()
        ModelTrainer(config=config).train_model()
        saved_path = mock_dump.call_args[0][1]
        self.assertIn(config.model_name, str(saved_path))

    @mock.patch("src.dscProject.components.model_trainer.joblib.dump")
    @mock.patch("src.dscProject.components.model_trainer.create_directories")
    @mock.patch("src.dscProject.components.model_trainer.pd.read_csv")
    def test_random_forest_uses_configured_hyperparams(self, mock_read_csv, _mock_dirs, mock_dump):
        mock_read_csv.return_value = _sample_df()
        config = _make_config()
        ModelTrainer(config=config).train_model()
        saved_model: RandomForestClassifier = mock_dump.call_args[0][0]
        self.assertEqual(saved_model.n_estimators, config.n_estimators)
        self.assertEqual(saved_model.max_depth, config.max_depth)
        self.assertEqual(saved_model.min_samples_split, config.min_samples_split)


if __name__ == "__main__":
    unittest.main()
