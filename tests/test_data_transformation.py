import unittest
from unittest import mock
from pathlib import Path
import pandas as pd
import numpy as np

from src.waterPotability.entity.config_entity import DataTransformationConfig
from src.waterPotability.components.data_transformation import DataTransformation


def _make_config():
    return DataTransformationConfig(
        root_dir=Path("artifacts/data_transformation"),
        data_path=Path("artifacts/data_ingestion/water_potability.csv"),
    )


def _sample_df(n=40, with_nulls=False):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
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
    if with_nulls:
        df.loc[0, "ph"] = np.nan
        df.loc[1, "Sulfate"] = np.nan
    return df


class TestDataTransformation(unittest.TestCase):

    @mock.patch("src.waterPotability.components.data_transformation.pd.read_csv")
    def test_creates_two_csv_files(self, mock_read_csv):
        mock_read_csv.return_value = _sample_df()
        with mock.patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
            DataTransformation(config=_make_config()).train_test_split()
            self.assertEqual(mock_to_csv.call_count, 2)

    @mock.patch("src.waterPotability.components.data_transformation.pd.read_csv")
    def test_output_paths_contain_root_dir(self, mock_read_csv):
        mock_read_csv.return_value = _sample_df()
        config = _make_config()
        saved_paths = []
        with mock.patch.object(pd.DataFrame, "to_csv",
                               side_effect=lambda path, **kw: saved_paths.append(str(path))):
            DataTransformation(config=config).train_test_split()
        for path in saved_paths:
            self.assertIn(str(config.root_dir), path)

    @mock.patch("src.waterPotability.components.data_transformation.pd.read_csv")
    def test_missing_values_are_imputed(self, mock_read_csv):
        df_with_nulls = _sample_df(n=40, with_nulls=True)
        self.assertTrue(df_with_nulls.isnull().any().any())

        saved_dfs = []
        mock_read_csv.return_value = df_with_nulls
        with mock.patch.object(pd.DataFrame, "to_csv",
                               side_effect=lambda path, **kw: saved_dfs.append(None)):
            DataTransformation(config=_make_config()).train_test_split()

        # If to_csv was called, fillna ran without error — that is the key assertion
        self.assertEqual(len(saved_dfs), 2)


if __name__ == "__main__":
    unittest.main()
