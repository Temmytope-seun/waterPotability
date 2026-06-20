import unittest
from unittest import mock
from pathlib import Path
import pandas as pd

from src.waterPotability.entity.config_entity import DataValidationConfig
from src.waterPotability.components.data_validation import DataValidation


SCHEMA = {
    "ph": "float64",
    "Hardness": "float64",
    "Solids": "float64",
    "Chloramines": "float64",
    "Sulfate": "float64",
    "Conductivity": "float64",
    "Organic_carbon": "float64",
    "Trihalomethanes": "float64",
    "Turbidity": "float64",
    "Potability": "int64",
}


def _make_config():
    return DataValidationConfig(
        root_dir=Path("artifacts/data_validation"),
        STATUS_FILE="artifacts/data_validation/status.txt",
        data_dir=Path("artifacts/data_ingestion/water_potability.csv"),
        all_schema=SCHEMA,
    )


class TestDataValidation(unittest.TestCase):

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("src.waterPotability.components.data_validation.pd.read_csv")
    def test_returns_true_for_valid_columns(self, mock_read_csv, _mock_open):
        mock_read_csv.return_value = pd.DataFrame(columns=list(SCHEMA.keys()))
        result = DataValidation(config=_make_config()).validate_all_columns()
        self.assertTrue(result)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("src.waterPotability.components.data_validation.pd.read_csv")
    def test_returns_false_for_extra_columns(self, mock_read_csv, _mock_open):
        columns = list(SCHEMA.keys()) + ["unexpected_col"]
        mock_read_csv.return_value = pd.DataFrame(columns=columns)
        result = DataValidation(config=_make_config()).validate_all_columns()
        self.assertFalse(result)

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("src.waterPotability.components.data_validation.pd.read_csv")
    def test_invalid_columns_do_not_raise_type_error(self, mock_read_csv, _mock_open):
        mock_read_csv.return_value = pd.DataFrame(columns=["ph", "bad_col"])
        try:
            DataValidation(config=_make_config()).validate_all_columns()
        except TypeError:
            self.fail("validate_all_columns raised TypeError on invalid columns")

    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("src.waterPotability.components.data_validation.pd.read_csv")
    def test_status_written_to_file(self, mock_read_csv, mock_open):
        mock_read_csv.return_value = pd.DataFrame(columns=list(SCHEMA.keys()))
        DataValidation(config=_make_config()).validate_all_columns()
        mock_open().write.assert_called()


if __name__ == "__main__":
    unittest.main()
