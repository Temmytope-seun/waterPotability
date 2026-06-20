import unittest
from unittest import mock
from pathlib import Path

from src.waterPotability.entity.config_entity import DataIngestionConfig
from src.waterPotability.components.data_ingestion import DataIngestion


def _make_config():
    return DataIngestionConfig(
        root_dir=Path("artifacts/data_ingestion"),
        source_URL="https://example.com/water_potability.csv",
        local_data_file=Path("artifacts/data_ingestion/water_potability.csv"),
    )


class TestDataIngestion(unittest.TestCase):

    @mock.patch("src.waterPotability.components.data_ingestion.os.path.exists", return_value=False)
    @mock.patch("src.waterPotability.components.data_ingestion.request.urlretrieve")
    def test_download_file_fetches_when_missing(self, mock_retrieve, _mock_exists):
        mock_retrieve.return_value = ("water_potability.csv", {})
        config = _make_config()
        DataIngestion(config=config).download_file()
        mock_retrieve.assert_called_once_with(
            url=config.source_URL,
            filename=config.local_data_file,
        )

    @mock.patch("src.waterPotability.components.data_ingestion.os.path.exists", return_value=True)
    @mock.patch("src.waterPotability.components.data_ingestion.request.urlretrieve")
    def test_download_file_skips_when_present(self, mock_retrieve, _mock_exists):
        DataIngestion(config=_make_config()).download_file()
        mock_retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
