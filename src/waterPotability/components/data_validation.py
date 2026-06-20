from src.waterPotability.entity.config_entity import DataValidationConfig
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            validation_status = all(col in all_schema for col in all_cols)

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")
                if not validation_status:
                    invalid_cols = set(all_cols) - set(all_schema)
                    f.write(f"\nInvalid columns: {list(invalid_cols)}")

            return validation_status
        except Exception as e:
            raise e
