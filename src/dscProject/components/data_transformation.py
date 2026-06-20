import os
from src.dscProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.dscProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self):
        logger.info("Reading data")
        df = pd.read_csv(self.config.data_path)

        logger.info("Imputing missing values with column medians")
        df.fillna(df.median(numeric_only=True), inplace=True)

        logger.info("Splitting into train and test sets")
        train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

        train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info(f"Train shape: {train_set.shape} | Test shape: {test_set.shape}")
