import os
from src.dscProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.dscProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config

    def train_test_split(self):
        logger.info("Reading data from csv file")
        df = pd.read_csv(self.config.data_path)
        logger.info("Splitting data into train and test set")
        train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
        logger.info("Saving train and test set to csv files")
        train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data transformation completed successfully")
        logger.info(train_set.shape)
        logger.info(test_set.shape)