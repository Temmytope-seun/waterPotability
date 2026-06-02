import os
from src.dscProject import logger
from src.dscProject.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import ElasticNet
import joblib
import pandas as pd
from src.dscProject.utils.common import create_directories


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        logger.info("Loading training data")
        train_data = pd.read_csv(self.config.train_data_path)

        logger.info("Splitting features and target variable")
        X_train = train_data.drop(columns=[self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]

        logger.info("Training the ElasticNet model")
        model = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        model_dir = os.path.join(self.config.root_dir, "model")
        create_directories([model_dir])

        model_path = os.path.join(model_dir, self.config.model_name)
        joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}")