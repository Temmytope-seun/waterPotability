import os
from src.dscProject import logger
from src.dscProject.entity.config_entity import ModelTrainerConfig
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from src.dscProject.utils.common import create_directories


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        logger.info("Loading training data")
        train_data = pd.read_csv(self.config.train_data_path)

        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]

        logger.info("Training RandomForestClassifier")
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)

        model_dir = os.path.join(self.config.root_dir, "model")
        create_directories([model_dir])

        model_path = os.path.join(model_dir, self.config.model_name)
        joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}")
