from src.waterPotability.config.configuration import ConfigurationManager
from src.waterPotability.components.model_trainer import ModelTrainer
from src.waterPotability import logger

STAGE_NAME="Model Trainer Stage"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def initiate_model_trainer(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainer(config=model_trainer_config)
        model_trainer.train_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
        obj =ModelTrainingPipeline()
        obj.initiate_model_trainer()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=====x")
    except Exception as e:
        logger.exception(e)
        raise e