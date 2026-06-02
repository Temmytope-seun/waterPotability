from src.dscProject.config.configuration import ConfigurationManager
from src.dscProject.components.data_transformation import DataTransformation
from src.dscProject import logger

from pathlib import Path

STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def initiate_data_transformation(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
            if status == "True":
                config=ConfigurationManager()
                data_transformation_config=config.get_data_transformation_config()
                data_transformation=DataTransformation(config=data_transformation_config)
                data_transformation.train_test_split()
            else:
                logger.info("Data validation failed. Data transformation cannot be initiated.")
                raise Exception("Data validation failed. Data transformation cannot be initiated.")
        except Exception as e:
            logger.exception(e)
            raise e