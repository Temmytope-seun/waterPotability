from src.dscProject.config.configuration import ConfigurationManager
from src.dscProject.components.data_ingestion import DataIngestion
from src.dscProject import logger

STAGE_NAME="Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def initiate_data_ingestion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=====x")
    except Exception as e:
        logger.exception(e)
        raise e