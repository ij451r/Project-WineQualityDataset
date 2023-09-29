from WineQuality import logger
from WineQuality.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from WineQuality.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_validation_training_pipeline = DataValidationTrainingPipeline()
    data_validation_training_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)