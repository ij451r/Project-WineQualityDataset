from WineQuality.config.configuration import ConfigurationManager
from WineQuality.components.data_transformation import DataTransformation 
from WineQuality import logger

STAGE_NAME = "Data DataTransformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.drop_non_validated_columns()
        data_transformation.train_test_split()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        data_transformation_training_pipeline = DataTransformationTrainingPipeline()
        data_transformation_training_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)