from WineQuality.config.configuration import ConfigurationManager
from WineQuality import logger
from WineQuality.components.model_trainer import ModelTrainer

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_config)
        model_trainer.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        model_trainer_training_pipeline = ModelTrainerTrainingPipeline()
        model_trainer_training_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)