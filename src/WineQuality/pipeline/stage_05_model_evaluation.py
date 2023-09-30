from WineQuality.config.configuration import ConfigurationManager
from WineQuality import logger
from WineQuality.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_eval_config)
        model_evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        model_evaluation_training_pipeline = ModelEvaluationTrainingPipeline()
        model_evaluation_training_pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    except Exception as e:
        logger.exception(e)