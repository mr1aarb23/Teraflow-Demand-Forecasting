from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.utils import logger
from pathlib import Path

STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
            X,y = model_evaluation_config.load_data()
            model_evaluation_config.LR_cross_validation(X,y)
            model_evaluation_config.SVR_cross_validation(X,y)
            model_evaluation_config.XGBoost_cross_validation(X,y)
            model_evaluation_config.CNN_cross_validation(X,y)
            model_evaluation_config.LSTM_cross_validation(X,y)
            model_evaluation_config.GRU_cross_validation(X,y)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========x")
    except Exception as e:
        logger.exception(e)
        raise e
