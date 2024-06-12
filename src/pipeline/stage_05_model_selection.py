from src.config.configuration import ConfigurationManager
from src.components.model_selection import ModelSelection
from src.utils import logger
from pathlib import Path

STAGE_NAME = "Model Selection stage"


class ModelSelectionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_selection_config = config.get_model_selection_config()
            model_selection_config = ModelSelection(config=model_selection_config)
            evaluation_metrics = model_selection_config.load_evalation_metrics_files()
            best_model = model_selection_config.get_evaluation_score(evaluation_metrics)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelSelectionPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========x")
    except Exception as e:
        logger.exception(e)
        raise e
