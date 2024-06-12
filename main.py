from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline 
from src.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.pipeline.stage_03_model_trainer import ModelTrainerPipeline 
from src.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from src.pipeline.stage_05_model_selection import ModelSelectionPipeline
from src.utils.common import logger


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(
        f">>>>>>> stage {STAGE_NAME} completed <<<<<<< \n\n x===========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(
        f">>>>>>> stage {STAGE_NAME} completed <<<<<<< \n\n x===========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(
        f">>>>>>> stage {STAGE_NAME} completed <<<<<<< \n\n x===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(
        f">>>>>>> stage {STAGE_NAME} completed <<<<<<< \n\n x===========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Selection stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelSelectionPipeline()
    obj.main()
    logger.info(
        f">>>>>>> stage {STAGE_NAME} completed <<<<<<< \n\n x===========x")
except Exception as e:
    logger.exception(e)
    raise e