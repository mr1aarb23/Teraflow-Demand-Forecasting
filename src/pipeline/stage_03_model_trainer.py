from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer_LR,ModelTrainer_XGBoost,ModelTrainer_CNN,ModelTrainer_LSTM,ModelTrainer_GRU,ModelTrainer_SVR
from src.utils import logger

STAGE_NAME = "Model Trainer stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config_LR()
            model_trainer_config = ModelTrainer_LR(config=model_trainer_config)
            X,y = model_trainer_config.load_data()
            lr = model_trainer_config.train(X,y)
                
        except Exception as e:
            raise e
        
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config_XGBoost()
            model_trainer_config = ModelTrainer_XGBoost(config=model_trainer_config)
            X,y = model_trainer_config.load_data()
            model_trainer_config.train(X,y)
                
        except Exception as e:
            raise e
        
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config_SVR()
            model_trainer_config = ModelTrainer_SVR(config=model_trainer_config)
            X,y = model_trainer_config.load_data()
            model_trainer_config.train(X,y)
                
        except Exception as e:
            raise e
        
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config_CNN()
            model_trainer_config = ModelTrainer_CNN(config=model_trainer_config)
            X,y = model_trainer_config.load_data()
            model_trainer_config.train(X,y)
        except Exception as e:
            raise e
        
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config_LSTM()
            model_trainer_config = ModelTrainer_LSTM(config=model_trainer_config)
            X,y = model_trainer_config.load_data()
            model_trainer_config.train(X,y)
        except Exception as e:
            raise e
        
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config_GRU()
            model_trainer_config = ModelTrainer_GRU(config=model_trainer_config)
            X,y = model_trainer_config.load_data()
            model_trainer_config.train(X,y)
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========x")
    except Exception as e:
        logger.exception(e)
        raise e
