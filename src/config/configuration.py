from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig , DataTransformationConfig , ModelTrainerconfig_LR, ModelTrainerconfig_XGBoost,  ModelTrainerconfig_SVR, ModelTrainerconfig_CNN, ModelTrainerconfig_LSTM, ModelTrainerconfig_GRU, ModelEvaluationConfig,ModelSelectionConfig
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            horizon=horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return data_transformation_config
    
    def get_model_trainer_config_LR(self) -> ModelTrainerconfig_LR:
        config = self.config.model_trainer_LR
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.model_dir_name])

        
        model_trainer_config = ModelTrainerconfig_LR(
            root_dir=config.root_dir,
            model_dir_name=config.model_dir_name,
            data_path=config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon=horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return model_trainer_config
    
    def get_model_trainer_config_XGBoost(self) -> ModelTrainerconfig_XGBoost:
        config = self.config.model_trainer_XGBoost
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.model_dir_name])

        
        model_trainer_config = ModelTrainerconfig_XGBoost(
            root_dir=config.root_dir,
            model_dir_name=config.model_dir_name,
            data_path=config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon=horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return model_trainer_config
    
    def get_model_trainer_config_SVR(self) -> ModelTrainerconfig_SVR:
        config = self.config.model_trainer_SVR
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.model_dir_name])

        
        model_trainer_config = ModelTrainerconfig_SVR(
            root_dir=config.root_dir,
            model_dir_name=config.model_dir_name,
            data_path=config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon=horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return model_trainer_config
    
    def get_model_trainer_config_CNN(self) -> ModelTrainerconfig_CNN:
        config = self.config.model_trainer_CNN
        learning_rate = self.params.CNN.learning_rate
        early_stopping = self.params.CNN.early_stopping 
        epochs = self.params.CNN.epochs 
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.model_dir_name])

        
        model_trainer_config = ModelTrainerconfig_CNN(
            root_dir = config.root_dir,
            model_dir_name = config.model_dir_name,
            learning_rate = learning_rate,
            early_stopping = early_stopping,
            epochs = epochs,
            data_path=config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon = horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return model_trainer_config
    
    def get_model_trainer_config_LSTM(self) -> ModelTrainerconfig_LSTM:
        config = self.config.model_trainer_LSTM
        learning_rate = self.params.LSTM.learning_rate
        early_stopping = self.params.LSTM.early_stopping 
        epochs = self.params.LSTM.epochs 
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.model_dir_name])

        
        model_trainer_config = ModelTrainerconfig_LSTM(
            root_dir = config.root_dir,
            model_dir_name = config.model_dir_name,
            learning_rate = learning_rate,
            early_stopping = early_stopping,
            epochs = epochs,
            data_path=config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon = horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return model_trainer_config
    
    def get_model_trainer_config_GRU(self) -> ModelTrainerconfig_GRU:
        config = self.config.model_trainer_GRU
        learning_rate = self.params.GRU.learning_rate
        early_stopping = self.params.GRU.early_stopping 
        epochs = self.params.GRU.epochs 
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        
        create_directories([config.model_dir_name])

        
        model_trainer_config = ModelTrainerconfig_GRU(
            root_dir = config.root_dir,
            model_dir_name = config.model_dir_name,
            learning_rate = learning_rate,
            early_stopping = early_stopping,
            epochs = epochs,
            data_path=config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon = horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        horizon = self.params.Horizon.H
        date_column = self.schema.ColumnsName.OrderDate
        product_column = self.schema.ColumnsName.ProductID
        quantity_column = self.schema.ColumnsName.Quantity
        models_dir_names = self.schema.models_dir_names
        learning_rate_CNN = self.params.CNN.learning_rate
        early_stopping_CNN = self.params.CNN.early_stopping 
        epochs_CNN = self.params.CNN.epochs
        learning_rate_LSTM = self.params.LSTM.learning_rate
        early_stopping_LSTM = self.params.LSTM.early_stopping 
        epochs_LSTM = self.params.LSTM.epochs 
        learning_rate_GRU = self.params.GRU.learning_rate
        early_stopping_GRU = self.params.GRU.early_stopping 
        epochs_GRU = self.params.GRU.epochs  
        
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            models_dir_names = models_dir_names,
            data_path = config.data_path,
            X_scaled_path = config.X_scaled_path,
            y_scaled_path = config.y_scaled_path,
            horizon = horizon,
            date_column = date_column,
            product_column = product_column,
            quantity_column = quantity_column,
            scaler_path = config.scaler_path,
            learning_rate_CNN = learning_rate_CNN,
            early_stopping_CNN = early_stopping_CNN,
            epochs_CNN = epochs_CNN, 
            learning_rate_LSTM = learning_rate_LSTM,
            early_stopping_LSTM = early_stopping_LSTM,
            epochs_LSTM = epochs_LSTM,
            learning_rate_GRU = learning_rate_GRU,
            early_stopping_GRU = early_stopping_GRU,
            epochs_GRU = epochs_GRU
        )
        return model_evaluation_config
    
    def get_model_selection_config(self) -> ModelSelectionConfig:
        config = self.config.model_selection
        horizon = self.params.Horizon.H
        
        create_directories([config.root_dir])
        create_directories([os.path.join(config.root_dir,"Forecast_model")])

        model_selection_config = ModelSelectionConfig(
            root_dir = config.root_dir,
            eval_dir = config.eval_dir,
            models_dir = config.models_dir,
            horizon = horizon
        )
        return model_selection_config