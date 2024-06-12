from dataclasses import dataclass
from pathlib import Path
from datetime import date

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    date_column: str
    product_column: str
    quantity_column: str
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    horizon: int
    date_column: str
    product_column: str
    quantity_column: str
    
@dataclass(frozen=True)
class ModelTrainerconfig_LR:
    root_dir: Path
    model_dir_name: str
    horizon: int
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    
    
@dataclass(frozen=True)
class ModelTrainerconfig_XGBoost:
    root_dir: Path
    model_dir_name: str
    horizon: int
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    
@dataclass(frozen=True)
class ModelTrainerconfig_SVR: 
    root_dir: Path
    model_dir_name: str
    horizon: int
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    
@dataclass(frozen=True)
class ModelTrainerconfig_LSTM:
    root_dir: Path
    model_dir_name: str
    learning_rate: float
    early_stopping: int
    epochs: int
    horizon: int
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    
    
@dataclass(frozen=True)
class ModelTrainerconfig_CNN:
    root_dir: Path
    model_dir_name: str
    learning_rate: float
    early_stopping: int
    epochs: int
    horizon: int
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    
@dataclass(frozen=True)
class ModelTrainerconfig_GRU:
    root_dir: Path
    model_dir_name: str
    learning_rate: float
    early_stopping: int
    epochs: int
    horizon: int
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    models_dir_names: dict
    horizon: int
    
    learning_rate_CNN: float
    early_stopping_CNN: int
    epochs_CNN: int
    
    learning_rate_LSTM: float
    early_stopping_LSTM: int
    epochs_LSTM: int
    
    learning_rate_GRU: float
    early_stopping_GRU: int
    epochs_GRU: int
    
    data_path: Path
    date_column: str
    product_column: str
    quantity_column: str
    X_scaled_path : Path
    y_scaled_path : Path
    scaler_path : Path
    

@dataclass(frozen=True)
class ModelSelectionConfig:
    root_dir: Path
    eval_dir: Path
    models_dir: Path
    horizon: int