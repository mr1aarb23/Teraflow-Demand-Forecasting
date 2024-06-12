from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src.utils import logger

STAGE_NAME = "Data Transformation stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            df = data_transformation.preprocess_demand_data()
            combined_df, X, y = data_transformation.data_windowing(df)
            data_transformation.scale_data(combined_df, X, y)           
            
        except Exception as e:
            print(e)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========x")
    except Exception as e:
        logger.exception(e)
        raise e

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    