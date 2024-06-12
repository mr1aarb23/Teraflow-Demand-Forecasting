
import os
import pandas as pd
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def get_file(self):
        df = pd.read_csv(self.config.local_data_file)
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        
        columns_to_keep = [self.config.date_column,self.config.product_column,self.config.quantity_column]
        df_filtered = df.loc[:, columns_to_keep]
        
        df_filtered[self.config.date_column] = pd.to_datetime(df_filtered[self.config.date_column], errors='coerce', format='%Y-%m-%dT%H:%M', exact=False)
        df_filtered[self.config.date_column] = df_filtered[self.config.date_column].dt.date
        df_filtered[self.config.date_column] = pd.to_datetime(df_filtered[self.config.date_column])

        #df_sorted = df.sort_values(by=[self.config.date_column,self.config.product_column])
        df_filtered.to_csv(os.path.join(self.config.root_dir,"data.csv"),index=False)