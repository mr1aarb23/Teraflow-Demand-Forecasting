import os
from src.utils import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config
        
    def replace_outliers_with_median(self,series):
            median = series.median()
            std = series.std()
            outliers = (series - median).abs() > 3 * std
            series.loc[outliers] = median
            return series
        
    def preprocess_demand_data(self):
        df = df = pd.read_csv(self.config.data_path)
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])
        df_agg = df.groupby([self.config.date_column, self.config.product_column])[self.config.quantity_column].sum().reset_index()

        all_dates = pd.date_range(start=df[self.config.date_column].min(), end=df[self.config.date_column].max())
        all_products = df[self.config.product_column].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_products], names=[self.config.date_column, self.config.product_column])

        df_full = df_agg.set_index([self.config.date_column, self.config.product_column]).reindex(full_index, fill_value=0).reset_index()

        df_full[self.config.quantity_column] = self.replace_outliers_with_median(df_full[self.config.quantity_column].copy())
        df_full.to_csv(os.path.join(self.config.root_dir,"train_data.csv"),index=False)

        return df_full
    
    def data_windowing(self,df_):
        df = df_.copy()
        combined_df_list = []

        for p in df[self.config.product_column].unique():
            data = df[df[self.config.product_column] == p][[self.config.date_column, self.config.quantity_column]].copy()
            quantity_values = data[self.config.quantity_column].values
            
            if len(quantity_values) == 2 * self.config.horizon:
                X = np.array([quantity_values[:self.config.horizon]])
                Y = np.array([quantity_values[self.config.horizon:]])
            else:
                X = np.array([quantity_values[i:i + self.config.horizon] for i in range(len(quantity_values) - 2 * self.config.horizon)])
                Y = np.array([quantity_values[i + self.config.horizon:i + 2 * self.config.horizon] for i in range(len(quantity_values) - 2 * self.config.horizon)])
        
            
            X_columns = [f'X (t-{self.config.horizon-i})' for i in range(self.config.horizon)]
            Y_columns = [f'Y (t+{i+1})' for i in range(self.config.horizon)]
            columns =  X_columns + Y_columns

            product_df = pd.DataFrame(np.hstack((X,Y)), columns=columns)
            product_df["Product ID"] = p

            combined_df_list.append(product_df)
        
        combined_df = pd.concat(combined_df_list, ignore_index=True)

        X = combined_df.iloc[:, list(range(self.config.horizon))]
        y = combined_df.iloc[:, self.config.horizon: self.config.horizon*2]        
        
        combined_df.to_csv(os.path.join(self.config.root_dir,"data_windowing.csv"),index=False)
        X.to_csv(os.path.join(self.config.root_dir,"X.csv"),index=False)
        y.to_csv(os.path.join(self.config.root_dir,"y.csv"),index=False)
        
        
        return combined_df, X, y

    def scale_data(self,df, X, y):
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.array(X))
        y_scaled = scaler.transform(np.array(y))
        
        joblib.dump(scaler,os.path.join(os.path.join(self.config.root_dir,"scaler.pkl")))

        np.savetxt(os.path.join(self.config.root_dir,"X_scaled.csv"), X_scaled, delimiter=',')
        np.savetxt(os.path.join(self.config.root_dir,"y_scaled.csv"), y_scaled, delimiter=',')
