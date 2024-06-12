import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import tensorflow as tf
from src.utils.common import load_json, read_yaml
from datetime import datetime, timedelta


class ModelForecastConfig:
    def __init__(self, data, params_filepath=Path("params.yaml"), schema_filepath=Path("schema.yaml")):
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        self.horizon = self.params.Horizon.H
        self.product_column = self.schema.ColumnsName.ProductID
        self.quantity_column = self.schema.ColumnsName.Quantity
        self.date_column = self.schema.ColumnsName.OrderDate
        self.data = data
        self.scaler_path = "src/artifacts/data_transformation/scaler.pkl"
        self.model_dir = "src/artifacts/model_selection/Forecast_model"


class ModelForecast:
    def __init__(self, config: ModelForecastConfig):
        self.config = config

    def replace_outliers_with_median(self, series):
        median = series.median()
        std = series.std()
        outliers = (series - median).abs() > 3 * std
        series.loc[outliers] = median
        return series

    def preprocess_demand_data(self, df):
        df[self.config.date_column] = pd.to_datetime(
            df[self.config.date_column])
        df_agg = df.groupby([self.config.date_column, self.config.product_column])[
            self.config.quantity_column].sum().reset_index()

        all_dates = pd.date_range(
            start=df[self.config.date_column].min(), end=df[self.config.date_column].max())
        all_products = df[self.config.product_column].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_products], names=[
                                                self.config.date_column, self.config.product_column])

        df_full = df_agg.set_index([self.config.date_column, self.config.product_column]).reindex(
            full_index, fill_value=0).reset_index()

        df_full[self.config.quantity_column] = self.replace_outliers_with_median(
            df_full[self.config.quantity_column].copy())

        return df_full

    def load_model(self):
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
        for file_name in os.listdir(self.config.model_dir):
            model_file = Path(self.config.model_dir) / file_name
            if model_file.suffix == '.joblib':
                return joblib.load(model_file)
            elif model_file.suffix in ['.h5', '.keras']:
                return tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        raise FileNotFoundError(
            "No suitable model file found in the specified directory.")

    def predict(self, model=None):
        if model is None:
            model = self.load_model()

        df = self.preprocess_demand_data(self.config.data)
        product_ids = df[self.config.product_column].unique()
        # date_obj = datetime.strptime(
        #    df[self.config.date_column].max(), "%Y-%m-%d")
        # new_date = date_obj + timedelta(days=-self.config.horizon)
        # start_date = new_date.strftime("%Y-%m-%d")
        # df = df[df[self.config.date_column] > start_date]

        scaler = joblib.load(self.config.scaler_path)
        product_groups = df.groupby(self.config.product_column)
        X_data_list = []

        for product, group in product_groups:
            X_data = np.array(
                group[self.config.quantity_column][-self.config.horizon:]).reshape(1, -1)
            X_data_list.append(X_data)

        X_data_batch = np.vstack(X_data_list)

        X_data_batch_transformed = scaler.transform(X_data_batch)
        forecast_batch = model.predict(X_data_batch_transformed)
        forecast_batch_inversed = scaler.inverse_transform(forecast_batch)
        forecast_batch_rounded = np.round(forecast_batch_inversed)

        return forecast_batch_rounded, product_ids

    def create_forecast_table(self, forecast_batch_rounded, product_ids):

        number_of_products = len(product_ids)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        monthly_forecasts = np.zeros((number_of_products, 12))
        day_index = 0
        for month in range(12):
            monthly_forecasts[:, month] = forecast_batch_rounded[:, day_index:day_index + days_in_month[month]].sum(axis=1)
            day_index += days_in_month[month]

        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN","JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        product_ids = [f"{p}" for p in product_ids]
        forecast_df = pd.DataFrame(monthly_forecasts, columns=months)
        forecast_df.insert(0, 'Product', product_ids)

        return forecast_df
