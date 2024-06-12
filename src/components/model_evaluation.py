from src.config.configuration import ModelEvaluationConfig
from src.utils.common import save_json
from pathlib import Path

import pandas as pd
import os
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import ShuffleSplit
import numpy as np
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Conv1D, Flatten , Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import logging

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_data(self):
        try:
            X = pd.read_csv(self.config.X_scaled_path, header=None)
            y = pd.read_csv(self.config.y_scaled_path, header=None)
            return np.array(X), np.array(y)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def monthly_metrics(self, y_true, y_pred):
        months = self.config.horizon // 30
        mse_scores = []
        rmse_scores = []
        mae_scores = []
        smape_scores = []

        for month in range(months):
            start = month * 30
            end = start + 30
            y_true_month = y_true[:, start:end]
            y_pred_month = y_pred[:, start:end]

            mse = mean_squared_error(y_true_month, y_pred_month)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_month, y_pred_month)
            smape = 100 * np.mean(2 * np.abs(y_pred_month - y_true_month) / (np.abs(y_pred_month) + np.abs(y_true_month) + 1e-10))

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            smape_scores.append(smape)

        return np.mean(mse_scores), np.mean(rmse_scores), np.mean(mae_scores), np.mean(smape_scores)
    
    def cross_validation(self, X, y, model_func, model_name):
        n_splits = 5
        rs = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        mse_scores = []
        rmse_scores = []
        mae_scores = []
        smape_scores = []
        training_time = []
        scaler = joblib.load(self.config.scaler_path)
        
        for train_idx, val_idx in rs.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = model_func()
            if model_name == "XGBoost":
                start_time = time()
                model.fit(X_train_fold, y_train_fold, eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)], verbose=False)
                end_time = time()
            elif model_name == "GRU":
                start_time = time()
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.config.early_stopping_GRU, mode='min')
                model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=self.config.epochs_GRU, callbacks=[early_stopping], verbose=0)
                end_time = time()
            elif model_name == "LSTM":
                start_time = time()
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.config.early_stopping_LSTM, mode='min')
                model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=self.config.epochs_LSTM, callbacks=[early_stopping], verbose=0)
                end_time = time()
            elif model_name == "CNN":
                start_time = time()
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.config.early_stopping_CNN, mode='min')
                model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=self.config.epochs_CNN, callbacks=[early_stopping], verbose=0)
                end_time = time()
            else:
                start_time = time()
                model.fit(X_train_fold, y_train_fold)
                end_time = time()

            y_val_pred = model.predict(X_val_fold)
            
            y_val_fold_unscaled = scaler.inverse_transform(y_val_fold)
            y_val_pred_unscaled = scaler.inverse_transform(y_val_pred)

            mse_fold, rmse_fold, mae_fold, smape_fold = self.monthly_metrics(y_val_fold_unscaled, y_val_pred_unscaled)

            mse_scores.append(mse_fold)
            rmse_scores.append(rmse_fold)
            mae_scores.append(mae_fold)
            smape_scores.append(smape_fold)
            training_time.append(end_time - start_time)

        metrics = {
            'Model': model_name,
            'Mean_Val_MSE': np.mean(mse_scores),
            'Mean_Val_RMSE': np.mean(rmse_scores),
            'Mean_Val_MAE': np.mean(mae_scores),
            'Mean_Val_SMAPE': np.mean(smape_scores),
            'Mean_Val_Training_time': np.mean(training_time)
        }

        save_json(path=Path(os.path.join(self.config.root_dir, f"{model_name}_evaluation_metrics.json")), data=metrics)

    def LR_cross_validation(self, X, y):
        logging.info("Starting cross-validation for Linear Regression")
        self.cross_validation(X, y, LinearRegression, "LR")

    def SVR_cross_validation(self, X, y):
        logging.info("Starting cross-validation for SVR")
        def svr_func():
            svr = SVR(kernel='rbf', C=100, gamma=0.001)
            return MultiOutputRegressor(svr)
        self.cross_validation(X, y, svr_func, "SVR")

    def XGBoost_cross_validation(self, X, y):
        logging.info("Starting cross-validation for XGBoost")
        def xgb_func():
            return XGBRegressor(n_estimators=10, early_stopping_rounds=10)
        self.cross_validation(X, y, xgb_func, "XGBoost")

    def CNN_cross_validation(self, X, y):
        logging.info("Starting cross-validation for CNN")
        def cnn_func():
            model = Sequential([
                Input((self.config.horizon, 1)),
                Conv1D(64, kernel_size=2, activation='relu'),
                Flatten(),
                Dense(100, activation='relu'),
                Dense(self.config.horizon, activation='linear')
            ])

            optimizer = Adam(learning_rate=self.config.learning_rate_CNN)
            model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])
            return model
        self.cross_validation(X, y, cnn_func, "CNN")

    def LSTM_cross_validation(self, X, y):
        logging.info("Starting cross-validation for LSTM")
        def lstm_func():
            model = Sequential([
                Input((self.config.horizon, 1)),
                LSTM(128),
                Dense(64, activation='relu'),
                Dense(self.config.horizon, activation='linear')
            ])

            optimizer = Adam(learning_rate=self.config.learning_rate_LSTM)
            model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])
            return model
        self.cross_validation(X, y, lstm_func, "LSTM")

    def GRU_cross_validation(self, X, y):
        logging.info("Starting cross-validation for GRU")
        def gru_func():
            model = Sequential([
                Input((self.config.horizon, 1)),
                GRU(64, return_sequences=True),
                GRU(32, return_sequences=True),
                GRU(8),
                Dense(self.config.horizon)
            ])

            optimizer = Adam(learning_rate=self.config.learning_rate_GRU)
            model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])
            return model
        self.cross_validation(X, y, gru_func, "GRU")