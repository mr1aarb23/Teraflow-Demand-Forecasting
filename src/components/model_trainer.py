from src.entity.config_entity import ModelTrainerconfig_LR , ModelTrainerconfig_XGBoost, ModelTrainerconfig_SVR,ModelTrainerconfig_CNN, ModelTrainerconfig_LSTM,ModelTrainerconfig_GRU
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,GRU,Dense, Flatten, Conv1D, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError

class ModelTrainer_LR:
    def __init__(self,config:ModelTrainerconfig_LR):
        self.config = config
        
    def load_data(self):
        X_scaled =pd.read_csv(self.config.X_scaled_path,header=None)
        y_scaled =pd.read_csv(self.config.y_scaled_path,header=None)
        return np.array(X_scaled), np.array(y_scaled)
        
    def train(self,X,y):
        lr = LinearRegression()
        lr.fit(X, y)
        joblib.dump(lr,os.path.join(self.config.model_dir_name,"LR.joblib"))
        return lr

class ModelTrainer_XGBoost:
    def __init__(self,config:ModelTrainerconfig_XGBoost):
        self.config = config
        
    def load_data(self):
        X_scaled =pd.read_csv(self.config.X_scaled_path)
        y_scaled =pd.read_csv(self.config.y_scaled_path)
        return np.array(X_scaled), np.array(y_scaled)
        
    def train(self,X,y):
        X_train, X_val, y_train,y_val = train_test_split(X,y)
        reg = XGBRegressor(n_estimators=10, early_stopping_rounds=10)
        reg.fit(X_train,y_train, eval_set=[(X_train,y_train), (X_val, y_val)], verbose=True)

        joblib.dump(reg, os.path.join(self.config.model_dir_name, "XGBoost.joblib"))
        
class ModelTrainer_SVR:
    def __init__(self, config):
        self.config = config
        
    def load_data(self):
        X_scaled = pd.read_csv(self.config.X_scaled_path,header=None)
        y_scaled = pd.read_csv(self.config.y_scaled_path,header=None)
        return np.array(X_scaled), np.array(y_scaled)
        
    def train(self, X, y):
        #svr = SVR()
        #multioutput_svr = MultiOutputRegressor(svr)
        #
        #param_grid = {
        #    'estimator__kernel': ['rbf'],
        #    'estimator__C': [1, 10, 100],
        #    'estimator__gamma': [0.1, 0.01, 0.001]
        #}
#
        ## Create the GridSearchCV object
        #grid = GridSearchCV(estimator=multioutput_svr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
#
        #grid_result = grid.fit(X, y)
#
        ## Instantiate SVR with the best parameters
        #best_svr = SVR(kernel=grid_result.best_params_['estimator__kernel'],
        #               C=grid_result.best_params_['estimator__C'],
        #               gamma=grid_result.best_params_['estimator__gamma'])
        #best_multioutput_svr = MultiOutputRegressor(best_svr)

        # Train the model with the best parameters
        svr = SVR(kernel='rbf', C=100, gamma=0.001)
        multioutput_svr = MultiOutputRegressor(svr)
        multioutput_svr.fit(X, y)

        # Save the trained model
        joblib.dump(multioutput_svr, os.path.join(self.config.model_dir_name, "SVR.joblib"))


class ModelTrainer_CNN:
    def __init__(self,config:ModelTrainerconfig_CNN):
        self.config = config
        
    def load_data(self):
        X_scaled = pd.read_csv(self.config.X_scaled_path,header=None)
        y_scaled = pd.read_csv(self.config.y_scaled_path,header=None)
        return np.array(X_scaled), np.array(y_scaled)
    
    def create_model(self,past_time_steps, future_time_steps):
        model = Sequential([
            Input((past_time_steps, 1)),
            Conv1D(64, kernel_size=2, activation='relu'),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(future_time_steps, activation='linear')
        ])

        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])

        return model
        
    def train(self,X,y):
        X_train, X_val, y_train,y_val = train_test_split(X,y)
        model = self.create_model(self.config.horizon,self.config.horizon)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=self.config.early_stopping,mode='min')

        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.config.epochs,callbacks=[early_stopping])
        model.save(os.path.join(self.config.model_dir_name,"CNN.keras"))
        
class ModelTrainer_LSTM:
    def __init__(self,config:ModelTrainerconfig_LSTM):
        self.config = config
        
    def load_data(self):
        X_scaled = pd.read_csv(self.config.X_scaled_path,header=None)
        y_scaled = pd.read_csv(self.config.y_scaled_path,header=None)
        return np.array(X_scaled), np.array(y_scaled)
    
    def create_model(self,past_time_steps, future_time_steps):
        model = Sequential([Input((past_time_steps,1)),
                            LSTM(128),
                            Dense(64, activation='relu'),
                            Dense(future_time_steps, activation='linear')])

        
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])

        return model
        
    def train(self,X,y):
        X_train, X_val, y_train,y_val = train_test_split(X,y)
        model = self.create_model(self.config.horizon,self.config.horizon)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=self.config.early_stopping,mode='min')

        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.config.epochs,callbacks=[early_stopping])
        model.save(os.path.join(self.config.model_dir_name,"LSTM.keras"))
        
class ModelTrainer_GRU:
    def __init__(self,config:ModelTrainerconfig_GRU):
        self.config = config
        
    def load_data(self):
        X_scaled = pd.read_csv(self.config.X_scaled_path,header=None)
        y_scaled = pd.read_csv(self.config.y_scaled_path,header=None)
        return np.array(X_scaled), np.array(y_scaled)
    
    def create_model(self,past_time_steps, future_time_steps):
        model = Sequential([
                Input(shape=(past_time_steps, 1)),
                GRU(64, return_sequences=True),
                GRU(32, return_sequences=True),
                GRU(8),
                Dense(future_time_steps) 
            ])

        
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])

        return model
        
    def train(self,X,y):
        X_train, X_val, y_train,y_val = train_test_split(X,y)
        model = self.create_model(self.config.horizon,self.config.horizon)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=self.config.early_stopping,mode='min')

        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.config.epochs,callbacks=[early_stopping])
        model.save(os.path.join(self.config.model_dir_name,"GRU.keras"))