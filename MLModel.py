import pickle
import sys

import keras
import pandas as pd
from flask import jsonify
from keras.src.callbacks import ModelCheckpoint

from constants import NOMINAL_COLUMNS, DISCRETE_COLUMNS, CONTINUOUS_COLUMNS
import numpy as np
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class MLModel:
    def __init__(self):
        # Load ML artifacts during initialization
        self.fill_values_nominal = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_nominal.pkl')
                                    if os.path.exists('artifacts/nan_outlier_handler/fill_values_nominal.pkl')
                                    else print('fill_values_nominal.pkl does not exist'))
        self.fill_values_discrete = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_discrete.pkl')
                                     if os.path.exists('artifacts/nan_outlier_handler/fill_values_discrete.pkl')
                                     else print('fill_values_discrete.pkl does not exist'))
        self.fill_values_continuous = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_continuous.pkl')
                                       if os.path.exists('artifacts/nan_outlier_handler/fill_values_continuous.pkl')
                                       else print('fill_values_continuous.pkl does not exist'))
        self.min_max_scaler_dict = (MLModel.load_model(
            'artifacts/encoders/min_max_scaler_dict.pkl')
                                    if os.path.exists('artifacts/encoders/min_max_scaler_dict.pkl')
                                    else print('min_max_scaler_dict.pkl does not exist'))
        self.onehot_encoders = (MLModel.load_model(
            'artifacts/encoders/onehot_encoders_dict.pkl')
                                if os.path.exists('artifacts/encoders/onehot_encoders_dict.pkl')
                                else print('onehot_encoders_dict.pkl does not exist'))
        self.model = (MLModel.load_model(
            'artifacts/models/model.pkl')
                      if os.path.exists('artifacts/models/model.pkl')
                      else print('model.pkl does not exist'))

    def predict(self, inference_row):
        try:
            y_pred = self.model.predict(inference_row)
            return y_pred

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return jsonify({'message': 'Internal Server Error. ',
                            'error': str(e)}), 500

    def preprocessing_pipeline(self, df):
        folder = 'artifacts/encoders'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/preprocessed_data'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/models'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/nan_outlier_handler'
        MLModel.create_new_folder(folder)

        df.drop(columns=['Unnamed: 0'], inplace=True)
        df.drop(columns=['flight'], inplace=True)

        # A dictionary that contains the mode (the most frequent value) of the categorical columns, plus column names.
        self.fill_values_nominal = {col: df[col].mode()[0] for col in NOMINAL_COLUMNS}
        self.fill_values_discrete = {col: df[col].median() for col in DISCRETE_COLUMNS}
        self.fill_values_continuous = {col: df[col].mean(skipna=True) for col in CONTINUOUS_COLUMNS}

        for col in NOMINAL_COLUMNS:
            df[col].fillna(self.fill_values_nominal[col], inplace=True)

        for col in DISCRETE_COLUMNS:
            df[col].fillna(self.fill_values_discrete[col], inplace=True)

        for col in CONTINUOUS_COLUMNS:
            df[col].fillna(self.fill_values_continuous[col], inplace=True)

        for col in CONTINUOUS_COLUMNS:
            df[col + '_zscore'] = stats.zscore(df[col])

            outlier_indices = df[abs(df[col + '_zscore']) > 3].index

            mean_value = df[col].mean()

            df.loc[outlier_indices, col] = mean_value

            df.drop(columns=[col + '_zscore'], inplace=True)

        self.onehot_encoders = {}
        new_columns = []

        for col in NOMINAL_COLUMNS:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            new_data = encoder.fit_transform(df[col].to_numpy().reshape(-1, 1))

            new_columns.extend(encoder.get_feature_names_out([col]))

            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            df = pd.concat([df, new_df], axis=1)

            self.onehot_encoders[col] = encoder

        df.drop(columns=NOMINAL_COLUMNS, inplace=True)

        self.min_max_scaler_dict = {}
        for col in df.columns:
            if not col == 'price':
                min_max_scaler = MinMaxScaler()
                df[col] = min_max_scaler.fit_transform(df[[col]])
                self.min_max_scaler_dict[col] = min_max_scaler

        MLModel.save_model(self.fill_values_nominal,
                           'artifacts/nan_outlier_handler/fill_values_nominal.pkl')
        MLModel.save_model(self.fill_values_discrete,
                           'artifacts/nan_outlier_handler/fill_values_discrete.pkl')
        MLModel.save_model(self.fill_values_continuous,
                           'artifacts/nan_outlier_handler/fill_values_continuous.pkl')
        MLModel.save_model(self.min_max_scaler_dict,
                           'artifacts/encoders/min_max_scaler_dict.pkl')
        MLModel.save_model(self.onehot_encoders,
                           'artifacts/encoders/onehot_encoders_dict.pkl')

        df = df.astype(float)
        return df

    def preprocessing_pipeline_inference(self, sample_data):
        sample_data = pd.DataFrame([sample_data])
        sample_data.columns = ['Unnamed: 0', 'airline', 'flight', 'source_city', 'departure_time',
                               'stops', 'arrival_time', 'destination_city', 'class', 'duration',
                               'days_left', 'price']
        sample_data.drop(columns=['Unnamed: 0'], inplace=True)
        sample_data.drop(columns=['flight'], inplace=True)

        for col in CONTINUOUS_COLUMNS:
            sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce')

        for col in NOMINAL_COLUMNS:
            sample_data[col] = sample_data[col].astype(type(self.fill_values_nominal[col]))
            sample_data[col].fillna(self.fill_values_nominal[col], inplace=True)

        for col in DISCRETE_COLUMNS:
            sample_data[col] = sample_data[col].astype(type(self.fill_values_discrete[col]))
            sample_data[col].fillna(self.fill_values_discrete[col], inplace=True)

        for col in CONTINUOUS_COLUMNS:
            sample_data[col] = sample_data[col].astype(float)
            sample_data[col].fillna(self.fill_values_continuous[col], inplace=True)

        for col, encoder in self.onehot_encoders.items():
            new_data = encoder.transform(sample_data[col].to_numpy().reshape(-1, 1))
            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            sample_data = pd.concat([sample_data, new_df], axis=1).drop(columns=[col])

        for col, scaler in self.min_max_scaler_dict.items():
            if col in sample_data.columns:
                sample_data[col] = scaler.transform(sample_data[[col]])
        if 'price' in sample_data.columns:
            sample_data = sample_data.drop(columns=['price'])

        return sample_data

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        """
        Calculate and print the accuracy of the model on both the training and test data sets.

        Args:
            X_train: Features for the training set.
            X_test: Features for the test set.
            y_train: Actual labels for the training set.
            y_test: Actual labels for the test set.

        Returns:
            A tuple containing the training accuracy and the test accuracy.
        """
        y_train_pred = self.model.predict(X_train)

        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy

    def get_accuracy_full(self, X, y):
        """
        Calculate and print the overall accuracy of the model using a data set.

        Args:
            X: Features for the data set.
            y: Actual labels for the data set.

        Returns:
            The accuracy of the model on the provided data set.
        """
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        print("Accuracy: ", accuracy)

        return accuracy

    def train_and_save_model(self, df):
        y = df["price"]
        X = df.drop(columns="price")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def mean_absolute_percentage_error(y_true, y_pred):
            return tf.reduce_mean(
                tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.float32.max))) * 100

        early_stopping = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

        optimizer = Adam(learning_rate=0.0005)
        neural_model = tf.keras.Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(20, activation='relu'),
            Dense(20, activation='relu'),
            Dense(20, activation='relu'),
            Dense(20, activation='relu'),
            # Dropout(0.3),
            Dense(1, activation='linear')
        ])

        neural_model.compile(optimizer=optimizer, loss='mean_squared_error',
                             metrics=['mse', mean_absolute_percentage_error])
        history = neural_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1,
                                   callbacks=[early_stopping, checkpoint])

        self.model = keras.models.load_model('best_model.keras')

        # train_accuracy, test_accuracy = self.get_accuracy(X_train, X_test, y_train, y_test)

        mape_score = self.get_mean_absolute_percentage_error(X_test, y_test)
        return mape_score, neural_model

    def get_mean_absolute_percentage_error(self, X_test, y_test):

        y_pred_test = self.model.predict(X_test, verbose=0)

        y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
        y_pred_test = y_pred_test.flatten()

        mape_score = np.mean(np.abs((y_test - y_pred_test) / np.where(y_test == 0, 1, y_test))) * 100

        return mape_score

    @staticmethod
    def create_new_folder(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
