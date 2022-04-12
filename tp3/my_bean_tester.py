"""
Team:
<<<<< TEAM NAME >>>>>
Authors:
<<<<< NOM COMPLET #1 - MATRICULE #1 >>>>>
<<<<< NOM COMPLET #2 - MATRICULE #2 >>>>>
"""

BEANS = ["SIRA", "HOROZ", "DERMASON", "BARBUNYA", "CALI", "BOMBAY", "SEKER"]
BEANS_INDEXES = {BEANS[i]: i for i in range(len(BEANS))}

from bean_testers import BeanTester
import numpy as np
import tensorflow as tf
from tensorflow import keras


class MyBeanTester(BeanTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(12, activation=tf.nn.sigmoid))
        self.model.add(keras.layers.Dense(3, activation=tf.nn.sigmoid))
        self.model.add(keras.layers.Dense(len(BEANS), activation=tf.nn.sigmoid))

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def min_max_scaler(self, features):
        """
        Function: scales values betweeb 0 and 1.

        Parameters: dataset as an NumPy array.

        Returns: scaled NumPy array in float32 dtype.
        """
        max_n = np.max(features)
        min_n = np.min(features)
        features_scaled = np.array([(x - min_n) / (max_n - min_n) for x in features])
        return features_scaled.astype("float32")

    def preprocess_data(self, x_data):
        x_train = [[float(feature) for feature in features[1:]] for features in x_data]
        x_train = self.min_max_scaler(x_train)
        return x_train


    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        # TODO: entrainer un modèle sur X_train & y_train

        # mnist = keras.datasets.mnist
        # ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
        # print(x_train)

        x_train = self.preprocess_data(X_train)
        y_train = [BEANS_INDEXES[bean[1]] for bean in y_train]

        # print(y_train)
        self.model.fit(x_train, np.array(y_train), epochs=3000)

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        # TODO: make predictions on X_data and return them
        x_data = self.preprocess_data(X_data)
        predictions = self.model.predict(x_data)
        predictions = [[i + 1, BEANS[np.argmax(prediction)]] for i, prediction in enumerate(predictions)]

        return predictions
