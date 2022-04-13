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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier


class MyBeanTester(BeanTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.model = OneVsOneClassifier(SVC(kernel="rbf", gamma=0.19, C=2.8))

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
        scaler = StandardScaler()
        x_train = np.array(
            [[float(feature) for feature in features[1:]] for features in x_data]
        )
        x_train = scaler.fit_transform(x_train.astype(np.float64))

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
        x_train = self.preprocess_data(X_train)
        y_train = np.array(y_train)[:, -1]

        self.model.fit(x_train, y_train)
        # y_train_score = cross_val_score(
        #     self.clf, x_train, y_train, cv=3, scoring="f1_micro"
        # )

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
        y_pred = self.model.predict(x_data)
        y_pred = [[i + 1, pred] for i, pred in enumerate(y_pred)]
        return y_pred
