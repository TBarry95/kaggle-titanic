import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score


class BaseModels:

    @staticmethod
    def train_random_forest(X_train, y_train, X_test, y_test, **kwargs):
        clf = RandomForestClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("RF ACCURACY: ", metrics.accuracy_score(y_test, y_pred))
        print("RF F1: ", metrics.f1_score(y_test, y_pred))
        print("RF RECALL: ", metrics.recall_score(y_test, y_pred))
        print("RF PRECISION: ", metrics.precision_score(y_test, y_pred))
        return y_pred

    @staticmethod
    def train_logistic_regression(X_train, y_train, X_test, y_test, **kwargs):
        # need to scale the data

        clf = LogisticRegression()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    @staticmethod
    def train_decision_tree(X_train, y_train, X_test, y_test, **kwargs):
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("DT ACCURACY: ", metrics.accuracy_score(y_test, y_pred))
        print("DT F1: ", metrics.f1_score(y_test, y_pred))
        print("DT RECALL: ", metrics.recall_score(y_test, y_pred))
        print("DT PRECISION: ", metrics.precision_score(y_test, y_pred))
        return y_pred


if __name__ == "__main__":

    train_data = pd.read_csv("./data/processed/train.csv")
    test_data = pd.read_csv("./data/processed/test.csv")

    train_data_x = train_data[['Pclass',  'SibSp', 'Parch', 'C', 'Q', 'S',
                                'Fare', 'AgeKnown', 'CabinKnown', 'SexMale']]
    train_data_y = train_data["Survived"]

    # Base models:
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    config = {
        "models": [rf, dt],
        "metrics": ["accuracy", "precision", "recall", "f1"]
    }

    for model in config["models"]:
        for metric in config["metrics"]:
            res = cross_val_score(model,
                                  train_data_x,
                                  train_data_y,
                                  cv=5,
                                  scoring=metric)
            print(f"{model} - {metric}: {res.mean()}")



