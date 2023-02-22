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


class ParameterSelection:

    @staticmethod
    def random_search_random_forest(X_train, y_train, X_test, y_test):

        random_state = 123

        # hyper params:
        rf_criteria = ['entropy', "gini", "log_loss"]
        rf_n_estimators = [int(i) for i in np.linspace(start = 20, stop = 2000, num = 10)]
        rf_max_depths = [int(i) for i in np.linspace(10, 100, num = 10)]
        rf_max_features = ['auto', 'sqrt']
        rf_min_samples_split = [2, 5, 10]
        rf_min_samples_leaf = [1, 2, 4]
        rf_bootstrap = [True, False]

        # ranodm search:
        random_parameters = {'n_estimators': rf_n_estimators,
                             'max_features': rf_max_features,
                             'max_depth': rf_max_depths,
                             'min_samples_split': rf_min_samples_split,
                             'min_samples_leaf': rf_min_samples_leaf,
                             'bootstrap': rf_bootstrap,
                             'criterion': rf_criteria}
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf,
                                       param_distributions=random_parameters,
                                       n_iter=100,
                                       cv=4,
                                       verbose=2,
                                       random_state=random_state,
                                       n_jobs=-1)
        rf_random.fit(X_train, y_train)

        # Get best model:
        print(rf_random.best_params_)
        print(rf_random.best_estimator_)
        rf_best = rf_random.best_estimator_

        # Prediction results:
        y_preds_best = rf_best.predict(X_test)
        print("RF best ACCURACY: ", metrics.accuracy_score(y_test, y_preds_best))
        print("RF BEST F1: ", metrics.f1_score(y_test, y_preds_best))
        print("RF BEST RECALL: ", metrics.recall_score(y_test, y_preds_best))
        print("RF BEST PRECISION: ", metrics.precision_score(y_test, y_preds_best))

        # Compare to a standard model:
        rf_base = RandomForestClassifier(random_state=random_state)
        rf_base.fit(X_train, y_train)
        y_preds_b = rf_base.predict(X_test)
        print("RF BASE ACCURACY: ", metrics.accuracy_score(y_test, y_preds_b))
        print("RF BASE F1: ", metrics.f1_score(y_test, y_preds_b))
        print("RF BASE RECALL: ", metrics.recall_score(y_test, y_preds_b))
        print("RF BASE PRECISION: ", metrics.precision_score(y_test, y_preds_b))

    @staticmethod
    def grid_search_random_forest(X_train, y_train, X_test, y_test):

        # grid search:
        grid_search_params = {'n_estimators': [1000, 1400, 1800, 2500, 3000, 3600],
                             'max_features': ["sqrt"],
                             'max_depth': [20, 30, 40, 60, 80],
                             'min_samples_split': [1, 2, 4, 6, 8],
                             'min_samples_leaf': [2, 4, 6, 8],
                             'bootstrap': [True],
                             'criterion': ['log_loss']}
        rf = RandomForestClassifier()
        rf_grid = GridSearchCV(estimator = rf,
                                 param_grid = grid_search_params,
                                 cv = 3,
                                 n_jobs = -1,
                                 verbose = 2)
        rf_grid.fit(X_train, y_train)

        print(rf_grid.best_params_)
        print(rf_grid.best_estimator_)

        rf_best_g = rf_grid.best_estimator_
        y_preds_best_g = rf_best_g.predict(X_test)
        print("BEST ACCURACY: ", metrics.accuracy_score(y_test, y_preds_best_g))
        print("BEST F1: ", metrics.f1_score(y_test, y_preds_best_g))
        print("BEST RECALL: ", metrics.recall_score(y_test, y_preds_best_g))
        print("BEST PRECISION: ", metrics.precision_score(y_test, y_preds_best_g))

    pass


class TrainModels:

    @staticmethod
    def train_random_forest(X_train, y_train, X_test, y_test, **kwargs):
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("ACCURACY: ", metrics.accuracy_score(y_test, y_pred))
        print("F1: ", metrics.f1_score(y_test, y_pred))
        print("RECALL: ", metrics.recall_score(y_test, y_pred))
        print("PRECISION: ", metrics.precision_score(y_test, y_pred))
        return y_pred


    def train_logistic_regression(self):
        pass


    def train_decision_tree(self):
        pass


if __name__ == "__main__":

    train_data = pd.read_csv("./data/processed/train.csv")
    test_data = pd.read_csv("./data/processed/test.csv")
    train_data.head(3)
    train_data_x = train_data[['Pclass',  'SibSp', 'Parch', 'C', 'Q', 'S',
                                'Fare', 'AgeKnown', 'CabinKnown', 'SexMale']]
    train_data_y = train_data["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(train_data_x,
                                                        train_data_y,
                                                        test_size=0.33,
                                                        random_state=42)

    print(f"{len(X_train)} {len(X_test)}")

    # Train models:
    y_rf = TrainModels.train_random_forest(X_train, y_train, X_test, y_test)
    y_rf




