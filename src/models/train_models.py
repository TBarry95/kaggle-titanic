import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', None)

SEED = 123
CV_FOLDS = 3


class RandomSearchParamTuning:

    def __init__(self, random_parameters):
        self.params = random_parameters

    def random_forest(self, X_train, y_train, X_test, y_test):

        # ranodm search:
        model = RandomForestClassifier(random_state=SEED)

        model_random = RandomizedSearchCV(estimator=model,
                                           param_distributions=self.params,
                                           n_iter=100,
                                           cv=CV_FOLDS,
                                           verbose=2,
                                           random_state=SEED,
                                           n_jobs=-1)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model_random.fit(X_train, y_train)

        # Get best model:
        print(model_random.best_params_)
        print(model_random.best_estimator_)
        rf_best = model_random.best_estimator_

        # Prediction results:
        y_preds_best = rf_best.predict(X_test)
        print("RF best ACCURACY: ", metrics.accuracy_score(y_test, y_preds_best))
        print("RF BEST F1: ", metrics.f1_score(y_test, y_preds_best))
        print("RF BEST RECALL: ", metrics.recall_score(y_test, y_preds_best))
        print("RF BEST PRECISION: ", metrics.precision_score(y_test, y_preds_best))

        return {
            "best_model": rf_best,
            "best_params": model_random.best_params_,
            "y_preds": y_preds_best,
            "y_test": y_test
        }

    def svc(self, X_train, y_train, X_test, y_test):

        model = SVC(random_state=SEED)

        model_random = RandomizedSearchCV(estimator=model,
                                          param_distributions=self.params,
                                          n_iter=50,
                                          cv=CV_FOLDS,
                                          verbose=2,
                                          random_state=SEED,
                                          n_jobs=-1)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model_random.fit(X_train, y_train)

        print(model_random.best_params_)
        print(model_random.best_estimator_)

        best_model = model_random.best_estimator_
        y_preds_best = best_model.predict(X_test)
        print("BEST ACCURACY: ", metrics.accuracy_score(y_test, y_preds_best))
        print("BEST F1: ", metrics.f1_score(y_test, y_preds_best))
        print("BEST RECALL: ", metrics.recall_score(y_test, y_preds_best))
        print("BEST PRECISION: ", metrics.precision_score(y_test, y_preds_best))

        return {
            "best_model": best_model,
            "best_params": model_random.best_params_,
            "y_preds": y_preds_best,
            "y_test": y_test
        }


class GridSearchParamTuning:

    @staticmethod
    def random_forest(X_train, y_train, X_test, y_test):

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

        return {
            "best_model": rf_best_g,
            "y_preds": y_preds_best_g,
            "y_test": y_test
        }


class TrainModels:

    def __init__(self, best_params):
        self.params = best_params

    @staticmethod
    def random_forest(X_train, y_train, X_test, y_test, **kwargs):
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
    train_data = train_data.set_index("PassengerId")
    test_data = test_data.set_index("PassengerId")

    train_data.head(3)
    train_data_x = train_data[['Pclass',  'SibSp', 'Parch', 'C', 'Q', 'S',
                                'Fare', 'CabinKnown', 'SexMale',
                               'Deck', 'MarriedWoman']]

    test_data = test_data[['Pclass',  'SibSp', 'Parch', 'C', 'Q', 'S',
                                'Fare', 'CabinKnown', 'SexMale',
                               'Deck', 'MarriedWoman']]
    train_data_y = train_data["Survived"]

    oversample = SMOTE()
    train_data_x_sm, train_data_y_sm = oversample.fit_resample(train_data_x, train_data_y)

    X_train, X_test, y_train, y_test = train_test_split(train_data_x_sm,
                                                        train_data_y_sm,
                                                        test_size=0.33,
                                                        random_state=42)
    print(f"{len(X_train)} {len(X_test)}")

    # random CV
    random_parameters_rf = {'n_estimators': [int(i) for i in np.linspace(start=5, stop=3000, num=20)],
                            'max_features': ['auto', 'sqrt'],
                            'max_depth': [int(i) for i in np.linspace(10, 300, num=10)],
                            'min_samples_split': [2, 5, 10, 20],
                            'min_samples_leaf': [1, 2, 4, 10],
                            'bootstrap': [True, False],
                            'criterion': ['entropy', "gini", "log_loss"]
                            }
    rf_rand = RandomSearchParamTuning(random_parameters_rf)
    rf_random = rf_rand.random_forest(X_train=X_train,
                                      X_test=X_test,
                                      y_train=y_train,
                                      y_test=y_test)
    print(rf_random)
    best_params = rf_random["best_params"]

    """# SVC
    random_parameters_sv = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear'],
        'degree': [0, 1, 2, 3, 4, 5, 6]
    }

    svc_rand = RandomSearchParamTuning(random_parameters_sv)
    svc_random = svc_rand.svc(X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test)

    print(svc_random)"""

    # TRAIN:
    rf_clf = RandomForestClassifier(**best_params)
    #rf_clf = rf_clf.fit(train_data_x, train_data_y)
    rf_clf = rf_clf.fit(train_data_x_sm, train_data_y_sm)

    # PREDICT:
    y_pred = rf_clf.predict(test_data)

    # Export V1:
    df = pd.DataFrame()
    df["PassengerId"] = test_data.index
    df["Survived"] = y_pred
    df.to_csv("./data/result/tbarry-titanic-preds-9apr23-rf.csv", index=False)


