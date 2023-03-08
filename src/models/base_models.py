import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 1


class BaseModels:

    @staticmethod
    def main(train_data_x, train_data_y):

        # Base models:

        rf_pipeline = Pipeline(steps=[
            ('model', RandomForestClassifier(random_state=SEED))
        ])

        dt_pipeline = Pipeline(steps=[
            ('model', DecisionTreeClassifier(random_state=SEED))
        ])

        logr_pipeline = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('model', LogisticRegression(random_state=SEED))
        ])

        config = {
            "models": [rf_pipeline, dt_pipeline, logr_pipeline],
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


if __name__ == "__main__":

    train_data = pd.read_csv("./data/processed/train.csv")
    test_data = pd.read_csv("./data/processed/test.csv")
    train_data_x = train_data[['Pclass',  'SibSp', 'Parch', 'C', 'Q', 'S',
                                'Fare', 'AgeKnown', 'CabinKnown', 'SexMale']]
    train_data_y = train_data["Survived"]

    BaseModels.main(train_data_x, train_data_y)


