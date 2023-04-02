import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)


SEED = 1
CV_FOLDS = 5


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

        svm_pipeline = Pipeline(steps=[
            ('scale', StandardScaler()),
            ('model', SVC())
        ])

        config = {
            "models": [rf_pipeline, dt_pipeline, logr_pipeline, svm_pipeline],
            "metrics": ["accuracy", "precision", "recall", "f1"]
        }

        model_name_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        fit_time_list = []

        for model in config["models"]:
            res = cross_validate(model,
                                 train_data_x,
                                 train_data_y,
                                 cv=CV_FOLDS,
                                 scoring=("accuracy", "precision", "recall", "f1"))

            model_name = model.get_params()["steps"]
            model_name_list.append(model_name)
            accuracy_list.append(round(res["test_accuracy"].mean(), CV_FOLDS))
            precision_list.append(round(res["test_precision"].mean(), CV_FOLDS))
            recall_list.append(round(res["test_recall"].mean(), CV_FOLDS))
            f1_list.append(round(res["test_f1"].mean(), CV_FOLDS))
            fit_time_list.append(round(res["fit_time"].mean(), CV_FOLDS))

            print(f"{model_name}: / "
                  f"accuracy: {round(res['test_accuracy'].mean(), CV_FOLDS)} / "
                  f"precision: {round(res['test_precision'].mean(), CV_FOLDS)} / "
                  f"recall: {round(res['test_recall'].mean(), CV_FOLDS)} / "
                  f"f1: {round(res['test_f1'].mean(), CV_FOLDS)} / "
                  f"fit_time: {round(res['fit_time'].mean(), CV_FOLDS)} / "
                  )

        df_res = pd.DataFrame()
        df_res["model"] = model_name_list
        df_res["accuracy"] = accuracy_list
        df_res["precision"] = precision_list
        df_res["recall"] = recall_list
        df_res["f1"] = f1_list
        df_res["fit_time"] = fit_time_list
        df_res["cv_folds"] = CV_FOLDS
        df_res["total_time"] = df_res["cv_folds"] * df_res["fit_time"]
        print(df_res)
        return df_res


if __name__ == "__main__":

    train_data = pd.read_csv("./data/processed/train.csv")
    test_data = pd.read_csv("./data/processed/test.csv")
    train_data_x = train_data[['Pclass',  'SibSp', 'Parch', 'C', 'Q', 'S',
                                'Fare', 'AgeKnown', 'CabinKnown', 'SexMale']]
    train_data_y = train_data["Survived"]

    df_res = BaseModels.main(train_data_x, train_data_y)
    df_res
