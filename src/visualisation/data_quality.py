import pandas as pd
import numpy as np
import os
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class DataQuality:

    @staticmethod
    def identify_missing_values(data):
        plt.figure(figsize=(15, 4))
        sns.heatmap(data.isna().transpose(),
                    cmap="YlGnBu",
                    cbar_kws={'label': 'Missing data'})
        plt.show()
        pass


class EDA:

    @staticmethod
    def correlation_matrix(data):
        plt.figure()
        corr = data.corr()
        sns.heatmap(corr, cmap="Blues", annot=True)


if __name__ == "__main__":
    train_data = pd.read_csv("./data/processed/train.csv")
    test_data = pd.read_csv("./data/processed/test.csv")
    print(train_data.dtypes)

    EDA.correlation_matrix(train_data)

    DataQuality.identify_missing_values(train_data)
    DataQuality.identify_missing_values(test_data)



