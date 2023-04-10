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
pd.set_option('display.max_columns', None)


class EDA:

    @staticmethod
    def correlation_matrix(data):
        plt.figure()
        corr = data.corr()
        sns.heatmap(corr, cmap="Blues", annot=True)


    @staticmethod
    def box_plot(data):
        ax = sns.boxplot(data=data, orient="h", palette="Set2")
        ax

if __name__ == "__main__":

    train_data = pd.read_csv("./data/processed/train.csv")
    test_data = pd.read_csv("./data/processed/test.csv")
    #train_data = train_data.set_index("PassengerId")
    #test_data = test_data.set_index("PassengerId")

    EDA.correlation_matrix(train_data)


    EDA.box_plot(train_data[['Age', 'SibSp', 'Parch', 'Fare']])
    EDA.box_plot(train_data[['Age']]) # filter any over 63
    EDA.box_plot(train_data[['SibSp']])
    EDA.box_plot(train_data[['Parch']])
    EDA.box_plot(train_data[['Fare']]) # filter over 70?

    #train_data = train_data[train_data["Age"] <= 63]
    #train_data = train_data[train_data["Fare"] <= 70]
    #train_data = train_data[train_data["AgeKnown"] == True]

    #train_data = train_data.reset_index()
    train_data.to_csv("./data/processed/train.csv", index=False)



