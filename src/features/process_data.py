import pandas as pd
import statistics
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', None)


class ProcessData:

    def age_analysis(self, data):

        train_data_a = data[~data["Age"].isna()]
        age_median = statistics.median(train_data_a["Age"])

        train_data_na = data[data["Age"].isna()]


    def main(self, data):

        decks = {"A": 1,
                 "B": 2,
                 "C": 3,
                 "D": 4,
                 "E": 5,
                 "F": 6,
                 "G": 7,
                 "T": 0,
                 "": 6} # assume low

        train_data_a = data[~data["Age"].isna()]
        age_median = statistics.median(train_data_a["Age"])

        data["Cabin"] = data["Cabin"].fillna("")
        data["Fare"] = data["Fare"].fillna("")
        data["Cabin"] = data["Cabin"].fillna("")

        data["Deck"] = [i[0] if i != "" else "" for i in data["Cabin"]]
        data["Deck"] = [decks[i] for i in data["Deck"]]

        fares = data["Fare"][data["Fare"] != ""]
        median_fare = statistics.median(fares)

        age_known = []
        cabin_known = []
        sex_male = []
        fare = []
        age_catgs = []
        married_woman = []
        def_has_kids = []
        for inx, row in data.iterrows():
            if row["Age"] > 0:
                age_known.append(row["Age"])
            else:
                age_known.append(age_median)

            if row["Cabin"] == "":
                cabin_known.append(0)
            else:
                cabin_known.append(1)

            if row["Sex"] == "male":
                sex_male.append(1)
            else:
                sex_male.append(0)

            if "mrs." in row["Name"].lower() and row["Sex"] == "female":
                married_woman.append(1)
            else:
                married_woman.append(0)

            if row["Fare"] == "":
                fare.append(median_fare)
                print(row)
            else:
                fare.append(row["Fare"])

            if row["Parch"] > 2:
                def_has_kids.append(1)
            else:
                def_has_kids.append(0)

        data["AgeKnown"] = age_known
        data["CabinKnown"] = cabin_known
        data["SexMale"] = sex_male
        data["Fare"] = fare
        data["MarriedWoman"] = married_woman
        #data["DefHasKids"] = def_has_kids

        emb_one_hot = pd.get_dummies(data['Embarked'])
        data = pd.merge(data, emb_one_hot, left_index=True, right_index=True)
        data = data.set_index("PassengerId")

        return data



if __name__ == "__main__":
    train_data = pd.read_csv("./data/raw/train.csv")
    test_data = pd.read_csv("./data/raw/test.csv")
    train_data.head(3)

    data_proc = ProcessData()
    train_data2 = data_proc.main(train_data)
    test_data2 = data_proc.main(test_data)

    train_data2.to_csv("./data/processed/train.csv", index=True)
    test_data2.to_csv("./data/processed/test.csv", index=True)

    train_data2_gb = train_data2.groupby("Survived").agg({
        "Survived": "count"
    })
    train_data2_gb

    print(f"{len(train_data2)} {len(test_data2)}")

