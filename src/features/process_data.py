import pandas as pd
import statistics


class ProcessData:

    def main(self, data):

        data["Cabin"] = data["Cabin"].fillna("")
        data["Fare"] = data["Fare"].fillna("")

        fares = data["Fare"][data["Fare"] != ""]
        median_fare = statistics.median(fares)

        age_known = []
        cabin_known = []
        sex_male = []
        fare = []
        for inx, row in data.iterrows():
            if row["Age"] > 0:
                age_known.append(1)
            else:
                age_known.append(0)

            if row["Cabin"] == "":
                cabin_known.append(0)
            else:
                cabin_known.append(1)

            if row["Sex"] == "male":
                sex_male.append(1)
            else:
                sex_male.append(0)

            if row["Fare"] == "":
                fare.append(median_fare)
                print(row)
            else:
                fare.append(row["Fare"])

        data["AgeKnown"] = age_known
        data["CabinKnown"] = cabin_known
        data["SexMale"] = sex_male
        data["Fare"] = fare

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

    train_data2.to_csv("./data/processed/train.csv", index=False)
    test_data2.to_csv("./data/processed/test.csv", index=False)

    print(f"{len(train_data2)} {len(test_data2)}")

