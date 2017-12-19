import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def simplify_to_first_letter(df: pd.DataFrame, feature: str):
    df[feature] = df[df[feature].notnull()][feature].apply(lambda x: x[0])
    return df


def get_surname(text: str):
    return re.findall(r'([ \w]+),', text)[0]


def get_title(text: str):
    return re.findall(r', ([\w]+).', text)[0]


def simplify_cabin(df: pd.DataFrame):
    df = simplify_to_first_letter(df, "Cabin")
    df["Cabin"] = df["Cabin"].fillna("N")
    return df


def simplify_embarked(df: pd.DataFrame):
    df = simplify_to_first_letter(df, "Embarked")
    df["Embarked"] = df["Embarked"].fillna("N")
    return df


def simplify_name_to_surname_title(df: pd.DataFrame):
    df["Surname"] = df["Name"].apply(get_surname)
    df["Title"] = df["Name"].apply(get_title)
    df = df.drop("Name", axis=1)
    return df


def simplify_age(df: pd.DataFrame):
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    return df


def encode_features(df: pd.DataFrame, features: list):
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


def process_data(df: pd.DataFrame):
    df = simplify_cabin(df)
    df = simplify_name_to_surname_title(df)
    df = simplify_embarked(df)

    print()
    df = simplify_age(df)
    df = df.drop("Ticket", axis=1)

    features_to_encode = ["Sex", "Cabin", "Embarked", "Surname", "Title"]

    df = encode_features(df, features_to_encode)
    return df


def format_tick(value, tick_number):
    return str(int(value+1))


def plot_heatmap(results):
    heat_map = sns.heatmap(results, annot=True, cmap=sns.color_palette(), fmt='.2g')
    heat_map.xaxis.set_major_formatter(plt.FuncFormatter(format_tick))
    heat_map.yaxis.set_major_formatter(plt.FuncFormatter(format_tick))

    plt.xlabel("First layer neurons")
    plt.ylabel("Second layer neurons")
    plt.title("Accuracy for different numbers of neurons")

    plt.show()


def run_neural_network(first_layer_n, second_layer_n, X_train, y_train, X_test, y_test):
    results = np.zeros((first_layer_n, second_layer_n))

    for test in ((x, y) for x in range(10) for y in range(10)):

        model = MLPClassifier(
            solver='lbfgs',
            alpha=1e-5,
            hidden_layer_sizes=(test[0]+1, test[1]+1),
            activation='relu',
            learning_rate='adaptive',
        )

        fit = model.fit(X_train, y_train)

        results[test] = model.score(X_test, y_test)
        # print(f"Model score for {test}: {results[test]}")

    #print(f"best performance with {results.argmax(axis=1)[0]+1} neurons and {results.argmax(axis=1)[1]+1} neurons: {results.max()}")
    print(f"best performance: {results.max()}")
    return results


def main():
    data_train = pd.read_csv('data/train.csv')
    data_test = pd.read_csv('data/test.csv')

    data_train = process_data(data_train)
    data_test = process_data(data_test)

    X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']


    scaler = MinMaxScaler()
    X_all[X_all.columns] = scaler.fit_transform(X_all[X_all.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2
    )

    results = run_neural_network(10, 10, X_train, y_train, X_test, y_test)
    plot_heatmap(results)


if __name__ == "__main__":
    main()
