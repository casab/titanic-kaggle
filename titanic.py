import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, RFE
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier


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


def main():
    data_train = pd.read_csv('data/train.csv')
    data_test = pd.read_csv('data/test.csv')

    data_train = process_data(data_train)
    data_test = process_data(data_test)

    X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']

    #mdl = ExtraTreesClassifier()
    #mdl.fit(X_all, y_all)
    #print(mdl.feature_importances_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2
    )

    model = LogisticRegression()

    rfecv = RFECV(estimator=model,
                  step=1,
                  cv=StratifiedKFold(2),
                  scoring='accuracy')

    rfecv = RFE(model)

    fit = rfecv.fit(X_train, y_train)
    X_train_new = rfecv.transform(X_train)
    X_test_new = rfecv.transform(X_test)

    model.fit(X_train_new, y_train)


    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)

    print(rfecv.support_)
    print(rfecv.ranking_)

    print("Model score: ", end="")
    print(model.score(X_test_new, y_test))


if __name__ == "__main__":
    main()
