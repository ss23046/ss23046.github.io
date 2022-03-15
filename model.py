# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train():
    df = pd.read_csv(r'/Users/sanjanasrinivasarao/Downloads/SBP_NEW.csv')

    x = df[["Age", "Weight"]]
    y = df["SBP"]

    regr = LinearRegression()
    regr.fit(x, y)

    joblib.dump(regr, "/Users/sanjanasrinivasarao/Downloads/newregr.pkl")

def load():
    clf = joblib.load("newregr.pkl")
    age = 18
    weight = 60
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = clf.predict(x)[0]
    print(prediction)

if __name__ == "__main__":
    train()
    load()
