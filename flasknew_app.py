#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:48:09 2022

@author: sanjanasrinivasarao
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib

app=Flask(__name__)

@app.route("/")
def bonus():
    return render_template('bonus.html')

@app.route("/predict", methods = ["POST"])
def predict():
    model = joblib.load(open("newregr.pkl","rb"))
    age = request.form['age']
    weight = request.form['weight']
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = model.predict(x)
    return render_template('bonus.html', prediction_text = prediction[0])
    