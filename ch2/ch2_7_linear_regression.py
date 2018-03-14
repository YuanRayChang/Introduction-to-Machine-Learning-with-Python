# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 11:57:56 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#mglearn.plots.plot_linear_regression_wave()
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
Ir = LinearRegression().fit(X_train, y_train)
print("Ir.coef_: {}".format(Ir.coef_))
print("Ir.intercept_: {}".format(Ir.intercept_))
print("Training set score: {:.2f}".format(Ir.score(X_train, y_train)))
print("Test set score: {:.2f}".format(Ir.score(X_test, y_test)))

