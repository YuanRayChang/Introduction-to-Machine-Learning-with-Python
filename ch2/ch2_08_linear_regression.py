# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:49:26 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
Ir = LinearRegression().fit(X_train, y_train)
print("training set score: {:.2f}".format(Ir.score(X_train, y_train)))
print("test set score: {:.2f}".format(Ir.score(X_test, y_test)))