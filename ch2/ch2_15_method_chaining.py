# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:56:27 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(\
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# method 1
logreg = LogisticRegression().fit(X_train, y_train)
# method 2
logreg = LogisticRegression()
y_pred = logreg.fit(X_train, y_train).predict(X_test)
# method 3
y_pred = LogisticRegression().fit(X_train, y_train).predict(X_test)