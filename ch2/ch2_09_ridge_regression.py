# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:55:20 2018

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

from sklearn.linear_model import Ridge
#alpha = 1
ridge = Ridge().fit(X_train, y_train)
print("training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("test set score: {:.2f}".format(ridge.score(X_test, y_test)))

#alpha = 10
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

#alpha = 0.1
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(Ir.coef_, 'o', label="LinearRegression") #alpha = 0
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(Ir.coef_))
plt.ylim(-25, 25)
plt.legend()


#mglearn.plots.plot_ridge_n_samples()




