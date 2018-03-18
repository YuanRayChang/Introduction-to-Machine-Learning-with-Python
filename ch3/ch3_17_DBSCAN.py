# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:50:17 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
dbscan = DBSCAN() # default eps=0.5
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, \
            cmap = mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")