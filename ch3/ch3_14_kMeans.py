# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:18:09 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:,0], \
            kmeans.cluster_centers_[:,1], \
            s=60, marker='^', c=range(kmeans.n_clusters), \
            linewidths=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("Cluster memberships:\n{}".format(y_pred))

distance_features = kmeans.transform(X)
print("Distance feature shape: {}".format(distance_features.shape))
print("Distance features:\n{}".format(distance_features))