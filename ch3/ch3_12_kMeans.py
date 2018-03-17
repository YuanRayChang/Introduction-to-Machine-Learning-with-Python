# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 11:33:20 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=mglearn.cm3)
plt.scatter(kmeans.cluster_centers_[:,0], \
            kmeans.cluster_centers_[:,1], marker='^', \
            c=[0, 1, 2], s=100, linewidth=2, \
            cmap=mglearn.cm3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=mglearn.cm2, s=60)
plt.scatter(kmeans.cluster_centers_[:,0], \
            kmeans.cluster_centers_[:,1], marker='^', \
            c=[mglearn.cm2(0), mglearn.cm2(1)], \
            s=100, linewidth=2)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")



