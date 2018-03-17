# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:33:20 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

mglearn.plots.plot_kmeans_algorithm()
plt.figure()
mglearn.plots.plot_kmeans_boundaries()

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("Cluster membership:\n{}".format(kmeans.labels_))

print(kmeans.predict(X))
plt.figure()
mglearn.discrete_scatter(X[:,0], X[:,1], kmeans.labels_, \
                         markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0], \
                         kmeans.cluster_centers_[:,1], \
                         [0, 1 ,2], markers='^', markeredgewidth=2)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:,0], X[:,1], assignments, ax=axes[0])

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:,0], X[:,1], assignments, ax=axes[1])








