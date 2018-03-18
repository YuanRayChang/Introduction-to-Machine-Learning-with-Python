# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:34:05 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

mglearn.plots.plot_agglomerative_algorithm()

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

plt.figure()
mglearn.discrete_scatter(X[:,0], X[:,1], assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"])

plt.figure()
mglearn.plots.plot_agglomerative()

plt.figure()
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(n_samples=12, random_state=0)
linkage_array = ward(X)
dendrogram(linkage_array)


ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', \
        fontdict={'size':15})
ax.text(bounds[1], 4, ' three clusters', va='center', \
        fontdict={'size':15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")





