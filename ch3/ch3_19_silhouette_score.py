# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:32:45 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), \
            subplot_kw={'xticks': (), 'yticks': ()})

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, \
                                      size=len(X))

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], \
    c=random_clusters, cmap=mglearn.cm3, s=60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(\
    silhouette_score(X_scaled, random_clusters)))

algorithms = [KMeans(n_clusters=2), \
              AgglomerativeClustering(n_clusters=2), \
              DBSCAN()]

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, \
           cmap=mglearn.cm3, s=60)
    ax.set_title("{} : {:.2f}".format(\
             algorithm.__class__.__name__, \
             silhouette_score(X_scaled, clusters)))

