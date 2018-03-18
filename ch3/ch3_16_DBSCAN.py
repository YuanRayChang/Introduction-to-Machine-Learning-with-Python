# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:39:41 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=12, random_state=0)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters))
                             
mglearn.plots.plot_dbscan()


