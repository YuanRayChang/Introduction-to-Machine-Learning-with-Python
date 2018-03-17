# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:51:55 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn



from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

        
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.
X_train, X_test, y_train, y_test = train_test_split( \
        X_people, y_people, stratify=y_people, random_state=0)

mglearn.plots.plot_nmf_illustration()
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)