# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:37:55 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

display(mglearn.plots.plot_logistic_regression_graph())

display(mglearn.plots.plot_single_hidden_layer_graph())

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), label='relu')
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")

display(mglearn.plots.plot_two_hidden_layer_graph())