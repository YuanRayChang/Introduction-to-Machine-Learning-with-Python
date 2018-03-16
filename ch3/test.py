# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:01:26 2018

@author: Yuan-Ray Chang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
x = np.array(['bb','b','c'])
counts = np.bincount(x)
print(counts)