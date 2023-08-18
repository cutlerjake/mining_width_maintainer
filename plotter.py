# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:19:57 2023

@author: 2jake
"""

import numpy as np
import matplotlib.pyplot as plt

def triangle(left, right, step = 1):
    inds = []
    i = 0
    while left < right:
        for j in range(left, right):
            inds.append([i,j])
        left += step
        right -= step
            
        i += 1
    return inds

sched = np.zeros((10,30))

bounds = [[10,20],
          [11, 19],
          [12, 18],
          [13, 17],
          [14, 16]]
periods = [5,
           4,
           3,
           2,
           1]

for (left, right), period in zip(bounds, periods):
    for ind in triangle(left, right):
        sched[ind[0], ind[1]] = period
    
    
fig, ax = plt.subplots()

ax.imshow(sched)


sched = np.zeros((10,30))

bounds = [[1,29],
          [4, 26],
          [7, 23],
          [10, 20],
          [13, 17]]
periods = [5,
           4,
           3,
           2,
           1]

for (left, right), period in zip(bounds, periods):
    for ind in triangle(left, right, 3):
        sched[ind[0], ind[1]] = period
    
    
fig, ax = plt.subplots()

ax.imshow(sched)

sched = np.zeros((10,30))

inds_periods = [[0,0,1],
                [0,1,1],
                [0,2,1],
                [0,3,2],
                [0,4,2],
                [0,5,2],
                [1,1,3],
                [1,2,3],
                [1,3,3],
                [0,6,3],
                [0,7,3],
                [0,8,3],
                [1,4,4],
                [1,5,4],
                [1,6,4],
                [2,2,4],
                [2,3,4],
                [2,4,4],
                [0,9,5],
                [0,10,5],
                [0,11,5],
                [1,7,5],
                [1,8,5],
                [1,9,5],
                [2,5,5],
                [2,6,5],
                [2,7,5]]

for i, j, period in inds_periods:
    sched[i,j + 9] = period

sched = np.ma.masked_where((sched == 0), sched)

fig, ax = plt.subplots()

ax.imshow(sched, cmap = 'cividis')

