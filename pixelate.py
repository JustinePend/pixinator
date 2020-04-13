#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:32:59 2020

@author: Justine
"""

import cv2
import numpy as np


img = cv2.imread('PHIDIAS.png', 1)
small = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5 , interpolation = cv2.INTER_LINEAR)
and_back_again = cv2.resize(small, (0,0), fx = 2, fy = 2 , interpolation = cv2.INTER_NEAREST)
Z = and_back_again.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1)
K = 10
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#cv2.imshow('res2',res2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

small2 = cv2.resize(res2, (0,0), fx = 0.143, fy = 0.143 , interpolation = cv2.INTER_LINEAR)
and_back_again2 = cv2.resize(small2, (0,0), fx = 7, fy = 7 , interpolation = cv2.INTER_NEAREST)

cv2.imshow('forest',and_back_again2)

cv2.waitKey(0)
cv2.destroyAllWindows()
