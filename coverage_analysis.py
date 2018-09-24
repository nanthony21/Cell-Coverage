# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:42:05 2018

@author: Scott
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#from scipy import ndimage

# var_map creates a map of the spatial variance 
# in a neighborhood of size dist at pixels in img
def var_map(img, dist):
    output_map = np.zeros(img.shape, dtype=np.uint16)
    
    # loop through all pixels
    for ind_x in range(img.shape[0]):
        for ind_y in range(img.shape[1]):
            
            # correct range if for pixels near the edge
            x1 = ind_x - dist if ind_x - dist > 0 else 0
            x2 = ind_x + dist + 1 if ind_x + dist + 1 <= img.shape[0] else img.shape[0]
            y1 = ind_y - dist if ind_y - dist > 0 else 0
            y2 = ind_y + dist + 1 if ind_y + dist + 1 <= img.shape[1] else img.shape[1]

            # calculate the spatial variance
            output_map[ind_x, ind_y] = img[x1:x2, y1:y2].var()
            
    return output_map

# calculates the threshold for binarization using Otsu's method
def otsu_1d(img):

    flat_img = img.flatten()
    var_b_max = 0
    bin_index = 0
    step = 25 # If dynamic range is high, then increase step to speed code
    for bin_val in range(flat_img.min(), flat_img.max(), step):

        # segment data based on bin
        g0 = flat_img[flat_img <= bin_val]
        g1 = flat_img[flat_img > bin_val]
        
        # determine weights of each bin
        w0 = g0.size/flat_img.size
        w1 = g1.size/flat_img.size
        
        # maximize inter-class variance
        var_b = w0 * w1 * (g0.mean() - g1.mean())**2
        [var_b_max, bin_index] = [var_b, bin_val] if var_b > var_b_max else [var_b_max, bin_index]
    return bin_index

# Main Code
file = 'H:\\Cell Coverage\\cellCvgSc\\corrT_0\\corrT_1_MMStack_4-Pos_006_015.ome.tif'

img = cv.imread(file, -1)

#flat_img = img.flatten()
#n, hist_bins, patches = plt.hist(flat_img,
#                                 range(flat_img.min(), flat_img.max()),
#                                 density=True)

# Calculate Binary mask otsu
ret,thresh1 = cv.threshold(img, otsu_1d(img), 65535, cv.THRESH_BINARY)

# calculate Variance Map
var_img = var_map(img, 4)

# display images
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(var_img, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(thresh1, cmap='gray')
