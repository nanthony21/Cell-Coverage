# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:42:05 2018

@author: Scott
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

def dist_mask(dist):
    output_mask = np.ones([dist*2 + 1, dist*2 + 1], dtype=np.uint16)
    output_mask[dist, dist] = 0
    dist_map = ndimage.distance_transform_edt(output_mask)
    output_mask[dist_map>dist] = 0
    output_mask[dist_map<=dist] = 1
    return output_mask

# var_map creates a map of the spatial variance 
# in a neighborhood of size dist at pixels in img
def var_map(img, dist):
    mask = dist_mask(dist)
    mask_size = dist*2 + 1
    output_map = np.zeros(img.shape, dtype=np.uint16)
    
    # loop through all pixels
    for ind_x in range(img.shape[0]):
        for ind_y in range(img.shape[1]):
            
            # calculate index offset
            x1_off = ind_x - dist if ind_x - dist < 0 else 0
            x2_off = img.shape[0] - (ind_x + dist + 1) if ind_x + dist + 1 > img.shape[0] else 0
            y1_off = ind_y - dist if ind_y - dist < 0 else 0
            y2_off = img.shape[1] - (ind_y + dist + 1) if ind_y + dist + 1 > img.shape[1] else 0
            
            # calculate index
            x1 = ind_x - dist - x1_off
            x2 = ind_x + dist + 1 + x2_off
            y1 = ind_y - dist - y1_off
            y2 = ind_y + dist + 1 + y2_off
            
            # calculate the local spatial variance
            local_img = img[x1:x2, y1:y2]
            local_mask = mask[-x1_off : mask_size + x2_off, -y1_off : mask_size + y2_off]            
            output_map[ind_x, ind_y] = (local_img[local_mask.astype(bool)]).var()

#            fig, ax = plt.subplots()
#            plt.subplot(1, 2, 1)
#            plt.imshow(test1, cmap='gray')
#            plt.subplot(1, 2, 2)
#            plt.imshow(test2, cmap='gray')
#
#            while plt.fignum_exists(fig.number):
#                fig.canvas.flush_events()
#            print(ind_y)
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
#n, hist_bins, patches = plt.hist(var_img2_m.flatten(),
#                                 range(var_img2_m.min(), var_img2_m.max()),
#                                 density=True)
#
#n, hist_bins, patches = plt.hist(var_img2_m.flatten())

# Calculate Binary mask otsu
#ret,thresh1 = cv.threshold(img, otsu_1d(img), 65535, cv.THRESH_BINARY)

# calculate Variance Map
var_img1_m = var_map(img, 1)
var_img2_m = var_map(img, 2)
var_img4 = var_map(img, 4)
var_img8 = var_map(img, 8)


#ret,thresh1 = cv.threshold(img, otsu_1d(var_img1), 65535, cv.THRESH_BINARY)
#ret,thresh2 = cv.threshold(img, otsu_1d(var_img2), 65535, cv.THRESH_BINARY)
#ret,thresh4 = cv.threshold(img, otsu_1d(var_img4), 65535, cv.THRESH_BINARY)
#ret,thresh8 = cv.threshold(img, otsu_1d(var_img8), 65535, cv.THRESH_BINARY)

# display images
#plt.subplot(2, 2, 1)
#plt.imshow(img[500:700, 100:300], cmap='gray')
#plt.subplot(2, 2, 2)
#plt.imshow(thresh2[500:700, 100:300], cmap='gray')
#plt.subplot(2, 2, 3)
#plt.imshow(thresh4[500:700, 100:300], cmap='gray')
#plt.subplot(2, 2, 4)
#plt.imshow(thresh8[500:700, 100:300], cmap='gray')
