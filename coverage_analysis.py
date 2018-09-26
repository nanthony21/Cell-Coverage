# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:42:05 2018

@author: Scott
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

#remove components smaller than min_size
def remove_component(img, min_size):
    #find all your connected components
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    #output_img
    img2 = np.zeros((output.shape))
    
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = img.max()
            
    return img2.astype(img.dtype)
     
# Create Mask for finding pixels in local neighborhood       
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

#n, hist_bins, patches = plt.hist(var_img2_m.flatten(),
#                                 range(var_img2_m.min(), var_img2_m.max()),
#                                 density=True)
#
#n, hist_bins, patches = plt.hist(blur_var1.flatten(), bins=400)

# Gaussian Filter to remove noise
img_blur = cv.GaussianBlur(img,(5,5),0)

# calculate Variance Map
var_img = var_map(img_blur, 1)

# Use Otsu to calculate binary threshold and binarize
bin_var_img = cv.threshold(var_img, otsu_1d(var_img), 65535, cv.THRESH_BINARY)[1]

# flip background and foreground
bin_var_img[bin_var_img == 0] = 1
bin_var_img[bin_var_img == 65535] = 0

# Set kernels for morphological operations and CC
kernel_er = np.ones((2,2),np.uint8)
kernel_dil = np.ones((4,4),np.uint8)
min_size = 50

# Erode->Remove small features->dilate
morph_img = cv.erode(bin_var_img, kernel_er)
morph_img = remove_component(morph_img, min_size)
morph_img = cv.dilate(morph_img, kernel_dil)

# display images
r = [0, 256, 512, 768, 1024]
for x_r in range(4):
    for y_r in range(4):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(morph_img[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap='gray')
