# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:42:05 2018

@author: Scott
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage

def remove_component(img, min_size):
    '''remove connected components smaller than min_size'''
    #find all your connected components
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    
    #connectedComponentswithStats yields every separated component with information on each of them, such as size
    sizes = stats[:, -1]

    #output_img
    img2 = np.zeros((output.shape))
    
    #for every component in the image, you keep it only if it's above min_size. We start at 1 because 0 is the backgroud which we don't care about.
    for i in range(1, nb_components):
        if sizes[i] >= min_size:
            img2[output == i] = 1
            
    return img2.astype(img.dtype)
     
    
def dist_mask(dist):
    ''' Create a circular mask with a radius of `dist`.''' 
    # Initialize output make
    output_mask = np.ones([dist*2 + 1, dist*2 + 1], dtype=np.uint16)
    
    #Create distance map
    output_mask[dist, dist] = 0
    dist_map = ndimage.distance_transform_edt(output_mask)
    
    # Turn distance map into binary mask
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
#
#            while plt.fignum_exists(fig.number):
#                fig.canvas.flush_events()
#            print(ind_y)
    return output_map

def alt_var_map(img, dist):
    mask = dist_mask(dist)
    mask /= mask.sum() #Normalize the mask
#    output_map = np.zeros(img.shape, dtype = np.float32)
    mean = cv.filter2D(img, cv.CV_32F, mask)
    sqrMean = cv.filter2D(img*img, cv.CV_32F, mask)
    return (sqrMean - mean*mean).astype(img.dtype)

def otsu_1d(img, wLow = None, wHigh = None):
    ''' calculates the threshold for binarization using Otsu's method.
    The weights for the low and high distribution can be overridden using the optional arguments.
    '''
    flat_img = img.flatten()
    var_b_max = 0
    bin_index = 0
    step = 25 # If dynamic range is high, then increase step to speed code
    for bin_val in range(flat_img.min(), flat_img.max(), step):
        # segment data based on bin
        gLow = flat_img[flat_img <= bin_val]
        gHigh = flat_img[flat_img > bin_val]
        
        # determine weights of each bin
        wLow = gLow.size/flat_img.size if (wLow is None) else wLow
        wHigh = gHigh.size/flat_img.size if (wHigh is None) else wLow
        
        # maximize inter-class variance
        var_b = wLow * wHigh * (gLow.mean() - gHigh.mean())**2
        [var_b_max, bin_index] = [var_b, bin_val] if var_b > var_b_max else [var_b_max, bin_index]
    return bin_index


# Segment out cells from background
def analyze_img(img, *mask):
    if len(mask) == 0:
        mask = np.ones(img.shape).astype(np.uint16)
    else:
        mask = mask[0]

    # calculate Variance Map
    var_img = var_map(img, 1)

    # Use Otsu to calculate binary threshold and binarize
    bin_var_img = cv.threshold(var_img, otsu_1d(var_img), 65535, cv.THRESH_BINARY)[1]
    del var_img

    # flip background and foreground
    bin_var_img[bin_var_img == 0] = 1
    bin_var_img[bin_var_img == 65535] = 0
    bin_var_img[~mask.astype(bool)] = 2

    # Set kernels for morphological operations and CC
    kernel_er = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    min_size = 60

    # Erode->Remove small features->dilate
    morph_img = cv.erode(bin_var_img, kernel_er)
    morph_img = remove_component(morph_img, min_size)
    morph_img[~mask.astype(bool)] = 2
    morph_img = cv.dilate(morph_img, kernel_dil)
    del bin_var_img
    #num_pix = morph_img.shape[0]*morph_img.shape[1]
    #(num_pix - morph_img.sum())/num_pix

    #binary outline for overlay
    outline = cv.dilate(cv.Canny(morph_img.astype(np.uint8), 0, 1), cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
    img[outline.astype(bool)] = 0
    
    return [outline, morph_img]

    # Main Code
if __name__ == '__main__':
#    file = 'K:\\Coverage\\corr_trans_10-3-2018_2\\corr_trans_10-3-2018_2_MMStack_3-Pos_005_018.ome.tif'
    file = 'H:\\Cell Coverage\\cellCvgSc\\corrT_0\\corrT_1_MMStack_4-Pos_009_016.ome.tif'

    img = cv.imread(file, -1)

    [outline, morph_img] = analyze_img(img)

#    my_cmap = cm.Purples
#    my_cmap.set_under('k', alpha=0)

    #n, hist_bins, patches = plt.hist(var_img2_m.flatten(),
    #                                 range(var_img2_m.min(), var_img2_m.max()),
    #                                 density=True)
    #
    #n, hist_bins, patches = plt.hist(blur_var1.flatten(), bins=400)
#    plt.figure()
#    plt.imshow(img, cmap='gray')
    
    
    # display images
#    r = [0, 256, 512, 768, 1024]
#    for x_r in range(4):
#        for y_r in range(4):
#            fig = plt.figure()
#            plt.subplot(1, 2, 1)
#            plt.imshow(img[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap='gray')
#            plt.subplot(1, 2, 2)
#            plt.imshow(img[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap='gray')
#            plt.imshow(outline[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap=my_cmap, clim=[0.9, 1])
#            while plt.fignum_exists(fig.number):
#                fig.canvas.flush_events()
