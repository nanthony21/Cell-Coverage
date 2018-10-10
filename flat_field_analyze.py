# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:18:32 2018

@author: Scott
"""
import sys
import glob
import cv2 as cv
from matplotlib import pyplot as plt
import os
import numpy as np
import coverage_analysis as ca




# root directory
root = 'K:\\Coverage\\'

# cell images prefix and folder
cell_folder = 'transmission10-3-2018_1'
cell_filename = 'transmission10-3-2018_1'

# flatfield images prefix and folder
ffc_folder = 'Treference10-3-2018_2'
ffc_filename = 'Treference10-3-2018_2'

# images prefix and folder to save corrected images
corr_folder = 'corr_out_trans_10-3-2018_1'
morph_folder = 'corr_binary_trans_10-3-2018_1'
corr_filename = 'corr_trans_10-3-2018_1'

# check if filename matches foldername
split_num = 2 if cell_filename == cell_folder else 1
dark_count = 624 # camera dark counts

# Set filename for an image containing edges and one in the center
center_file = '_MMStack_3-Pos_003_015.ome.tif'
edge_file = '_MMStack_3-Pos_000_006.ome.tif'

# Mean value of center image is used for flat field correction
ffc_center = cv.imread(root + ffc_folder + '\\' + ffc_filename + center_file, -1)
ffc_center -= dark_count
img_mean = ffc_center.mean()

# FFC edge images are used to threshold the area outside the dish
ffc_edge = cv.imread(root + ffc_folder + '\\'  + ffc_filename + edge_file, -1)
ffc_edge -= dark_count
ffc_thresh = ca.otsu_1d(ffc_edge, wLow = 1)    #Overriding the weight for the low distribution improved segmentation when one population is very narrow and the other is very wide

# FF corrected cell edge images are used to threshold the edge effects from the dish
cell_edge = cv.imread(root + cell_folder + '\\'  + cell_filename + edge_file, -1)
cell_edge -= dark_count
cell_edge = ((cell_edge * img_mean)/ffc_edge).astype(np.uint16)
cell_thresh = ca.otsu_1d(cell_edge, wLow = 1)

# create save folder
if not os.path.exists(root + corr_folder):
    os.makedirs(root + corr_folder)
 # create save folder
if not os.path.exists(root + morph_folder):
    os.makedirs(root + morph_folder)   
    
# Intialize coverage variables
cell_area = 0
background_area = 0
removed_area = 0

# loop through cell images
for cell_img_loc in glob.glob(root + cell_folder + '\\' + cell_filename + '*'):
    # load flat field
    ffc_img_loc = (root + ffc_folder + '\\' + ffc_filename + cell_img_loc.split(cell_filename)[split_num])
    ffc_img = cv.imread(ffc_img_loc, -1)
    ffc_img -= dark_count
    
    # load cell
    cell_img = cv.imread(cell_img_loc, -1)
    cell_img -= dark_count

    # calculated corrected image
    corr_img = ((cell_img * img_mean)/ffc_img).astype(np.uint16)

    # Determine mask to remove dark regions and regions outside of dish
    ffc_mask = cv.threshold(ffc_img, ffc_thresh, 65535, cv.THRESH_BINARY)[1]
    corr_mask = cv.threshold(cell_img, cell_thresh, 65535, cv.THRESH_BINARY)[1]
    background_mask = ffc_mask * corr_mask
    
    # Segment out cells from background
    [outline, morph_img] = ca.analyze_img(corr_img, background_mask)
    
    # Keep track of areas to calculate coverage
    removed_area += np.count_nonzero(morph_img == 2)
    background_area += np.count_nonzero(morph_img == 1)
    cell_area += np.count_nonzero(morph_img == 0)

    # Add segmentation outline to corrected image
    corr_img[outline.astype(bool)] = 0
    
    # flip orientation for stitching
    morph_img = cv.flip(morph_img, 0)
    corr_img = cv.flip(corr_img, 0)
    
    # write image to file
    cv.imwrite((root + morph_folder + '\\' + corr_filename + cell_img_loc.split(cell_filename)[split_num]), morph_img)
    cv.imwrite((root + corr_folder + '\\' + corr_filename + cell_img_loc.split(cell_filename)[split_num]), corr_img)

print('The coverage is ', 100*cell_area/(cell_area + background_area), ' %')