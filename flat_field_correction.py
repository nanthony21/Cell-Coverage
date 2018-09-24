# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:18:32 2018

@author: Scott
"""
import glob
import cv2 as cv
#from matplotlib import pyplot as plt
import os
import numpy as np

# root directory
root = 'H:\\Cell Coverage\\cellCvgSc\\'

# cell images prefix and folder
cell_folder = 'cellR_1'
cell_filename = 'cellR_1'

# flatfield images prefix and folder
ffc_folder = 'noneR_1'
ffc_filename = 'noneR_1'

# images prefix and folder to save corrected images
corr_folder = 'corrR_0'
corr_filename = 'corrT_1'

split_num = 2 if cell_filename == cell_folder else 1
dark_count = 624 # camera dark counts

# create save folder
if not os.path.exists(root + corr_folder):
    os.makedirs(root + corr_folder)

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
    corr_img = ((cell_img * ffc_img.mean())/ffc_img).astype(np.uint16)
    # flip orientation for stitching
    corr_img = cv.flip(corr_img, 0)
    # write image to file
    cv.imwrite((root + corr_folder + '\\' + corr_filename + cell_img_loc.split(cell_filename)[split_num]), corr_img)
