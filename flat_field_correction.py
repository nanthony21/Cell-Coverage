# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:18:32 2018

@author: Scott
"""

from glob import glob
import cv2 as cv
from matplotlib import pyplot as plt
import os
import os.path as osp
import numpy as np

# root directory
root = 'K:\\Coverage\\'

# cell images prefix and folder
cell_folder = 'transmission10-2-2018_2'
cell_filename = 'transmission10-2-2018_2'

# flatfield images prefix and folder
ffc_folder = 'Treference10-3-2018_2'
ffc_filename = 'Treference10-3-2018_2'
ffc_center_file = 'Treference10-3-2018_2_MMStack_3-Pos_003_015.ome.tif'
ffc_edge_file = 'Treference10-3-2018_2_MMStack_3-Pos_000_006.ome.tif'

# images prefix and folder to save corrected images
corr_folder = 'corr_trans_10-2-2018_2'
corr_filename = 'corr_trans_10-2-2018_2'

# check if filename matches foldername
split_num = 2 if cell_filename == cell_folder else 1
dark_count = 624 # camera dark counts

# Mean value of center image is used for flat field correction
ffc_center = cv.imread(osp.join(root, ffc_folder, ffc_center_file), -1)
ffc_center -= dark_count
img_mean = ffc_center.mean()

# create save folder
if not osp.exists(osp.join(root, corr_folder)):
    os.makedirs(osp.join(root, corr_folder))

# loop through cell images
for cell_img_loc in glob(osp.join(root, cell_folder, cell_filename + '*')):
    # load flat field
    ffc_img_loc = (osp.join(root, ffc_folder, ffc_filename, cell_img_loc.split(cell_filename)[split_num]))
    ffc_img = cv.imread(ffc_img_loc, -1)
    ffc_img -= dark_count
    
    # load cell
    cell_img = cv.imread(cell_img_loc, -1)
    cell_img -= dark_count

    # calculated corrected image
#    corr_img = ((cell_img * ffc_img.mean())/ffc_img).astype(np.uint16)
    corr_img = ((cell_img * img_mean)/ffc_img).astype(np.uint16)
    
    # flip orientation for stitching
    corr_img = cv.flip(corr_img, 0)
    
    # write image to file
    cv.imwrite(osp.join(root, corr_folder, corr_filename, cell_img_loc.split(cell_filename)[split_num]), corr_img)