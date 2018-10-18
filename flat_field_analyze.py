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
import datetime

# Root folder for experiment
root = 'K:\\Coverage\\10-17-18\\'
# Folder names for each plate to analyze
plate_folder_list = ['0HourA2780_1', '0HourA2780_plate2']
# Folder and file names for individual well in plate
well_folder_list = ['BottomLeft_1', 'BottomMid_1', 'BottomRight_1', 'TopLeft_1', 'TopMid_1', 'TopRight_1']
# image index for center image for flatfielding
center_locations = [('003','004'), ('003','004'), ('003','004'), ('003','004'), ('003','004'), ('003','004')]
# image index for edge image for masking
edge_locations = [('000','009'), ('002','003'), ('001','009'), ('001','006'), ('002','005'), ('000','005')]
# Flat Field correction folder
ffc_folder = 'flatfield_1'
# Filename prefix
file_prefix = '_MMStack_Pos_'
# Folders and File prefix for saving analyzed images
outline_folder = 'Outline_fix'
binary_folder = 'Binary_fix'
analyzed_filename = 'analyzed'

# Number list used for grid error correction
num_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020']

dark_count = 624 # camera dark counts

# Loop through plates
for plate_folder in plate_folder_list:
    
    # Create folder for results
    analyzed_folder = root + plate_folder + '\\Analyzed\\'
    if not os.path.exists(analyzed_folder):
        os.makedirs(analyzed_folder)
    # Initialize txt file to save coverage numbers
    f= open((analyzed_folder + 'Coverage Results.txt'),"a")
    f.write(str(datetime.datetime.now()) + '\n')
    
    #loop through wells
    for well_index, well_folder in enumerate(well_folder_list):

        # Mean value of center image is used for flat field correction
        ffc_center = cv.imread(root + ffc_folder + '\\' + well_folder + '\\' + 
                               well_folder + file_prefix + center_locations[well_index][0] +
                               '_' + center_locations[well_index][1] + '.ome.tif', -1)
        ffc_center -= dark_count
        img_mean = ffc_center.mean()
        
        # FFC edge images are used to threshold the area outside the dish
        ffc_edge = cv.imread(root + ffc_folder + '\\' + well_folder + '\\' + 
                               well_folder + file_prefix + edge_locations[well_index][0] +
                               '_' + edge_locations[well_index][1] + '.ome.tif', -1)
        ffc_edge -= dark_count
        ffc_thresh = ca.otsu_1d(ffc_edge, wLowOpt = 1)    #Overriding the weight for the low distribution improved segmentation when one population is very narrow and the other is very wide
        
        # FF corrected cell edge images are used to threshold the edge effects from the dish
        cell_edge = cv.imread(root + plate_folder + '\\' + well_folder + '\\' + 
                               well_folder + file_prefix + edge_locations[well_index][0] +
                               '_' + edge_locations[well_index][1] + '.ome.tif', -1)
        cell_edge -= dark_count
        cell_edge = ((cell_edge * img_mean)/ffc_edge).astype(np.uint16)
        cell_thresh = ca.otsu_1d(cell_edge, wLowOpt = 1)
        
        # create save folder
        if not os.path.exists(analyzed_folder + '\\' + well_folder + '_' + outline_folder):
            os.makedirs(analyzed_folder + '\\' + well_folder + '_' + outline_folder)
         # create save folder
        if not os.path.exists(analyzed_folder + '\\' + well_folder + '_' + binary_folder):
            os.makedirs(analyzed_folder + '\\' + well_folder + '_' + binary_folder)   
            
        # Intialize coverage variables
        cell_area = 0
        background_area = 0
        removed_area = 0

        # loop through cell images        
        file_list = glob.glob(root + plate_folder + '\\' + well_folder + '\\' + well_folder + file_prefix + '*')
        for cell_img_loc in file_list:
            # load flat field
            ffc_img_loc = (root + ffc_folder + '\\' + well_folder + '\\' + well_folder + cell_img_loc.split(well_folder)[2])
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

            # Code to deal with snaking error 
            cell_labels = cell_img_loc.split('.')[0].split('_')
            cell_num = file_list[-1].split('.')[0].split('_')[6]
            temp_num_list = num_list[:num_list.index(cell_num)+1]
            temp_num_list.reverse()
            relabel_num = temp_num_list[num_list.index(cell_labels[6])]
            
            if int(cell_labels[7])%2 == 0:
                # write image to file
                cv.imwrite((analyzed_folder + '\\' + well_folder + '_' + binary_folder + '\\' + analyzed_filename + cell_img_loc.split(well_folder)[2]), morph_img)
                cv.imwrite((analyzed_folder + '\\' + well_folder + '_' + outline_folder + '\\' + analyzed_filename + cell_img_loc.split(well_folder)[2]), corr_img)
            else:
                # For odd rows flip image order
                cv.imwrite((analyzed_folder + well_folder + '_' + binary_folder + '\\' + analyzed_filename + file_prefix + relabel_num + '_' + cell_labels[7] + '.ome.tif') , morph_img)
                cv.imwrite((analyzed_folder + well_folder + '_' + outline_folder + '\\' + analyzed_filename + file_prefix + relabel_num + '_' + cell_labels[7] + '.ome.tif'), corr_img)
        
        # Output and save coverage numbers
        print('The coverage is ', 100*cell_area/(cell_area + background_area), ' %')
        f = open((analyzed_folder + '\\Coverage Results.txt'),"a")
        f.write(plate_folder + '  ' + well_folder + '\n')
        f.write(('The coverage is '+ str(100*cell_area/(cell_area + background_area)) + ' %') + '\n \n')
        f.close() 