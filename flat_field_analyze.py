# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:18:32 2018

@author: Scott
"""
import sys
from glob import glob
import cv2 as cv
from matplotlib import pyplot as plt
import os
import os.path as osp
import numpy as np
import coverage_analysis as ca
import datetime
import imageJStitching
import csv

'''********User Inputs!!*******'''

## Root folder for experiment
root = r'K:\Coverage\10-23-18_Jane\10-25-18'
# Folder names for each plate to analyze
plate_folder_list = ['A2780_48Hour_Plate1','A2780_48Hour_Plate2','A2780_48Hour_Plate3']
# Folder and file names for individual well in plate
well_folder_list = ['BottomLeft_1', 'BottomMid_1', 'BottomRight_1', 'TopLeft_1', 'TopMid_1', 'TopRight_1']
# image index for center image for flatfielding
center_locations = [('000','006'), ('002','005'), ('002','005'), ('002','005'), ('002','005'), ('002','005')]
# image index for edge image for masking
edge_locations = [('001','001'), ('001','008'), ('001','008'), ('001','001'), ('001','008'), ('001','008')]
#A number to be added as asuffix to the output files
analysisNum:int = 3
dark_count = 624 # camera dark counts
imageJPath = r'"C:\Program Files (x86)\Fiji.app\ImageJ-win64.exe"'
# Flat Field correction folder
ffc_folder = 'FlatField'

'''**********************'''


# Filename prefix
file_prefix = '_MMStack_1-Pos'
# Folders and File prefix for saving analyzed images
outline_folder = 'Outline'
binary_folder = 'Binary'
ff_corr_folder = 'Corrected'
analyzed_filename = 'analyzed'

# Number list used for grid error correction
num_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020']

binaryProcess = None
outlineProcess = None


  # Loop through plates
for plate_folder in plate_folder_list:
    print(plate_folder)
    # Create folder for results
    analyzed_folder = osp.join(root, plate_folder, 'Analyzed_{}'.format(analysisNum))
    if not osp.exists(analyzed_folder):
        os.makedirs(analyzed_folder)
    
    results = {}
    #loop through wells
    for well_index, well_folder in enumerate(well_folder_list):
        print('\t'+well_folder)
        # Mean value of center image is used for flat field correction
        ffc_center = cv.imread(osp.join(root, ffc_folder, well_folder, 
                               well_folder + file_prefix + center_locations[well_index][0] +
                               '_' + center_locations[well_index][1] + '.ome.tif'), -1)
        ffc_center -= dark_count
        ffc_mean = ffc_center.mean()
        ffc_std = ffc_center.std()
        
        # FFC edge images are used to threshold the area outside the dish
        ffc_edge = cv.imread(osp.join(root, ffc_folder, well_folder,
                               well_folder + file_prefix + edge_locations[well_index][0] +
                               '_' + edge_locations[well_index][1] + '.ome.tif'), -1)
        ffc_edge -= dark_count
        ffc_thresh = ca.otsu_1d(ffc_edge, wLowOpt = 1)    #Overriding the weight for the low distribution improved segmentation when one population is very narrow and the other is very wide
        
        # FF corrected cell edge images are used to threshold the edge effects from the dish
        cell_edge = cv.imread(osp.join(root, plate_folder, well_folder, 
                               well_folder + file_prefix + edge_locations[well_index][0] +
                               '_' + edge_locations[well_index][1] + '.ome.tif'), -1)
        cell_edge -= dark_count
        cell_edge = ((cell_edge * ffc_mean)/ffc_edge).astype(np.uint16)
        cell_thresh = ca.otsu_1d(cell_edge, wLowOpt = 1)
        
        # create save folder
        if not osp.exists(osp.join(analyzed_folder, well_folder + '_' + outline_folder)):
            os.makedirs(osp.join(analyzed_folder, well_folder + '_' + outline_folder))
         # create save folder
        if not osp.exists(osp.join(analyzed_folder, well_folder + '_' + binary_folder)):
            os.makedirs(osp.join(analyzed_folder, well_folder + '_' + binary_folder))  
        if not osp.exists(osp.join(analyzed_folder, well_folder + '_' + ff_corr_folder)):
            os.makedirs(osp.join(analyzed_folder, well_folder + '_' + ff_corr_folder)) 
            
        # Intialize coverage variables
        cell_area = 0
        background_area = 0
        removed_area = 0

        # loop through cell images        
        file_list = glob(osp.join(root, plate_folder, well_folder, well_folder + file_prefix + '*'))
        tileSize = [0,0]
        for ind in range(2):
            tileSize[ind] = max([int(i.split('Pos')[-1].split('.')[0].split('_')[ind]) for i in file_list]) + 1
        
        for cell_img_loc in file_list:
            # load flat field
            ffc_img_loc = osp.join(root, ffc_folder, well_folder, well_folder + cell_img_loc.split(well_folder)[2])
            ffc_img = cv.imread(ffc_img_loc, -1)
            ffc_img -= dark_count
            
            # load cell
            cell_img = cv.imread(cell_img_loc, -1)
            cell_img -= dark_count
        
            # calculated corrected image
            corr_img = ((cell_img * ffc_mean)/ffc_img).astype(np.uint16)
            # Data Standardization
            standard_img = (corr_img-ffc_mean)/ffc_std
            
            # Determine mask to remove dark regions and regions outside of dish
            ffc_mask = cv.threshold(ffc_img, ffc_thresh, 65535, cv.THRESH_BINARY)[1]
            corr_mask = cv.threshold(cell_img, cell_thresh, 65535, cv.THRESH_BINARY)[1]
            background_mask = ffc_mask * corr_mask
            
            # Segment out cells from background
            [outline, morph_img] = ca.analyze_img(standard_img, background_mask)
            
            # Keep track of areas to calculate coverage
            removed_area += np.count_nonzero(morph_img == 2)
            background_area += np.count_nonzero(morph_img == 1)
            cell_area += np.count_nonzero(morph_img == 0)
            
            # flip orientation for stitching
#            morph_img = cv.flip(morph_img, 0)
#            corr_img = cv.flip(corr_img, 0)
#            outline = cv.flip(outline, 0)
#            morph_img = np.rot90(morph_img)
#            corr_img = np.rot90(corr_img)
#            outline = np.rot90(outline)
            
            # Write images to file
            cv.imwrite(osp.join(analyzed_folder, well_folder + '_' + binary_folder, analyzed_filename + cell_img_loc.split(well_folder)[2]), morph_img)
            cv.imwrite(osp.join(analyzed_folder,well_folder + '_' + ff_corr_folder, analyzed_filename + cell_img_loc.split(well_folder)[2]), corr_img)
            # Add segmentation outline to corrected image
            corr_img[outline.astype(bool)] = 0
            cv.imwrite(osp.join(analyzed_folder, well_folder + '_' + outline_folder, analyzed_filename + cell_img_loc.split(well_folder)[2]), corr_img)
            
            # Code to deal with snaking error 
#            cell_labels = cell_img_loc.split('.')[0].split('_')
#            cell_num = file_list[-1].split('.')[0].split('_')[6]
#            temp_num_list = num_list[:num_list.index(cell_num)+1]
#            temp_num_list.reverse()
#            relabel_num = temp_num_list[num_list.index(cell_labels[6])]
#            
#            if int(cell_labels[7])%2 == 0:
#                # write image to file
#                cv.imwrite((analyzed_folder + '\\' + well_folder + '_' + binary_folder + '\\' + analyzed_filename + cell_img_loc.split(well_folder)[2]), morph_img)
#                cv.imwrite((analyzed_folder + '\\' + well_folder + '_' + outline_folder + '\\' + analyzed_filename + cell_img_loc.split(well_folder)[2]), corr_img)
#            else:
#                # For odd rows flip image order
#                cv.imwrite((analyzed_folder + well_folder + '_' + binary_folder + '\\' + analyzed_filename + file_prefix + relabel_num + '_' + cell_labels[7] + '.ome.tif') , morph_img)
#                cv.imwrite((analyzed_folder + well_folder + '_' + outline_folder + '\\' + analyzed_filename + file_prefix + relabel_num + '_' + cell_labels[7] + '.ome.tif'), corr_img)
        
        # Output and save coverage numbers
        print('The coverage is ', 100*cell_area/(cell_area + background_area), ' %')
        results[well_folder] = 100*cell_area/(cell_area + background_area)
        imjProcess = imageJStitching.stitchCoverage(root, plate_folder, well_folder, tileSize, outline_folder, binary_folder, imageJPath, outlineProcess)
            # Initialize txt file to save coverage numbers
    with csv.writer(open(osp.join(analyzed_folder, 'Coverage Percentage Results.txt'),'w')) as f:
        f.writerow(str(datetime.datetime.now()))
        f.writerow(plate_folder)     
        f.writerow(list(results.keys())) #Well folder names
        f.writerow(list(results.values()))
        
imjProcess.communicate() #wait for the last imagej process to finish.