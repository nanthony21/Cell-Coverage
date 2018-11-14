# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:18:32 2018

@author: Scott
"""
from glob import glob
from matplotlib import pyplot as plt
import os
from src.coverageAnalysisFuncs import analyzeCoverage


'''********User Inputs!!*******'''

## Root folder for experiment
root = r'E:\SeqExp\m248'
# Folder names for each plate to analyze
plate_folder_list = ['16hour-plate2-control-pac']#glob(osp.join(root, '*plate*')) # Select all subfolders of root if they have plate in the name. ['A2780_48Hour_Plate1','A2780_48Hour_Plate2','A2780_48Hour_Plate3']
# Folder and file names for individual well in plate
well_folder_list = ['BottomLeft_1', 'BottomMid_1', 'BottomRight_1', 'TopLeft_1', 'TopMid_1', 'TopRight_1']
# image index for center image for flatfielding
center_locations = [('000','006'), ('002','005'), ('002','005'), ('002','005'), ('002','005'), ('002','005')]
# image index for edge image for masking
edge_locations = [('001','001'), ('001','008'), ('001','008'), ('001','001'), ('001','008'), ('001','008')]
#A number to be added as asuffix to the output files
analysisNum:int = 1
dark_count = 624 # camera dark counts
imageJPath = r'C:\Users\N2-LiveCell\Documents\fiji-win64\Fiji.app\ImageJ-win64.exe'
# Flat Field correction path
ffc_folder = r'E:\SeqExp\FlatField'

'''**********************'''

analyzeCoverage(root, plate_folder_list, well_folder_list,
                center_locations, edge_locations, analysisNum,
                dark_count, imageJPath, ffc_folder)