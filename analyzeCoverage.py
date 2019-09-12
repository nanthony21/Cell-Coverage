# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:18:32 2018

@author: Nick Anthony
"""
from glob import glob
from matplotlib import pyplot as plt
import os
from src.coverageAnalysisFuncs import analyzeCoverage


'''********User Inputs!!*******'''

## Root folder for experiment
rootPath = r'G:\Coverage vs sigma (3-20-19)'
# Flat Field correction path
ffc_folder = r'G:\Coverage vs sigma (3-20-19)\Blank'
# Folder names for each plate to analyze relative to rootPath
plate_folder_list = ['Coverageplate1']# e.g. glob(osp.join(rootPath, '*plate*')) # Select all subfolders of root if they have plate in the name. ['A2780_48Hour_Plate1','A2780_48Hour_Plate2','A2780_48Hour_Plate3']

# Folder and file names for individual well in plate. The last bit of the file name can be ignored for increased flexibility.
well_folder_list = ['BottomLeft', 'BottomMid', 'BottomRight', 'TopLeft', 'TopMid', 'TopRight']
# image index for center image for flatfielding
center_locations = [('000','006'), ('002','005'), ('002','005'), ('002','005'), ('002','005'), ('002','005')]
# image index for edge image for masking
edge_locations = [('001','001'), ('001','008'), ('001','008'), ('001','001'), ('001','008'), ('001','008')]

analysisNum:int = 1#A number to be added as asuffix to the output files
dark_count = 624 # camera dark counts
imageJPath = r'C:\Users\backman05\Documents\Fiji.app\ImageJ-win64.exe'


'''**********************'''

analyzeCoverage(rootPath, plate_folder_list, well_folder_list,
                center_locations, edge_locations, analysisNum,
                dark_count, imageJPath, ffc_folder)