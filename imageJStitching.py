# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:18:10 2018

@author: backman05
"""
import subprocess
import typing

imageJPath = r'C:\Users\backman05\Documents\Fiji.app\ImageJ-win64.exe'

def stitchCoverage(rootDir:str, plateFolderList:typing.List[str], wellFolderList:typing.List[str], gridSizes:typing.List[typing.Tuple[int,int]]):
    file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif";
    outline_pre = "_Outline 2"
    binary_pre = "_Binary 2"

    for plate in plateFolderList:
        for i, well in enumerate(wellFolderList):
            imJCmd = f'''
            run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] xGridSizes="
                + {gridSizes[i][0]} + " yGridSizes=" + {gridSizes[i][1]} + " tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory=["
                + {rootDir} + {plate} + "\\Analyzed\\" + {well} + {outline_pre} + "] file_names=" + {file_name} + 
                " output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
            run("Enhance Contrast", "saturated=0.35");
            run("Apply LUT");
            run("8-bit");
            saveAs("Jpeg", {rootDir} + {plate} + "\\analyzed\\" + {well} + {outline_pre} + ".jpg");
            close();
            '''
            proc = subprocess.Popen(imageJPath + ' --headless --console -eval ' + imJCmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            stdout, stderr = proc.communicate()
            a=1
    '''
    for plate in plateFolderList:
        for i, well in enumerate(wellFolderList):
            run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] xGridSizes="
               + xGridSizes[i] + " yGridSizes=" + yGridSizes[i] + " tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory=["
               + rootDir + plate + "\\Analyzed\\" + well + binary_pre + "] file_names=" + file_name + 
               " output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]")
            run("8-bit")
            saveAs("Jpeg", rootDir + plate + "\\analyzed\\" + well + binary_pre + ".jpg")
            close()
    '''