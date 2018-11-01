# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:18:10 2018

@author: backman05
"""
import subprocess
import typing
import os


#%%


def stitchCoverage(rootDir:str, plate:str, well:str, gridSize:typing.Tuple[int,int], outlineFolderName:str, binaryFolderName:str, imageJPath:str, previousStitchingProcess = None):
    
    if previousStitchingProcess is not None:
        if previousStitchingProcess.poll() is None: #this means the process is still running
            print("\t\tfinishing imagej stitch process")
            previousStitchingProcess.wait()
            print('\t\tdone')
        stdout, stderr = previousStitchingProcess.communicate()

    
    file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif";

    imJCmd = f'''
    "run('Grid/Collection stitching', 'type=[Filename defined position] order=[Defined by filename] grid_size_x={gridSize[0]} grid_size_y={gridSize[1]}
    tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory=[{os.path.join(rootDir,plate,"Analyzed",well+'_'+outlineFolderName)}] file_names={file_name}
    output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]');
    run('Enhance Contrast', 'saturated=0.35');
    run('Apply LUT');
    run('8-bit');
    saveAs('Jpeg', '{os.path.join(rootDir,plate,"Analyzed",well + outlineFolderName + ".jpg")}');
    close();
    run('Grid/Collection stitching', 'type=[Filename defined position] order=[Defined by filename] grid_size_x={gridSize[0]} grid_size_y={gridSize[1]}
    tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory=[{os.path.join(rootDir,plate,"Analyzed",well+'_'+binaryFolderName)}] file_names={file_name}
    output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]');
    run('8-bit');
    saveAs('Jpeg', '{os.path.join(rootDir,plate,"Analyzed",well + binaryFolderName + ".jpg")}');
    close();"
    '''
    imJCmd = imJCmd.replace('\n','')    #Remove newlines which mess everything up.
    imJCmd = imJCmd.replace('\\','\\\\') #Escape out our file separators
    with open(os.path.join(rootDir,plate, 'Analyzed','stdoutlog.txt'),'a') as f, open(os.path.join(rootDir,plate, 'Analyzed','stderrlog.txt'),'a') as f2:
        f.write(well+'\n')
        f2.write(well+'\n')
        proc = subprocess.Popen(imageJPath + ' --headless --console -eval ' + imJCmd, stdout = f, stderr = f2, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return proc

