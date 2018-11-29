# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:18:10 2018

@author: backman05
"""
import subprocess
import typing
import os


#%%


def stitchCoverage(rootDir:str, plate:str, well:str, gridSize:typing.Tuple[int,int], analysisFolder:str, outlineFolderName:str, binaryFolderName:str, imageJPath:str, previousStitchingProcess = None):
    
    if previousStitchingProcess is not None:
        if previousStitchingProcess.poll() is None: #this means the process is still running
            print("\t\tfinishing imagej stitch process")
            previousStitchingProcess.wait()
            print('\t\tdone')
        stdout, stderr = previousStitchingProcess.communicate()

    
    file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif";

    outlineString = genStitchString(gridSize, 10, (0, 0), os.path.join(rootDir,plate, analysisFolder,well+'_'+outlineFolderName), file_name)
    binaryString = genStitchString(gridSize, 10, (0, 0), os.path.join(rootDir,plate, analysisFolder,well+'_'+binaryFolderName), file_name)

    imJCmd = outlineString +  \
    f'''run('Apply LUT');
    run('8-bit');
    saveAs('Jpeg', '{os.path.join(rootDir,plate,analysisFolder,well + outlineFolderName + ".jpg")}');
    close();''' +   \
    binaryString +  \
    f'''run('8-bit');
    saveAs('Jpeg', '{os.path.join(rootDir,plate,analysisFolder,well + binaryFolderName + ".jpg")}');
    close();'''

    proc = runImJCmd(imJCmd, imageJPath, os.path.join(rootDir,plate, analysisFolder))
    return proc

def genStitchString(gridSize:typing.Tuple[int,int], overlap:int, firstFileIndices:typing.Tuple[int,int], directory:str, fileName:str):
    cmd = f'''
    run('Grid/Collection stitching', 'type=[Filename defined position] order=[Defined by filename] grid_size_x={gridSize[0]} grid_size_y={gridSize[1]}
    tile_overlap={overlap} first_file_index_x={firstFileIndices[0]} first_file_index_y={firstFileIndices[1]} directory=[{directory}] file_names={fileName}
    output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]');
    '''
    return cmd
    
def runImJCmd(imJCmd:str, imageJPath:str, logDirectory:str = None):
    imJCmd = '"' + imJCmd + '"' #Wrap the command in quotes so the terminal treats it like a string.
    imJCmd = imJCmd.replace('\n','')    #Remove newlines which mess everything up.
    imJCmd = imJCmd.replace('\\','\\\\') #Escape out our file separators
    if not (logDirectory is None):
        with open(os.path.join(logDirectory,'stdoutlog.txt'),'a') as f, open(os.path.join(logDirectory,'stderrlog.txt'),'a') as f2:
            f.write('New Process\n')
            f2.write('New Process\n')
            proc = subprocess.Popen(imageJPath + ' --headless --console -eval ' + imJCmd, stdout = f, stderr = f2, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        proc = subprocess.Popen(imageJPath + ' --headless --console -eval ' + imJCmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    return proc