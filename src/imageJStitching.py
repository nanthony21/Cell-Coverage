# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:18:10 2018

@author: Nick Anthony
"""
import subprocess
from typing import Tuple
import os

class ImageJStitcher:
    def __init__(self, imageJPath: str):
        if not os.path.exists(imageJPath):
            raise OSError("imageJ could not be found at {}".format(imageJPath))
        self.imjPath = imageJPath

    def stitchCoverage(self, rootDir: str, plate: str, well: str, gridSize: Tuple[int, int], analysisFolder: str,
                       outlineFolderName: str, binaryFolderName: str, previousStitchingProcess=None):
        """Used by the coverage analysis code to stich outline images and binarized images."""
        if previousStitchingProcess is not None:
            if previousStitchingProcess.poll() is None: #this means the process is still running
                print("\t\tfinishing imagej stitch process")
                previousStitchingProcess.wait()
                print('\t\tdone')
            stdout, stderr = previousStitchingProcess.communicate()

        file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif"

        outlinePath = os.path.join(rootDir, plate, analysisFolder, well+'_'+outlineFolderName)
        binaryPath = os.path.join(rootDir, plate, analysisFolder, well+'_'+binaryFolderName)
        outlineString = self._genStitchString(gridSize, 10, (0, 0), outlinePath, file_name)
        binaryString = self._genStitchString(gridSize, 10, (0, 0), binaryPath, file_name)

        imJCmd = outlineString +  \
        f'''run('Apply LUT');
        run('8-bit');
        saveAs('Jpeg', '{os.path.join(rootDir, plate, analysisFolder, well + outlineFolderName + ".jpg")}');
        close();''' +   \
        binaryString +  \
        f'''run('8-bit');
        saveAs('Jpeg', '{os.path.join(rootDir, plate, analysisFolder, well + binaryFolderName + ".jpg")}');
        close();'''

        proc = self.runImJCmd(imJCmd, os.path.join(rootDir, plate, analysisFolder))
        return proc


    def _genStitchString(self, gridSize: Tuple[int,int], overlap: int, firstFileIndices: Tuple[int,int], directory: str, fileName: str):
        """Generates a string that can be used to run a stich process in imagej. Feed this string along with other commands to `runImJCmd`"""
        cmd = f'''
        run('Grid/Collection stitching', 'type=[Filename defined position] order=[Defined by filename] grid_size_x={gridSize[0]} grid_size_y={gridSize[1]}
        tile_overlap={overlap} first_file_index_x={firstFileIndices[0]} first_file_index_y={firstFileIndices[1]} directory=[{directory}] file_names={fileName}
        output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]');
        '''
        return cmd

    def runImJCmd(self, imJCmd: str, logDirectory: str = None):
        """Given an imagej macro in string form this will start running the macro and return you a reference to the process.
         If given a log directory it will save the stdout and stderr output to two text files."""
        imJCmd = '"' + imJCmd + '"' #Wrap the command in quotes so the terminal treats it like a string.
        imJCmd = imJCmd.replace('\n', '')    #Remove newlines which mess everything up.
        imJCmd = imJCmd.replace('\\', '\\\\') #Escape out our file separators
        if logDirectory is None:
            proc = subprocess.Popen(self.imjPath + ' --headless --console -eval ' + imJCmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            with open(os.path.join(logDirectory, 'stdoutlog.txt'), 'a') as f, open(os.path.join(logDirectory, 'stderrlog.txt'), 'a') as f2:
                f.write('New Process\n')
                f2.write('New Process\n')
                proc = subprocess.Popen(self.imjPath + ' --headless --console -eval ' + imJCmd, stdout = f, stderr = f2, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        return proc