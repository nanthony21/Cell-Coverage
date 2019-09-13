import json
from typing import Tuple, List
import re
from PyQt5.QtWidgets import QMessageBox

from src.analyzers.singleWell import SingleWellCoverageAnalyzer
from src.imageJStitching import ImageJStitcher
import os
import shutil
from glob import glob
from src.utility import Names
from src import utility
import matplotlib.pyplot as plt

class SinglePlateAnalyzer:
    def __init__(self, outPath: str, platePath: str, ffcPath: str, darkCount: int,
                 stitcher: ImageJStitcher, rotate90: int = 0, debug: bool = False):
        self.darkCount = darkCount
        self.stitcher = stitcher
        self.rot = rotate90
        self.debug = debug
        self.outPath = outPath
        self.platePath = platePath
        self.ffcPath = ffcPath
        if not os.path.exists(os.path.join(self.platePath, 'plateConfig.json')):
            QMessageBox.information(None, 'On No', f'Could not find a `plateConfig.json` file in {self.platePath}')
            raise Exception("plateConfig.json missing.")
        with open(os.path.join(self.platePath, 'plateConfig.json')) as f:
            self.plateConfig = json.load(f)
        plateStructure = self.detectPlateFolderStructure(platePath)
        ffcStructure = self.detectPlateFolderStructure(ffcPath)
        assert all([any([re.match(pattern, name) for name in plateStructure.keys()]) for pattern in self.plateConfig.keys()]), "The subdirectory names in `plateConfig.json` could not be matched to the folder structure."
        assert plateStructure.keys() == ffcStructure.keys()
        for k, v in plateStructure.items():
            assert ffcStructure[k] == v, f"For Well {k}: plate has locations: {v}, but flatField has locations: {ffcStructure[k]}"
        self.plateStructure = plateStructure

        if os.path.exists(self.outPath):
            button = QMessageBox.question(None, 'Hey',
                                          f'Analysis folder already exists. Delete and continue?\n\n {self.outPath}')
            if button == QMessageBox.Yes:
                shutil.rmtree(self.outPath)
            else:
                return

    @staticmethod
    def detectPlateFolderStructure(path: str):
        """Returns a dictionary where the keys are the detected `well` folders of `path` and the values or the location tuples for that `well` folder"""
        allFiles = glob(os.path.join(path, '*', f'*{Names.prefix}*')) #All files matching the image fileName prefix and in subdirectories of the main path.
        subDirs = set([os.path.split(os.path.split(i)[0])[-1] for i in allFiles]) #The subdirectories of `path` that contain image files.
        d = {}
        for subDir in subDirs:
            d[subDir] = [utility.getLocationFromFileName(os.path.split(name)[-1]) for name in allFiles if subDir == os.path.split(os.path.split(name)[0])[-1]]
        return d

    def run(self):
        for i, wellFolder in enumerate(self.plateStructure.keys()):
            print(wellFolder)
            config = [v for k,v in self.plateConfig.items() if re.match(k, wellFolder)][0]
            well = SingleWellCoverageAnalyzer(outPath=os.path.join(self.outPath, wellFolder),
                                       wellPath=os.path.join(self.platePath, wellFolder),
                                       ffcPath=os.path.join(self.ffcPath, wellFolder),
                                       centerImgLocation=config['Center'],
                                       edgeImgLocation=config['Edge'],
                                       darkCount=self.darkCount,
                                       stitcher=self.stitcher,
                                       rotate90=self.rot,
                                       debug=self.debug)
            well.run()
            if self.debug:
                print(f"Press ctrl+c to when done with well {wellFolder}: ")
                try:
                    while True:
                        plt.pause(0.05)
                except KeyboardInterrupt:
                    print("continuing.")

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    with ImageJStitcher(r'C:\Users\backman05\Documents\Fiji.app\ImageJ-win64.exe') as stitcher:
        plate = SinglePlateAnalyzer(outPath=r'H:\HT29 coverage myo + cele (8-26-19)\48h\Analyzeddd',
                                    platePath=r'H:\HT29 coverage myo + cele (8-26-19)\48h',
                                    ffcPath=r'H:\HT29 coverage myo + cele (8-26-19)\Flat field corr 48h',
                                    darkCount=624,
                                    stitcher=stitcher,
                                    rotate90=1,
                                    debug=False)
        plate.run()