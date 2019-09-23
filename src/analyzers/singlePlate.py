
import h5py
from PyQt5.QtWidgets import QMessageBox
from src.analyzers.singleWell import SingleWellCoverageAnalyzer, OutputOptions
import os
import shutil
from glob import glob
from src.utility import Names
from src import utility
import numpy as np
from enum import Enum
from src import masksPath
import json


class Masks(Enum):
    """Used to select which mask files to use"""
    SixWell = '6WellPlate'
    Diy = None #Draw the masks yourself.

class SinglePlateAnalyzer:
    def __init__(self, platePath: str, ffcPath: str, darkCount: int, maskOption: Masks, rotate90: int = 0):
        self.darkCount = darkCount
        self.rot = rotate90
        self.maskOption = maskOption
        self.outPath = os.path.join(platePath, 'Analysis')
        self.platePath = platePath
        self.ffcPath = ffcPath
        plateStructure = self.detectPlateFolderStructure(platePath)
        ffcStructure = self.detectPlateFolderStructure(ffcPath)
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
        os.mkdir(self.outPath)

    @staticmethod
    def detectPlateFolderStructure(path: str):
        """Returns a dictionary where the keys are the detected `well` folders of `path` and the values or the location tuples for that `well` folder"""
        allFiles = glob(os.path.join(path, '*', f'*{Names.prefix}*')) #All files matching the image fileName prefix and in subdirectories of the main path.
        subDirs = set([os.path.split(os.path.split(i)[0])[-1] for i in allFiles]) #The subdirectories of `path` that contain image files.
        d = {}
        for subDir in subDirs:
            d[subDir] = [utility.getLocationFromFileName(os.path.split(name)[-1]) for name in allFiles if subDir == os.path.split(os.path.split(name)[0])[-1]]
        return d

    def run(self, varianceThreshold: float = 0.01, kernelDiameter: int = 5, minimumComponentSize: int = 100):
        results = {}
        for i, wellFolder in enumerate(self.plateStructure.keys()):
            print(wellFolder)
            well = SingleWellCoverageAnalyzer(outPath=os.path.join(self.outPath, wellFolder),
                                              wellPath=os.path.join(self.platePath, wellFolder),
                                              ffcPath=os.path.join(self.ffcPath, wellFolder),
                                              darkCount=self.darkCount,
                                              rotate90=self.rot,
                                              outputOption=OutputOptions.Outline | OutputOptions.Binary)
            if self.maskOption == Masks.Diy:
                print("Draw the analysis area")
                mask = well.selectAnalysisArea()
                if not os.path.exists(os.path.join(self.platePath, 'masks')):
                    os.mkdir(os.path.join(self.platePath, 'masks'))
                with h5py.File(os.path.join(self.platePath, 'masks', f'{wellFolder}.h5'), 'w') as f:
                    f.create_dataset('mask', dtype=np.bool, data=mask, compression='gzip')
            else:
                maskPath = os.path.join(masksPath, self.maskOption.value, f'{wellFolder}.h5')
                with h5py.File(maskPath, 'r') as f:
                    mask = np.array(f['mask'])
                print("Mask loaded")
            results[wellFolder] = well.run(mask,
                                           varianceThreshold=varianceThreshold,
                                           kernelDiameter=kernelDiameter,
                                           minimumComponentSize=minimumComponentSize)
        with open(os.path.join(self.outPath, 'results.csv'), 'w') as f:
            for well, result in results.items():
                f.write(f"{well}, {result['coverage']}\n")
        with open(os.path.join(self.outPath, 'settings.json'), 'w') as f:
            json.dump(f, {'kernelDiameter': kernelDiameter, 'minimumComponentSize': minimumComponentSize, 'varianceThreshold': varianceThreshold})
        return results

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    plate = SinglePlateAnalyzer(
                                platePath=r'H:\HT29 coverage for Nick (8-20-19)\HT29 coverage48h (8-20-19)\Low conf',
                                ffcPath=r'H:\HT29 coverage for Nick (8-20-19)\HT29 coverage48h (8-20-19)\Flat field corr',
                                darkCount=624,
                                maskOption=Masks.SixWell,
                                rotate90=2)
    results = plate.run()
