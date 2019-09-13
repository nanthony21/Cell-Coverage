from typing import Tuple, List
from PyQt5.QtWidgets import QMessageBox
from src.imageJStitching import ImageJStitcher
import os
import shutil
from glob import glob


class SinglePlateAnalyzer:
    def __init__(self, outPath: str, platePath: str, ffcPath: str, centerImgLocation: List[Tuple[int, int]], edgeImgLocation: List[Tuple[int, int]], darkCount: int,
                 stitcher: ImageJStitcher, rotate90: int = 0, debug: bool = False):
        self.darkCount = darkCount
        self.stitcher = stitcher
        self.rot = rotate90
        self.debug = debug
        self.outPath = outPath
        assert len(centerImgLocation) == len(edgeImgLocation)
        glob(os.path.join(platePath, '*', '*.tif'))

    def run(self):
        if os.path.exists(self.outPath):
            button = QMessageBox.question(None, 'Hey', f'Analysis folder already exists. Delete and continue?\n\n {self.outPath}')
            if button == QMessageBox.Yes:
                shutil.rmtree(self.outPath)
            else:
                return

