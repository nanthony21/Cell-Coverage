import os
import shutil
from glob import glob
from os import path as osp
from typing import Tuple

import cv2 as cv
import imageio
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from matplotlib import pyplot as plt

from src.utility import Names
from src.analyzers.singleImage import SingleImageAnalyzer
from src.imageJStitching import ImageJStitcher
from src import utility


class SingleWellCoverageAnalyzer:
    def __init__(self, outPath: str, wellPath: str, ffcPath: str, centerImgLocation: Tuple[int, int], edgeImgLocation: Tuple[int, int], darkCount: int,
                 stitcher: ImageJStitcher, rotate90: int = 0, debug: bool = False):
        """root: Root folder for experiment."""
        self.wellPath = wellPath
        self.ffcPath = ffcPath
        self.outPath = outPath
        self.center = centerImgLocation
        self.edge = edgeImgLocation
        self.darkCount = darkCount
        self.stitcher = stitcher
        self.rot = rotate90
        self.debug = debug



    def run(self):
        if osp.exists(self.outPath):
            button = QMessageBox.question(None, 'Hey', f'Analysis folder already exists. Delete and continue?\n\n {self.outPath}')
            if button == QMessageBox.Yes:
                shutil.rmtree(self.outPath)
            else:
                return
        for folder in [Names.outline, Names.binary, Names.corrected]:
            fullfolder = osp.join(self.outPath, folder)
            os.makedirs(fullfolder)

        # Mean value of center image is used for flat field correction
        ffc_center, _ = utility.loadImage(self.center, self.ffcPath)
        ffc_center -= self.darkCount
        ffc_mean = ffc_center.mean()
        ffc_std = ffc_center.std()

        # FFC edge images are used to threshold the area outside the dish
        ffc_edge, _ = utility.loadImage(self.edge, self.ffcPath)
        ffc_edge -= self.darkCount
        ffc_thresh = self.otsu_1d(ffc_edge, wLowOpt=1, debug=self.debug)

        # FF corrected cell edge images are used to threshold the edge effects from the dish
        cell_edge, _ = utility.loadImage(self.edge, self.wellPath)
        cell_edge -= self.darkCount
        cell_edge = ((cell_edge * ffc_mean) / ffc_edge).astype(np.uint16) #TODO what does this achieve
        cell_thresh = self.otsu_1d(cell_edge, wLowOpt=1, debug=self.debug)


        # Intialize coverage variables
        Cell_area = Background_area = Removed_area = 0

        # loop through cell images
        fileNames = [os.path.split(path)[-1] for path in glob(osp.join(self.wellPath, '*' + Names.prefix + '*'))]
        locations = [utility.getLocationFromFileName(name) for name in fileNames]
        tileSize = tuple(max(i)+1 for i in zip(*locations))
        for location in locations:
            ffc_img, _ = utility.loadImage(location, self.ffcPath)
            cell_img, _ = utility.loadImage(location, self.wellPath)

            analyzer = SingleImageAnalyzer(self.darkCount, self.rot, cell_img, ffc_img)
            standard_img, background_mask, morph_img, removed_area, background_area, cell_area = analyzer.run(ffc_mean, ffc_std, cell_thresh, ffc_thresh)
            if self.debug:
                plt.figure()
                plt.imshow(standard_img, cmap='gray')
                plt.imshow(np.logical_not(background_mask), alpha=0.6, clim=[0, 1], cmap='Reds')
                plt.figure()
                n, bins, patches = plt.hist(cell_img.flatten(), bins=100)
                plt.vlines(cell_thresh, 0, max(n))
                plt.title("Cell Image")
                plt.figure()
                n, bins, patches = plt.hist(ffc_img.flatten(), bins=100)
                plt.vlines(ffc_thresh, 0, max(n))
                plt.title("FFC Image")
                plt.figure()
                plt.imshow(morph_img)
                plt.show(block=False)
            # Keep track of areas to calculate coverage
            Removed_area += removed_area
            Background_area += background_area
            Cell_area += cell_area

            # Write images to file
            fileName = f"{location[0]:03d}_{location[1]:03d}.tif"
            imageio.imwrite(osp.join(self.outPath, Names.binary, fileName), morph_img*127) #We multiply by 127 to use the whole 255 color range.
            imageio.imwrite(osp.join(self.outPath, Names.corrected, fileName), standard_img.astype(np.float32))
            # Add segmentation outline to corrected image
            outlinedImg = np.zeros((standard_img.shape[0], standard_img.shape[1], 3))
            outlinedImg[:, :, :] = standard_img[:, :, None] #Extend to 3rd dimension to make it RGB.
            outlinedImg = ((outlinedImg - outlinedImg.min()) / (outlinedImg.max() - outlinedImg.min()) * 255).astype(np.uint8)  # scale data to 8bit.
            outline = cv.dilate(cv.Canny(morph_img.astype(np.uint8), 0, 1),cv.getStructuringElement(cv.MORPH_ELLIPSE,(2, 2)))  # binary outline for overlay
            rgboutline = np.zeros((*outline.shape, 3), dtype=np.bool)
            rgboutline[:, :, 0] = outline
            outlinedImg[rgboutline] = 255
            imageio.imwrite(osp.join(self.outPath, Names.outline, fileName), outlinedImg)

        # Output and save coverage numbers
        results = 100 * Cell_area / (Cell_area + Background_area)
        print(f'The coverage is {results} %')

        self.stitcher.stitch(os.path.join(self.outPath, Names.outline), tileSize, "{xxx}_{yyy}.tif")
        self.stitcher.stitch(os.path.join(self.outPath, Names.binary), tileSize, "{xxx}_{yyy}.tif")
        # self.stitcher.waitOnProcesses()
        return results

    @staticmethod
    def otsu_1d(img, wLowOpt=None, wHighOpt=None, debug: bool = False):
        ''' calculates the threshold for binarization using Otsu's method.
        The weights for the low and high distribution can be overridden using the optional arguments.
        '''
        flat_img = img.flatten()
        var_b_max = 0
        bin_index = 0

        num_bins = 100  # Can reduce num_bins to speed code, but reduce accuracy of threshold
        img_min = np.percentile(flat_img, 1)
        img_max = np.percentile(flat_img, 99)
        variances = []
        thresholdVals = np.linspace(img_min, img_max, num_bins, endpoint=False)
        for bin_val in thresholdVals:
            # segment data based on bin
            gLow = flat_img[flat_img <= bin_val]
            gHigh = flat_img[flat_img > bin_val]

            # determine weights of each bin
            wLow = gLow.size / flat_img.size if (wLowOpt is None) else wLowOpt
            wHigh = gHigh.size / flat_img.size if (wHighOpt is None) else wLowOpt

            # maximize inter-class variance
            var_b = wLow * wHigh * (gLow.mean() - gHigh.mean()) ** 2
            variances.append(var_b)
        threshold = thresholdVals[np.argmax(variances)]
        if debug:
            fig, ax = plt.subplots()
            ax2 = plt.twinx(ax)
            ax2.plot(thresholdVals, variances, color='r')
            ax.hist(flat_img, bins=num_bins)
            ax2.vlines(threshold, 0, max(variances))
            ax.set_xlim(img_min, img_max)
            fig.show()
        return threshold


if __name__ == '__main__':
    stitcher = ImageJStitcher(r'C:\Users\backman05\Documents\Fiji.app\ImageJ-win64.exe')
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    an = SingleWellCoverageAnalyzer(outPath=r'H:\HT29 coverage myo + cele (8-26-19)\48h\BottomLeft_1\Ana3',
                                    wellPath=r'H:\HT29 coverage myo + cele (8-26-19)\48h\BottomLeft_1',
                                    ffcPath=r'H:\HT29 coverage myo + cele (8-26-19)\Flat field corr 48h\BottomLeft_1',
                                    centerImgLocation=(0,6),
                                    edgeImgLocation=(1,1),
                                    darkCount=624,
                                    stitcher=stitcher,
                                    rotate90=0,
                                    debug=False)
    an.run()