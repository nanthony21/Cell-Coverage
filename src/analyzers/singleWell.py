import json
import os
import shutil
from enum import IntFlag
from glob import glob
from os import path as osp
import cv2 as cv
import imageio
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from matplotlib import pyplot as plt
from pwspy.dataTypes import Roi
from scipy import ndimage
from src.stitcher import Stitcher
from src.utility import Names
from src import utility
from pwspy.utility.matplotlibwidg import AdjustableSelector, MyEllipse
from PIL import Image

class OutputOptions(IntFlag):
    Full = 0xFF
    Outline = 0x01
    Binary = 0x02
    Corrected = 0x04
    ResultsJson = 0x08
    Nothing = 0x00

class SingleWellCoverageAnalyzer:
    def __init__(self, outPath: str, wellPath: str, ffcPath: str, darkCount: int,
                 rotate90: int = 0, outputOption: OutputOptions = OutputOptions.Full, varianceThreshold: float = 0.01):
        """root: Root folder for experiment."""
        self.wellPath = wellPath
        self.ffcPath = ffcPath
        self.outPath = outPath
        self.darkCount = darkCount
        self.rot = rotate90
        self.outputOption = outputOption
        self.varianceThreshold = varianceThreshold

        self.img = self._loadImage(self.wellPath)
        self.ffc = self._loadImage(self.ffcPath)

    def _loadImage(self, path: str):
        assert os.path.exists(path)
        fileNames = [os.path.split(impath)[-1] for impath in glob(osp.join(path, '*' + Names.prefix + '*'))]
        assert len(fileNames) > 0
        locations = [utility.getLocationFromFileName(name) for name in fileNames]
        tileSize = tuple(max(i) + 1 for i in zip(*locations))
        imgs = np.zeros((tileSize), dtype=object) # a 2d object array containing images by position.
        for loc in locations:
            im = utility.loadImage(loc, path)[0]
            im = np.rot90(im, self.rot) #rotate each individual image if needed.
            imgs[loc[0], loc[1]] = im
        s = Stitcher(imgs, .1, invertY=True)
        img = s.stitch() #Create a stitched image.
        img -= self.darkCount
        return img

    def _getStandardizedImg(self, mask: np.ndarray):
        ffcMean, ffcStd = self.ffc[mask].mean(), self.ffc[mask].std()
        stdImg = self.img * ffcMean / self.ffc
        stdImg = (stdImg - ffcMean) / ffcStd #An image that has been standardized by the flatfield correction.
        print(f"{np.percentile(stdImg[mask], 16)}, {np.percentile(stdImg[mask], 84)}")
        return stdImg

    def run(self, mask: np.ndarray) -> dict:
        if osp.exists(self.outPath):
            button = QMessageBox.question(None, 'Hey', f'Analysis folder already exists. Delete and continue?\n\n {self.outPath}')
            if button == QMessageBox.Yes:
                shutil.rmtree(self.outPath)
            else:
                return
        os.mkdir(self.outPath)
        stdImg = self._getStandardizedImg(mask)
        var = self.calculateLocalVariance(stdImg, 2)
        varMask = var <= self.varianceThreshold #The mask is true where variance is below threshold. (background regions.)
        morph_img = self.removeSmallComponents(varMask, 100)  # Remove small features
        kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Set kernels for morphological operations and CC
        morph_img = cv.dilate(morph_img.astype(np.uint8), kernel_dil) #This dilation helps the mask actually line up with the cells better.
        morph_img[~mask] = 2 #We now have a trinary array. (cell, background, ignored)
        # Write images to file
        with open(os.path.join(self.outPath, 'readme.txt'), 'w') as txtfile:
            if self.outputOption & OutputOptions.Binary:
                imageio.imwrite(osp.join(self.outPath, f'{Names.binary}.tif'), morph_img*127) #We multiply by 127 to use the whole 255 color range.
                txtfile.write("Binary: Black is cells, gray is background, white was not analyzed.\n\n")
            if self.outputOption & OutputOptions.Corrected:
                imageio.imwrite(osp.join(self.outPath, f'{Names.corrected}.tif'), stdImg.astype(np.float32))
                txtfile.write("Corrected: Recommend opening this in ImageJ. This is the raw data after being normalized by the flat field correction.\n\n")
            if self.outputOption & OutputOptions.Outline:
                # Add segmentation outline to corrected image
                outlinedImg = np.zeros((stdImg.shape[0], stdImg.shape[1], 3))
                outlinedImg[:, :, :] = stdImg[:, :, None] #Extend to 3rd dimension to make it RGB.
                Min, Max = np.percentile(outlinedImg, 1), np.percentile(outlinedImg, 99)
                outlinedImg = ((outlinedImg - Min) / (Max - Min) * 255).astype(np.uint8)  # scale data to 8bit.
                outline = cv.dilate(cv.Canny(morph_img.astype(np.uint8), 0, 1), cv.getStructuringElement(cv.MORPH_ELLIPSE,(2, 2)))  # binary outline for overlay
                rgboutline = np.zeros((*outline.shape, 3), dtype=np.bool)
                rgboutline[:, :, 0] = outline
                outlinedImg[rgboutline] = 255
                background = np.zeros((*outline.shape, 3), dtype=np.bool)
                background[:, :, 0] = (morph_img == 2)
                outlinedImg[background] = 0
                imageio.imwrite(osp.join(self.outPath, f'{Names.outline}.tif'), outlinedImg)
                txtfile.write("Outline: Check that the red outline matches the clusters of cells. Cyan regions were not analyzed.\n\n")

        # Output and save coverage numbers
        cellArea = float(np.sum(morph_img == 0))
        backgroundArea = float(np.sum(morph_img == 1))
        results = {'coverage': 100 * cellArea / (cellArea + backgroundArea)}
        if self.outputOption & OutputOptions.ResultsJson:
            with open(os.path.join(self.outPath, 'output.json'), 'w') as file:
                json.dump(results, file, indent=4)
        print(f"The coverage is {results['coverage']} %")
        return results

    @staticmethod
    def calculateLocalVariance(img, dist):
        """Creates a map of the spatial variance
        in a neighborhood of radius `dist` pixels in img"""
        def circularMask(radius):
            """Create a circular boolean mask with a radius of `dist`."""
            output_mask = np.ones([radius * 2 + 1, radius * 2 + 1], dtype=np.uint16)  # Initialize output mask
            # Create distance map
            output_mask[radius, radius] = 0
            dist_map = ndimage.distance_transform_edt(output_mask)
            # Turn distance map into binary mask
            output_mask[dist_map > radius] = 0
            output_mask[dist_map <= radius] = 1
            return output_mask
        img = img.astype(np.float32)
        mask = circularMask(dist)
        mask = mask / mask.sum()  # Normalize the mask to 1
        mean = cv.filter2D(img, cv.CV_32F, mask)
        sqrMean = cv.filter2D(img * img, cv.CV_32F, mask)
        return (sqrMean - mean * mean)  # Variance is the mean of the square minus the square of the mean.

    @staticmethod
    def removeSmallComponents(img, min_size):
        '''remove connected components smaller than min_size'''
        # find all your connected components
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
        sizes = stats[:, -1]  # connectedComponentswithStats yields every separated component with information on each of them, such as size
        img2 = np.zeros((output.shape))  # output_img
        selected = np.array(np.where(sizes[1:]>=min_size)) + 1
        img2[np.isin(output, selected)] = 1
        return img2.astype(img.dtype)

    def selectAnalysisArea(self) -> np.ndarray:
        downSampleScaler = max([self.img.shape[0]//1024, self.img.shape[1]//1024])
        dsimg = self.img[::downSampleScaler, ::downSampleScaler] # The Roi.fromVerts process takes forever on a huge stitched image. downsample to 1024,1024 then upsample back up.
        mask = [None] #Using a list since its mutable
        def finish(verts):
            r = Roi.fromVerts('doesntmatter', 0, verts=np.array(verts), dataShape=dsimg.shape)
            Mask = r.mask
            mask[0] = np.array(Image.fromarray(Mask).resize((self.img.shape[1], self.img.shape[0]))) #Upsample back to full resolution
            a.setActive(True)  # Reenable the selector for the next run.
        fig, ax = plt.subplots()
        fig.suptitle("Select the analysis region. Close to proceed.")
        im = ax.imshow(dsimg, clim=[np.percentile(dsimg, 1), np.percentile(dsimg, 99)], cmap='gray')
        a = AdjustableSelector(ax, im, MyEllipse, onfinished=finish)
        a.setActive(True)
        fig.show()
        while plt.fignum_exists(fig.number):
            fig.canvas.flush_events()
        return mask[0]


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    an = SingleWellCoverageAnalyzer(outPath=r'H:\HT29 coverage for Nick (8-20-19)\Low conf\BottomLeft_1\Ana2',
                                    wellPath= r'H:\HT29 coverage for Nick (8-20-19)\Low conf\BottomLeft_1',
                                    ffcPath=r'H:\HT29 coverage for Nick (8-20-19)\Flat field corr\BottomLeft_1',
                                    darkCount=624,
                                    rotate90=1,
                                    outputOption=OutputOptions.Outline)
    mask = an.selectAnalysisArea()
    an.run(mask)
