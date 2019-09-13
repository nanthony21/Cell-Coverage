import itertools
import os
import shutil
from glob import glob
from os import path as osp
from typing import Tuple

import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication
from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
from src.imageJStitching import ImageJStitcher
import skimage.filters


class SingleWellCoverageAnalyzer:
    def __init__(self, outPath: str, wellPath: str, ffcPath: str, centerImgLocation: Tuple[int, int], edgeImgLocation: Tuple[int, int], darkCount: int,
                 stitcher: ImageJStitcher, rotate90: int = 0):
        """root: Root folder for experiment."""
        self.wellPath = wellPath
        self.ffcPath = ffcPath
        self.outPath = outPath
        self.center = centerImgLocation
        self.edge = edgeImgLocation
        self.darkCount = darkCount
        self.stitcher = stitcher
        self.rot = rotate90

    @staticmethod
    def loadImage(location: Tuple[int, int], path: str):
        fileName = f'*{Names.prefix}{location[0]:03d}_{location[1]:03d}.ome.tif'
        try:
            searchPattern = osp.join(path, fileName)
            fullPath = glob(searchPattern)[0]
            fileName = os.path.split(fullPath)[-1]
        except IndexError:
            raise OSError(f"Couldn't find a file matching pattern {searchPattern}")
        img = cv.imread(fullPath, -1)
        if img is None:
            raise OSError(f"The image, {fullPath}, was not found")
        return img, fileName

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
        ffc_center, _ = self.loadImage(self.center, self.ffcPath)
        ffc_center -= self.darkCount
        ffc_mean = ffc_center.mean()
        ffc_std = ffc_center.std()

        # FFC edge images are used to threshold the area outside the dish
        ffc_edge, _ = self.loadImage(self.edge, self.ffcPath)
        ffc_edge -= self.darkCount
        ffc_thresh = skimage.filters.threshold_otsu(ffc_edge)

        # FF corrected cell edge images are used to threshold the edge effects from the dish
        cell_edge, _ = self.loadImage(self.edge, self.wellPath)
        cell_edge -= self.darkCount
        cell_edge = ((cell_edge * ffc_mean) / ffc_edge).astype(np.uint16) #TODO what does this achieve
        cell_thresh = skimage.filters.threshold_otsu(cell_edge)


        # Intialize coverage variables
        Cell_area = Background_area = Removed_area = 0

        # loop through cell images
        file_list = glob(osp.join(self.wellPath, '*' + Names.prefix + '*'))
        tileSize = (max([int(i.split('Pos')[-1].split('.')[0].split('_')[0]) for i in file_list]) + 1,
                    max([int(i.split('Pos')[-1].split('.')[0].split('_')[1]) for i in file_list]) + 1)

        for location in itertools.product(range(tileSize[0]), range(tileSize[1])):
            ffc_img, _ = self.loadImage(location, self.ffcPath)
            cell_img, _ = self.loadImage(location, self.wellPath)

            analyzer = SingleImageAnalyzer(self.darkCount, self.rot)
            standard_img, background_mask, morph_img, removed_area, background_area, cell_area = analyzer.run(ffc_img, cell_img, ffc_mean, ffc_std, cell_thresh, ffc_thresh)

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
            rgboutline[:,:,0] = outline
            outlinedImg[rgboutline] = 255
            imageio.imwrite(osp.join(self.outPath, Names.outline, fileName), outlinedImg)

        # Output and save coverage numbers
        results = 100 * Cell_area / (Cell_area + Background_area)
        print(f'The coverage is {results} %')

        self.stitcher.stitch(os.path.join(self.outPath, Names.outline), tileSize, "{xxx}_{yyy}.tif")
        self.stitcher.stitch(os.path.join(self.outPath, Names.binary), tileSize, "{xxx}_{yyy}.tif")
        self.stitcher.waitOnProcesses()
        return results



class SingleImageAnalyzer:
    def __init__(self, darkCount: int, rotate90: int):
        self.darkCount = darkCount
        self.rot90 = rotate90

    def run(self, ffcImg: np.ndarray, img: np.ndarray, ffcMean: float, ffcStd: float, cellThresh: float, ffcThresh: float):
        ffcImg -= self.darkCount
        img -= self.darkCount
        ffcImg = np.rot90(ffcImg, self.rot90)
        img = np.rot90(img, self.rot90)
        std = self.standardizeImage(img, ffcImg, ffcMean, ffcStd)
        backgroundMask = self.calculateBackground(img, ffcImg, cellThresh, ffcThresh)
        morph_img = self.analyze_img(std)
        morph_img[~backgroundMask.astype(bool)] = 2 #Use a third value to show regions that are considered to be outside the dish
        # Keep track of areas to calculate coverage
        removed_area = np.count_nonzero(morph_img == 2)
        background_area = np.count_nonzero(morph_img == 1)
        cell_area = np.count_nonzero(morph_img == 0)
        return std, backgroundMask, morph_img, removed_area, background_area, cell_area

    @staticmethod
    def calculateBackground(image: np.ndarray, flatField: np.ndarray, imageThreshold: float, flatFieldThreshold: float):
        # Determine mask to remove dark regions and regions outside of dish
        ffc_mask = cv.threshold(flatField, flatFieldThreshold, 65535, cv.THRESH_BINARY)[1]
        corr_mask = cv.threshold(image, imageThreshold, 65535, cv.THRESH_BINARY)[1]
        backgroundMask = ffc_mask * corr_mask
        return backgroundMask

    @staticmethod
    def standardizeImage(image: np.ndarray, flatField: np.ndarray, meanIntensity: float, stdIntensity: float):
        corr_img = ((image * meanIntensity) / flatField).astype(np.uint16)  # calculated corrected image
        standardImg = (corr_img - meanIntensity) / stdIntensity  # Data Standardization
        return standardImg

    @staticmethod
    def analyze_img(img: np.ndarray) -> np.ndarray:
        """Segment out cells from background. Returns a binary uint8 array. 0 is background, 1 is cells"""

        def remove_component(img, min_size):
            '''remove connected components smaller than min_size'''
            # find all your connected components
            nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
            sizes = stats[:, -1]  # connectedComponentswithStats yields every separated component with information on each of them, such as size
            img2 = np.zeros((output.shape))  # output_img
            for i in range(1, nb_components):  # for every component in the image, you keep it only if it's above min_size. We start at 1 because 0 is the backgroud which we don't care about.
                if sizes[i] >= min_size:
                    img2[output == i] = 1
            return img2.astype(img.dtype)

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

        var_img = calculateLocalVariance(img, 2)  # calculate Variance Map
        bin_var_img = cv.threshold(var_img, 0.015, 255, cv.THRESH_BINARY)[1]  # Use Otsu to calculate binary threshold and binarize #TODO this isn't otsu
        bin_var_img = bin_var_img.astype(np.uint8)
        bin_var_img[bin_var_img == 0] = 1  # flip background and foreground
        bin_var_img[bin_var_img == 1] = 0
        kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Set kernels for morphological operations and CC
        morph_img = remove_component(bin_var_img, 100)  # Erode->Remove small features->dilate
        morph_img = cv.dilate(morph_img, kernel_dil)
        return morph_img


class Names:
    """Names that are used in filenaming."""
    prefix = '_MMStack_1-Pos'
    # Folders and File prefix for saving analyzed images
    outline = 'Outline'
    binary = 'Binary'
    corrected = 'Corrected'
    analyzed = 'analyzed'

if __name__ == '__main__':
    stitcher = ImageJStitcher(r'C:\Users\backman05\Documents\Fiji.app\ImageJ-win64.exe')
    import sys
    app = QApplication(sys.argv)
    an = SingleWellCoverageAnalyzer(outPath=r'H:\HT29 coverage myo + cele (8-26-19)\48h\BottomLeft_1\Ana',
                               wellPath=r'H:\HT29 coverage myo + cele (8-26-19)\48h\BottomLeft_1',
                               ffcPath=r'H:\HT29 coverage myo + cele (8-26-19)\Flat field corr 48h\BottomLeft_1',
                               centerImgLocation=(0,6),
                               edgeImgLocation=(1,1),
                               darkCount=624,
                               stitcher=stitcher,
                               rotate90=0)
    an.run()