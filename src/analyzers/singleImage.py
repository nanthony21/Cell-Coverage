import cv2 as cv
import numpy as np
from scipy import ndimage
from typing import Tuple
import matplotlib.pyplot as plt

# TODO remove small components of mask

class SingleImageAnalyzer:
    def __init__(self, darkCount: int, rotate90: int, image: np.ndarray, flatField: np.ndarray, debug: bool = False):
        self.darkCount = darkCount
        self.rot90 = rotate90
        self.img = image
        self.ffc = flatField
        self.img -= self.darkCount
        self.ffc -= self.darkCount
        self.img = np.rot90(self.img, self.rot90)
        self.ffc = np.rot90(self.ffc, self.rot90)
        self.debug = debug
        self.figs = []

    def run(self, ffcMean: float, ffcStd: float, cellThresh: float, ffcThresh: float):
        std = self.getStandardizedImage(ffcMean, ffcStd)
        backgroundMask = self.calculateBackground(cellThresh, ffcThresh)
        morph_img = self.analyze_img(std, backgroundMask)
        # Keep track of areas to calculate coverage
        removed_area = np.count_nonzero(morph_img == 2)
        background_area = np.count_nonzero(morph_img == 1)
        cell_area = np.count_nonzero(morph_img == 0)
        if self.debug:
            self.figs.append(plt.figure())
            plt.imshow(std, clim=[np.percentile(std, 1), np.percentile(std, 99)],
                       cmap='gray')
            plt.imshow(np.logical_not(backgroundMask), alpha=0.6, clim=[0, 1], cmap='Reds')
            plt.title("Red = Ignored area")
            plt.show(block=False)
            while all([plt.fignum_exists(fig.number) for fig in self.figs]):
                [fig.canvas.flush_events() for fig in self.figs]
            plt.close('all')
            self.figs = []
        return std, backgroundMask, morph_img, removed_area, background_area, cell_area

    def getStandardizedImage(self, ffcMean, ffcStd):
        corrected = ((self.img * ffcMean) / self.ffc).astype(np.uint16)
        standardized = (corrected - ffcMean) / ffcStd  # Data Standardization
        return standardized

    def calculateBackground(self, imageThreshold: float, flatFieldThreshold: float):
        # Determine mask to remove dark regions and regions outside of dish
        ffc_mask = cv.threshold(self.ffc, flatFieldThreshold, 65535, cv.THRESH_BINARY)[1]
        corr_mask = cv.threshold(self.img, imageThreshold, 65535, cv.THRESH_BINARY)[1]
        backgroundMask = ffc_mask * corr_mask #TODO this seems to work but i don't know why. shouldn't it be an OR operation?
        if self.debug:
            fig,ax = plt.subplots()
            im = ax.imshow(ffc_mask, clim=[0, 1])
            fig.colorbar(im)
            fig.suptitle("FFC Mask. 0=Ignored Area")
            fig2, ax2 = plt.subplots()
            im = ax2.imshow(corr_mask, clim=[0, 1])
            fig2.colorbar(im)
            fig2.suptitle("Img Mask. 0=Ignored Area")
            fig3, ax3 = plt.subplots()
            n, bins, patches = ax3.hist(self.img.flatten(), bins=100)
            ax3.vlines(imageThreshold, 0, max(n))
            fig3.suptitle("Image Hist")
            self.figs += [fig, fig2, fig3]
        return backgroundMask

    def analyze_img(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
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

        variance = calculateLocalVariance(img, 2)  # calculate Variance Map
        bin_var_img = cv.threshold(variance, 0.015, 255, cv.THRESH_BINARY)[1]  # Use Otsu to calculate binary threshold and binarize #TODO this isn't otsu
        bin_var_img = bin_var_img.astype(np.uint8)
        bin_var_img[bin_var_img == 0] = 1  # flip background and foreground
        bin_var_img[bin_var_img == 255] = 0
        bin_var_img[~mask.astype(bool)] = 2  # Use a third value to show regions that are considered to be outside the dish
        kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Set kernels for morphological operations and CC
        morph_img = remove_component(bin_var_img, 100)  # Erode->Remove small features->dilate
        morph_img[~mask.astype(bool)] = 2  # Use a third value to show regions that are considered to be outside the dish
        morph_img = cv.dilate(morph_img, kernel_dil)
        if self.debug:
            self.figs.append(plt.figure())
            plt.imshow(variance, clim=(0, np.percentile(variance, 99)))
            plt.colorbar()
            plt.title("Variance")
            self.figs.append(plt.figure())
            plt.imshow(cv.dilate(bin_var_img, kernel_dil))
            plt.title("Pre component removal. 0=Cell, 1=Blank, 2=Ignored")
            plt.colorbar()
            self.figs.append(plt.figure())
            plt.imshow(morph_img)
            plt.title("Post component removal. 0=Cell, 1=Blank, 2=Ignored")
            plt.colorbar()
        return morph_img
