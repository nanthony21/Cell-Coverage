from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QSpinBox, QDoubleSpinBox, QGridLayout, QLabel, QPushButton, QCheckBox
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from src.analyzers.singleWell import SingleWellCoverageAnalyzer, OutputOptions
import numpy as np
import cv2

class ParameterTester(SingleWellCoverageAnalyzer):
    def __init__(self,  outPath: str, wellPath: str, ffcPath: str, darkCount: int, rotate90: int = 0):
        super().__init__(outPath, wellPath, ffcPath, darkCount, rotate90, outputOption=OutputOptions.Nothing)
        self.window = QWidget()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.diameterSpinBox = QSpinBox(self.window)
        self.thresholdSpinBox = QDoubleSpinBox(self.window)
        self.componentSizeSpinBox = QSpinBox(self.window)
        self.varShowButton = QPushButton("Show Variance")
        self.varShowButton.released.connect(self.swapVarDisplay)
        self.refreshButton = QPushButton("Refresh")
        self.refreshButton.released.connect(self.refreshAll)
        self.binaryCheckBox = QCheckBox("ShowBinary: ", self.window)
        self.binaryCheckBox.stateChanged.connect(self.binaryCheckChanged)
        self.dilateCheckBox = QCheckBox("Dilate: ", self.window)
        self.dilateCheckBox.setCheckState(2)

        self.diameterSpinBox.setValue(5)
        self.componentSizeSpinBox.setMaximum(1000)
        self.componentSizeSpinBox.setValue(100)
        self.thresholdSpinBox.setValue(0.01)

        l = QGridLayout()
        l.addWidget(self.canvas, 0, 0, 8, 8)
        label = QLabel("Kernel Diameter:", self.window)
        label.setAlignment(QtCore.Qt.AlignRight)
        l.addWidget(label, 8, 0)
        l.addWidget(self.diameterSpinBox, 8, 1)
        label = QLabel("Variance Threshold:", self.window)
        label.setAlignment(QtCore.Qt.AlignRight)
        l.addWidget(label, 8, 2)
        l.addWidget(self.thresholdSpinBox, 8, 3)
        label = QLabel("Min Component Size:", self.window)
        label.setAlignment(QtCore.Qt.AlignRight)
        l.addWidget(label, 8, 4)
        l.addWidget(self.componentSizeSpinBox, 8, 5)
        l.addWidget(self.varShowButton, 8, 6)
        l.addWidget(self.refreshButton, 8, 7)
        l.addWidget(self.binaryCheckBox, 9, 6)
        l.addWidget(self.dilateCheckBox, 9, 7)
        l.addWidget(NavigationToolbar2QT(self.canvas, self.window), 9, 0, 1, 6)
        self.window.setLayout(l)

    def run(self, mask: np.ndarray):
        self.stdImg = self._getStandardizedImg(mask)
        crop = self._getRectangle()
        self.stdImg = self.stdImg[crop[1], crop[0]]
        self.refreshVariance()
        self.refreshBinary()

        self.stdAxIm = self.ax.imshow(self.stdImg, cmap='gray', clim=[-1,1])
        self.varAxIm = self.ax.imshow(self.var, cmap='gray')
        self.varAxIm.set_visible(False)
        self.binOverlay = self.ax.imshow(self.binary, alpha=.2, clim=[0,1])
        self.binOverlay.set_visible(False)
        self.refreshPlot()
        self.window.show()

    def _getRectangle(self):
        from matplotlib.widgets import RectangleSelector
        downSampleScaler = max([self.img.shape[0] // 1024, self.img.shape[1] // 1024])
        dsimg = self.img[::downSampleScaler, ::downSampleScaler]  # The Roi.fromVerts process takes forever on a huge stitched image. downsample to 1024,1024 then upsample back up.
        slices = [0,0]
        def finish(eclick, erelease):
            #save the selected slices. Use the downsample scaler to make sure the coordinates are scaled properly
            slices[0] = slice(int(eclick.xdata * downSampleScaler), int(erelease.xdata * downSampleScaler))
            slices[1] = slice(int(eclick.ydata * downSampleScaler), int(erelease.ydata * downSampleScaler))
        fig, ax = plt.subplots()
        fig.suptitle("Select the analysis region. Close to proceed.")
        im = ax.imshow(dsimg, clim=[np.percentile(dsimg, 1), np.percentile(dsimg, 99)], cmap='gray')
        a = RectangleSelector(ax, finish)
        fig.show()
        while plt.fignum_exists(fig.number):
            fig.canvas.flush_events()
        return slices

    def binaryCheckChanged(self, checked):
        if checked:
            self.binOverlay.set_visible(True)
        else:
            self.binOverlay.set_visible(False)
        self.canvas.draw_idle()

    def swapVarDisplay(self):
        if self.varShowButton.text() == 'Show Variance':
            self.varShowButton.setText("Show Raw")
            self.varAxIm.set_visible(True)
        else:
            self.varShowButton.setText("Show Variance")
            self.varAxIm.set_visible(False)
        self.canvas.draw_idle()

    def refreshVariance(self):
        varDiameter = self.diameterSpinBox.value()
        self.var = self.calculateLocalVariance(self.stdImg, varDiameter)

    def refreshBinary(self):
        thresh = self.thresholdSpinBox.value()
        compSize = self.componentSizeSpinBox.value()
        dilate = self.dilateCheckBox.checkState()
        diam = self.diameterSpinBox.value()
        self.binary = self.var <= thresh
        self.binary = self.removeSmallComponents(self.binary, compSize)
        if dilate:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
            self.binary = cv2.dilate(self.binary.astype(np.uint8), kernel)

    def refreshPlot(self):
        self.varAxIm.set_data(self.var)
        self.varAxIm.set_clim([np.percentile(self.var, 1), np.percentile(self.var, 99)])
        self.binOverlay.set_data(self.binary)
        self.canvas.draw_idle()

    def refreshAll(self):
        self.refreshVariance()
        self.refreshBinary()
        self.refreshPlot()

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    an = ParameterTester(outPath=r'H:\HT29 coverage for Nick (8-20-19)\HT29 coverage48h (8-20-19)\Low conf\BottomLeft_1\Ana',
                                    wellPath=r'H:\HT29 coverage for Nick (8-20-19)\HT29 coverage48h (8-20-19)\Low conf\BottomLeft_1',
                                    ffcPath=r'H:\HT29 coverage for Nick (8-20-19)\HT29 coverage48h (8-20-19)\Flat field corr\BottomLeft_1',
                                    darkCount=624,
                                    rotate90=2)
    mask = an.selectAnalysisArea()
    an.run(mask)
    sys.exit(app.exec_())