import numpy as np

class Stitcher:
    """This class helps to stitch together a grid of images into a single large image."""
    def __init__(self, imageGrig: np.ndarray, overlap: float, invertX: bool = False, invertY: bool = False):
        self.images = imageGrig
        self.overlap = overlap
        self.invX = invertX
        self.invY = invertY

    def stitch(self) -> np.ndarray:
        s = 1-self.overlap
        imShape = self.images[0,0].shape
        output = np.zeros((int(imShape[0]*(self.images.shape[0]-1)*s+imShape[0]+1), int(imShape[1]*(self.images.shape[1]-1)*s+1+imShape[1])), dtype=self.images[0,0].dtype) #We add one extra pixel here in case of rounding errors with s.
        for i in range(self.images.shape[0]):
            for j in range(self.images.shape[1]):
                img = self.images[i, j]
                vpos = int(imShape[0]*(self.images.shape[0]-i-1)*s) if self.invY else int(imShape[0]*i*s)
                hpos = int(imShape[1]*(self.images.shape[1]-j-1)*s) if self.invX else int(imShape[1]*j*s)
                output[vpos:vpos+imShape[0], hpos:hpos+imShape[1]] = img
        if np.all(output[-1,:] == 0):
            output = output[:-1,:]
        if np.all(output[:,-1] == 0):
            output = output[:,:-1]
        return output

if __name__ == '__main__':
    from cellcoverage.utility import loadImage, getLocationFromFileName
    from glob import glob
    import os
    import matplotlib.pyplot as plt
    wdir = r'H:\HT29 coverage myokine (8-7-19)\Coverageplate48h\BottomMid_1'
    files = glob(os.path.join(wdir, '*MMStack*.tif'))
    rot90 = 1
    locs = [getLocationFromFileName(f) for f in files]
    x,y = zip(*locs)
    images = np.zeros((max(x)+1, max(y)+1), dtype=object)
    print("Start Loading")
    for loc in locs:
        im = loadImage(loc, wdir)[0]
        im = np.rot90(im, rot90)
        images[loc[0],loc[1]] = im
    print("Done loading")
    s = Stitcher(images, 0.1, invertY=True)
    out = s.stitch()
    plt.imshow(out)
    plt.show()
    pass