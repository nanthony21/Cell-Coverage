from typing import Tuple
from glob import glob
import os
import cv2

def loadImage(location: Tuple[int, int], path: str):
    fileName = f'*{Names.prefix}{location[0]:03d}_{location[1]:03d}.ome.tif'
    searchPattern = os.path.join(path, fileName)
    try:
        fullPath = glob(searchPattern)[0]
        fileName = os.path.split(fullPath)[-1]
    except IndexError:
        raise OSError(f"Couldn't find a file matching pattern {searchPattern}")
    img = cv2.imread(fullPath, -1)
    if img is None:
        raise OSError(f"The image, {fullPath}, was not found")
    return img, fileName

def getLocationFromFileName(fileName: str) -> Tuple[int, int]:
    fileName = fileName.split(Names.prefix)[1] #Remove first part of name
    fileName = fileName.split('.')[0] #remove the extension
    location = tuple(int(i) for i in fileName.split('_'))
    assert len(location) == 2
    return location


class Names:
    """Names that are used in filenaming."""
    prefix = '_MMStack_1-Pos'
    # Folders and File prefix for saving analyzed images
    outline = 'Outline'
    binary = 'Binary'
    corrected = 'Corrected'
    analyzed = 'analyzed'