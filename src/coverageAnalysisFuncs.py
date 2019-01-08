# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:42:05 2018

@author: Scott
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import datetime
import src.imageJStitching
import csv
import os
import os.path as osp
from glob import glob

def analyzeCoverage(root, plate_folder_list, well_folder_list, center_locations, edge_locations, analysisNum:int, dark_count:int, imageJPath:str, ffc_folder, rotate90=0):
    '''
    This is the main routing to call.
    root: Root folder for experiment
    plate_folder_list: Folder names for each plate to analyze
    well_folder_list: Folder and file names for individual well in plate
    center_locations: image index for center image for flatfielding
    edge_locations: image index for edge image for masking
    analysisNum: A number to be added as asuffix to the output files
    dark_count: camera dark counts
    imageJPath: path to imagej.exe
    ffc_folder: Flat Field correction path
    rotate90: the number of times to rotate the images by 90 degrees.
    '''
    # Filename prefix
    file_prefix = '_MMStack_1-Pos'
    # Folders and File prefix for saving analyzed images
    outline_folder = 'Outline'
    binary_folder = 'Binary'
    ff_corr_folder = 'Corrected'
    analyzed_filename = 'analyzed'

    # Number list used for grid error correction
    num_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020']

    stitchingProcess = None
      # Loop through plates
    for plate_folder in plate_folder_list:
        print(plate_folder)
        # Create folder for results
        analyzed_folder = osp.join(root, plate_folder, 'Analyzed_{}'.format(analysisNum))
        if not osp.exists(analyzed_folder):
            os.makedirs(analyzed_folder)
        
        results = {}
        #loop through wells
        for well_index, well_folder in enumerate(well_folder_list):
            print('\t'+well_folder)
            # Mean value of center image is used for flat field correction
            ffc_center = cv.imread(osp.join(ffc_folder, well_folder, 
                                   well_folder + file_prefix + center_locations[well_index][0] +
                                   '_' + center_locations[well_index][1] + '.ome.tif'), -1)
            if ffc_center is None:
                raise ValueError("The flat field image was not found")
            ffc_center -= dark_count
            ffc_mean = ffc_center.mean()
            ffc_std = ffc_center.std()
            
            # FFC edge images are used to threshold the area outside the dish
            ffc_edge = cv.imread(osp.join(ffc_folder, well_folder,
                                   well_folder + file_prefix + edge_locations[well_index][0] +
                                   '_' + edge_locations[well_index][1] + '.ome.tif'), -1)
            ffc_edge -= dark_count
            ffc_thresh = otsu_1d(ffc_edge, wLowOpt = 1)    #Overriding the weight for the low distribution improved segmentation when one population is very narrow and the other is very wide
            
            # FF corrected cell edge images are used to threshold the edge effects from the dish
            cell_edge = cv.imread(osp.join(root, plate_folder, well_folder, 
                                   well_folder + file_prefix + edge_locations[well_index][0] +
                                   '_' + edge_locations[well_index][1] + '.ome.tif'), -1)
            cell_edge -= dark_count
            cell_edge = ((cell_edge * ffc_mean)/ffc_edge).astype(np.uint16)
            cell_thresh = otsu_1d(cell_edge, wLowOpt = 1)
            
            # create save folder
            if not osp.exists(osp.join(analyzed_folder, well_folder + '_' + outline_folder)):
                os.makedirs(osp.join(analyzed_folder, well_folder + '_' + outline_folder))
             # create save folder
            if not osp.exists(osp.join(analyzed_folder, well_folder + '_' + binary_folder)):
                os.makedirs(osp.join(analyzed_folder, well_folder + '_' + binary_folder))  
            if not osp.exists(osp.join(analyzed_folder, well_folder + '_' + ff_corr_folder)):
                os.makedirs(osp.join(analyzed_folder, well_folder + '_' + ff_corr_folder)) 
                
            # Intialize coverage variables
            cell_area = 0
            background_area = 0
            removed_area = 0

            # loop through cell images        
            file_list = glob(osp.join(root, plate_folder, well_folder, well_folder + file_prefix + '*'))
            tileSize = [0,0]
            for ind in range(2):
                tileSize[ind] = max([int(i.split('Pos')[-1].split('.')[0].split('_')[ind]) for i in file_list]) + 1

            for cell_img_loc in file_list:
                # load flat field
                ffc_img_loc = osp.join(ffc_folder, well_folder, well_folder + cell_img_loc.split(well_folder)[2])
                ffc_img = cv.imread(ffc_img_loc, -1)
                ffc_img -= dark_count
                ffc_img = np.rot90(ffc_img, rotate90)
                
                # load cell
                cell_img = cv.imread(cell_img_loc, -1)
                cell_img -= dark_count
                cell_img = np.rot90(cell_img, rotate90)
            
                # calculated corrected image
                corr_img = ((cell_img * ffc_mean)/ffc_img).astype(np.uint16)
                # Data Standardization
                standard_img = (corr_img-ffc_mean)/ffc_std
                
                # Determine mask to remove dark regions and regions outside of dish
                ffc_mask = cv.threshold(ffc_img, ffc_thresh, 65535, cv.THRESH_BINARY)[1]
                corr_mask = cv.threshold(cell_img, cell_thresh, 65535, cv.THRESH_BINARY)[1]
                background_mask = ffc_mask * corr_mask
                
                # Segment out cells from background
                [outline, morph_img] = analyze_img(standard_img, background_mask)
                
                # Keep track of areas to calculate coverage
                removed_area += np.count_nonzero(morph_img == 2)
                background_area += np.count_nonzero(morph_img == 1)
                cell_area += np.count_nonzero(morph_img == 0)
                
                # Write images to file
                cv.imwrite(osp.join(analyzed_folder, well_folder + '_' + binary_folder, analyzed_filename + cell_img_loc.split(well_folder)[2]), morph_img)
                cv.imwrite(osp.join(analyzed_folder,well_folder + '_' + ff_corr_folder, analyzed_filename + cell_img_loc.split(well_folder)[2]), corr_img)
                # Add segmentation outline to corrected image
                corr_img[outline.astype(bool)] = 0
                cv.imwrite(osp.join(analyzed_folder, well_folder + '_' + outline_folder, analyzed_filename + cell_img_loc.split(well_folder)[2]), corr_img)
            
            # Output and save coverage numbers
            print('The coverage is ', 100*cell_area/(cell_area + background_area), ' %')
            results[well_folder] = 100*cell_area/(cell_area + background_area)
            imjProcess = src.imageJStitching.stitchCoverage(root, plate_folder, well_folder, tileSize, analyzed_folder, outline_folder, binary_folder, imageJPath, stitchingProcess)
                # Initialize txt file to save coverage numbers
        with open(osp.join(analyzed_folder, 'Coverage Percentage Results.csv'),'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(datetime.datetime.now())])
            writer.writerow([plate_folder])
            writer.writerow(['FlatField Folder: {}'.format(ffc_folder)])
            writer.writerow(list(results.keys())) #Well folder names
            writer.writerow(list(results.values()))
            
    imjProcess.communicate() #wait for the last imagej process to finish.


def remove_component(img, min_size):
    '''remove connected components smaller than min_size'''
    #find all your connected components
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    
    #connectedComponentswithStats yields every separated component with information on each of them, such as size
    sizes = stats[:, -1]

    #output_img
    img2 = np.zeros((output.shape))
    
    #for every component in the image, you keep it only if it's above min_size. We start at 1 because 0 is the backgroud which we don't care about.
    for i in range(1, nb_components):
        if sizes[i] >= min_size:
            img2[output == i] = 1
            
    return img2.astype(img.dtype)
     
    
def dist_mask(dist):
    ''' Create a circular mask with a radius of `dist`.''' 
    # Initialize output make
    output_mask = np.ones([dist*2 + 1, dist*2 + 1], dtype=np.uint16)
    
    #Create distance map
    output_mask[dist, dist] = 0
    dist_map = ndimage.distance_transform_edt(output_mask)
    
    # Turn distance map into binary mask
    output_mask[dist_map>dist] = 0
    output_mask[dist_map<=dist] = 1
    return output_mask

def var_map(img, dist):
    ''' var_map creates a map of the spatial variance 
    in a neighborhood of size dist at pixels in img'''
    img = img.astype(np.float32)
    mask = dist_mask(dist)
    mask = mask / mask.sum() #Normalize the mask
    mean = cv.filter2D(img, cv.CV_32F, mask)
    sqrMean = cv.filter2D(img*img, cv.CV_32F, mask)
    return (sqrMean - mean*mean)

def otsu_1d(img, wLowOpt = None, wHighOpt = None):
    ''' calculates the threshold for binarization using Otsu's method.
    The weights for the low and high distribution can be overridden using the optional arguments.
    '''
    flat_img = img.flatten()
    var_b_max = 0
    bin_index = 0
    
    num_bins = 100 # Can reduce num_bins to speed code, but reduce accuracy of threshold
    img_min = np.percentile(flat_img,1)
    img_max = np.percentile(flat_img,99)
    for bin_val in np.linspace(img_min, img_max, num_bins, endpoint = False):
        # segment data based on bin
        gLow = flat_img[flat_img <= bin_val]
        gHigh = flat_img[flat_img > bin_val]
        
        # determine weights of each bin
        wLow = gLow.size/flat_img.size if (wLowOpt is None) else wLowOpt
        wHigh = gHigh.size/flat_img.size if (wHighOpt is None) else wLowOpt
        
        # maximize inter-class variance
        var_b = wLow * wHigh * (gLow.mean() - gHigh.mean())**2
        [var_b_max, bin_index] = [var_b, bin_val] if var_b > var_b_max else [var_b_max, bin_index]
    return bin_index


# Segment out cells from background
def analyze_img(img, *mask):
    if len(mask) == 0:
        mask = np.ones(img.shape).astype(np.uint16)
    else:
        mask = mask[0]

#    # Data Standardization
#    img = (img-img.mean())/img.std()

    # calculate Variance Map
    var_img = var_map(img, 2)
    
    # Use Otsu to calculate binary threshold and binarize
    bin_var_img = cv.threshold(var_img, 0.015, 65535, cv.THRESH_BINARY)[1]

    # flip background and foreground
    bin_var_img[bin_var_img == 0] = 1
    bin_var_img[bin_var_img == 65535] = 0
    bin_var_img[~mask.astype(bool)] = 2

    # Set kernels for morphological operations and CC
    kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    min_size = 100

    # Erode->Remove small features->dilate
    morph_img = remove_component(bin_var_img, min_size)

    morph_img[~mask.astype(bool)] = 2
    morph_img = cv.dilate(morph_img, kernel_dil)

    #binary outline for overlay
    outline = cv.dilate(cv.Canny(morph_img.astype(np.uint8), 0, 1), cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)))
    
    return [outline, morph_img]

    # Main Code
if __name__ == '__main__':
    file1 = 'K:\\Coverage\\10-2-18 and 10-3-18\\corr_trans_10-3-2018_2\\corr_trans_10-3-2018_2_MMStack_3-Pos_005_018.ome.tif'
    file1f = r'K:\Coverage\10-2-18 and 10-3-18\Treference10-3-2018_2\Treference10-3-2018_2_MMStack_3-Pos_005_018.ome.tif'
    file2 = 'K:\\Coverage\\10-2-18 and 10-3-18\\corr_trans_10-3-2018_2\\corr_trans_10-3-2018_2_MMStack_3-Pos_008_020.ome.tif'
    file2f = r'K:\Coverage\10-2-18 and 10-3-18\Treference10-3-2018_2\Treference10-3-2018_2_MMStack_3-Pos_008_020.ome.tif'
    file3 = 'K:\\Coverage\\10-2-18 and 10-3-18\\corr_trans_10-2-2018_2\\corr_trans_10-2-2018_2_MMStack_3-Pos_005_018.ome.tif'
    file4 = 'K:\\Coverage\\10-2-18 and 10-3-18\\corr_trans_10-2-2018_2\\corr_trans_10-2-2018_2_MMStack_3-Pos_003_003.ome.tif'
    file5 = 'H:\\Cell Coverage\\cellCvgSc\\corrT_0\\corrT_1_MMStack_4-Pos_009_016.ome.tif'
    file5f = r'H:\Cell Coverage\cellCvgSc\noneT_1\noneT_1_MMStack_4-Pos_009_016.ome.tif'
    file6 = 'H:\\Cell Coverage\\cellCvgSc\\corrT_0\\corrT_1_MMStack_4-Pos_006_005.ome.tif'
    file6f = r'H:\Cell Coverage\cellCvgSc\noneT_1\noneT_1_MMStack_4-Pos_006_005.ome.tif'
    file7 = 'K:\\Coverage\\10-23-18\\A2780Plate1\\Analyzed\\TopLeft_1_Corrected\\analyzed_MMStack_1-Pos003_007.ome.tif'
    file8 = r'K:\Coverage\10-24-18_Greta\A2780_0Hour_Plate1\Analyzed\TopLeft_1_Corrected\analyzed_MMStack_1-Pos003_007.ome.tif'
    file8f = r'K:\Coverage\10-24-18_Greta\FlatField_0Hour\TopLeft_1\TopLeft_1_MMStack_1-Pos003_007.ome.tif'
    file9 = r'K:\Coverage\10-23-18\A2780Plate1\Analyzed\BottomLeft_1_Corrected\analyzed_MMStack_1-Pos007_005.ome.tif'
    file10 = r'K:\Coverage\10-23-18_Jane\10-24-18\A2780_Plate1\Analyzed\TopMid_1_Corrected\analyzed_MMStack_1-Pos002_004.ome.tif'
    file10f = r'K:\Coverage\10-23-18_Jane\10-24-18\FlatField\TopMid_1\TopMid_1_MMStack_1-Pos002_004.ome.tif'
    file11 = r'K:\Coverage\10-24-18_Greta\A2780_24Hour_Plate1\Analyzed\TopLeft_1_Corrected\analyzed_MMStack_1-Pos001_004.ome.tif'
    file11f = r'K:\Coverage\10-24-18_Greta\FlatField_24Hour\TopLeft_1\TopLeft_1_MMStack_1-Pos001_004.ome.tif'
    
    file12 = r'K:\Coverage\10-24-18_Greta\A2780_48Hour_Plate4\Analyzed\TopLeft_1_Corrected\analyzed_MMStack_1-Pos007_005.ome.tif'
    file12f = r'K:\Coverage\10-24-18_Greta\FlatField_48Hour\TopLeft_1\TopLeft_1_MMStack_1-Pos007_005.ome.tif'
    file_list = [file1, file6, file8, file10, file11, file12]
    ff_list = [file1f, file6f, file8f, file10f, file11f, file12f]
#    file_list = [file5]
#    ff_list = [file5f]
#    plt.imshow(cv.imread(file, -1))
#    file_list = [file7]
    
    for ind in range(len(file_list)):
        img = cv.imread(file_list[ind], -1)
        img_sta = cv.imread(ff_list[ind], -1)
        corr_img = (img-img_sta.mean())/img_sta.std()
        [outline, morph_img] = analyze_img(corr_img)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        img[outline.astype(bool)] = 0
        cv.imwrite(r'C:\Users\Scott\Documents\cell-coverage\presentation images\example'+str(ind)+'.tif', img)
        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        
    
    
#    corr_img = img
#    corr_img[outline.astype(bool)] = 0
#    my_cmap = cm.Purples
#    my_cmap.set_under('k', alpha=0)

#    n, hist_bins, patches = plt.hist(var_img.flatten(),bins=400,
#                                     density=True)
    #
    #n, hist_bins, patches = plt.hist(blur_var1.flatten(), bins=400)
#    plt.figure()
#    plt.imshow(corr_img, cmap='gray')
    
    
    # display images
#    r = [0, 256, 512, 768, 1024]
#    for x_r in range(4):
#        for y_r in range(4):
#            fig = plt.figure()
#            plt.subplot(1, 2, 1)
#            plt.imshow(img[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap='gray')
#            plt.subplot(1, 2, 2)
#            plt.imshow(img[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap='gray')
#            plt.imshow(outline[r[x_r]:r[x_r+1]+1, r[y_r]:r[y_r+1]+1], cmap=my_cmap, clim=[0.9, 1])
#            while plt.fignum_exists(fig.number):
#                fig.canvas.flush_events()
