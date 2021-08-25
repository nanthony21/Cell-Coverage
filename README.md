# Cell Coverage Analysis
Full dish maps of adherent cells acquired in Micro-Manager can be analyzed by this program to provide accurate measurements of coverage (Area_Covered_By_Cells / Total_Area)

After initial preprocessing of the images, the analysis uses a series of two convolutions to generate an image of the local variance of the original image. This local variance provides a good contrast between cell / non-cell regions which can be easily binarized using Otsu thresholding.
