/*
root = "K:\\Coverage\\10-17-18\\";
file_name = "analyzed_MMStack_Pos_{xxx}_{yyy}.ome.tif";
plate_folder_list = newArray("0HourA2780_1", "0HourA2780_plate2");
//plate_folder_list = newArray("0HourA2780_plate2");
well_folder_list = newArray("BottomLeft_1", "BottomMid_1", "BottomRight_1", "TopLeft_1", "TopMid_1", "TopRight_1");
grid_size_x = newArray(13,14,12,11,14,12);
grid_size_y = newArray(14,12,13,14,14,13);
*/
/*
root = "K:\\Coverage\\10-18-18\\";
file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif";
plate_folder_list = newArray("24HourA2780_plate1", "24HourA2780_plate2");
well_folder_list = newArray("BottomLeft_1", "BottomMid_1", "BottomRight_1", "TopLeft_1", "TopMid_1", "TopRight_1");
grid_size_x = newArray(13,14,12,11,14,12);
grid_size_y = newArray(14,12,13,14,14,13);
*/

root = "K:\\Coverage\\10-19-18\\";
file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif";
plate_folder_list = newArray("48HourA2780_plate1", "48HourA2780_plate2");
//well_folder_list = newArray("BottomLeft_1", "BottomMid_1", "BottomRight_1", "TopLeft_1", "TopMid_1", "TopRight_1");
well_folder_list = newArray("TopMid_1");
//grid_size_x = newArray(11,14,14,11,14,13);
//grid_size_y = newArray(12,12,12,12,15,14);
grid_size_x = newArray("14");
grid_size_y = newArray("14");

outline_pre = "_Outline_Standardized_Single"
binary_pre = "_Binary_Standardized_Single"

for (j = 1; j <= 1; j++){

	for (i = 0 ; i <= 0; i++) {
		test = plate_folder_list[j];
		test = well_folder_list[i];
		test = grid_size_x[i];
		test = grid_size_y[i];
		run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x="+grid_size_x[i]+" grid_size_y="+grid_size_y[i]+" tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory="+root+plate_folder_list[j]+"\\Analyzed\\"+well_folder_list[i]+outline_pre+" file_names="+file_name+" output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
		run("Enhance Contrast", "saturated=0.35");
		run("Apply LUT");
		run("8-bit");
		saveAs("Jpeg", root + plate_folder_list[j] + "\\analyzed\\"+well_folder_list[i]+outline_pre+".jpg");
		close();
	}
}

for (j = 0; j <= 1; j++){

	for (i = 0 ; i <= 0; i++) {
		test = plate_folder_list[j];
		test = well_folder_list[i];
		test = grid_size_x[i];
		test = grid_size_y[i];
		run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x="+grid_size_x[i]+" grid_size_y="+grid_size_y[i]+" tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory="+root+plate_folder_list[j]+"\\Analyzed\\"+well_folder_list[i]+binary_pre+" file_names="+file_name+" output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
		run("8-bit");
		saveAs("Jpeg", root + plate_folder_list[j] + "\\analyzed\\"+well_folder_list[i]+binary_pre+".jpg");
		close();
	}
}
