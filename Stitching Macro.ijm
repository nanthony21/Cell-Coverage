root = "K:\\Coverage\\10-24-18_Greta\\";
file_name = "analyzed_MMStack_1-Pos{xxx}_{yyy}.ome.tif";
plate_folder_list = newArray("A2780_0Hour_Plate1", "A2780_0Hour_Plate2", "A2780_0Hour_Plate3", "A2780_0Hour_Plate4");
well_folder_list = newArray("BottomLeft_1", "BottomMid_1", "BottomRight_1", "TopLeft_1", "TopMid_1", "TopRight_1");
grid_size_x = newArray(10,12,12,10,12,12);
grid_size_y = newArray(12,12,12,12,12,12);

outline_pre = "_Outline 2"
binary_pre = "_Binary 2"

for (j = 0; j < plate_folder_list.length; j++){

	for (i = 0 ; i < well_folder_list.length; i++) {
		test = plate_folder_list[j];
		test = well_folder_list[i];
		test = grid_size_x[i];
		test = grid_size_y[i];
		run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x="+grid_size_x[i]+" grid_size_y="+grid_size_y[i]+" tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory=["+root+plate_folder_list[j]+"\\Analyzed\\"+well_folder_list[i]+outline_pre+"] file_names="+file_name+" output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
		run("Enhance Contrast", "saturated=0.35");
		run("Apply LUT");
		run("8-bit");
		saveAs("Jpeg", root + plate_folder_list[j] + "\\analyzed\\"+well_folder_list[i]+outline_pre+".jpg");
		close();
	}
}

for (j = 0; j < plate_folder_list.length; j++){

	for (i = 0 ; i < well_folder_list.length; i++) {
		test = plate_folder_list[j];
		test = well_folder_list[i];
		test = grid_size_x[i];
		test = grid_size_y[i];
		run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x="+grid_size_x[i]+" grid_size_y="+grid_size_y[i]+" tile_overlap=10 first_file_index_x=0 first_file_index_y=0 directory=["+root+plate_folder_list[j]+"\\Analyzed\\"+well_folder_list[i]+binary_pre+"] file_names="+file_name+" output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
		run("8-bit");
		saveAs("Jpeg", root + plate_folder_list[j] + "\\analyzed\\"+well_folder_list[i]+binary_pre+".jpg");
		close();
	}
}
