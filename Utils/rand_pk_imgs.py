import os
import random
import shutil

def randomly_pick_files_png(source_dir, label_dir, dest_dir, dest_label_dir, n):
    """
    Randomly picks n PNG files from the source directory and their corresponding ground truth PNG files,
    then moves them to the destination directory.

    :param source_dir: The directory containing the PNG files for features and ground truths.
    :param dest_dir: The directory where the selected files will be moved.
    :param n: The number of files to randomly pick.
    """
    # List all PNG files in the source directory
    files = os.listdir(source_dir)
    
    # Filter out non-PNG files
    png_files = [file for file in files if file.endswith('.png')]

    # Assuming each feature file has a corresponding ground truth file with the same name
    # Randomly pick n PNG files (considering pairs)
    selected_files = random.sample(png_files, n)

    # Move the selected PNG files and their corresponding ground truth files
    for file in selected_files:
        # Construct the full path for the file and its ground truth
        file_path = os.path.join(source_dir, file)
        ground_truth_file_path = os.path.join(label_dir, file)  # Assuming ground truth files have the same name

        # Construct the destination paths
        dest_file_path = os.path.join(dest_dir, file)
        dest_ground_truth_path = os.path.join(dest_label_dir, file)

        # Move the files
        shutil.copy(file_path, dest_file_path)
        shutil.copy(ground_truth_file_path, dest_ground_truth_path)


feature_directory = './Data/val/blur/blur/'
label_directory = './Data/val/sharp/sharp/'
dest_feature_directory = './Data/pred/blur/'
dest_label_directory = './Data/pred/sharp/'
number_of_files = 50  # Number of files to randomly pick

randomly_pick_files_png(feature_directory,label_directory,dest_feature_directory,dest_label_directory,number_of_files)