import os

def rename_files(folder_path):
    """
    Renames all files in the specified folder by removing 'corrected' from their names.
    Only affects files that end with 'corrected.png'.

    :param folder_path: Path to the folder containing the files to be renamed.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith("corrected.png"):
            new_filename = filename.replace("corrected", "")
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            #print(f"Renamed {filename} to {new_filename}")

# Example usage
folder_path = '../Data/train/colorful/'  # Replace with your actual folder path
rename_files(folder_path)
