import os
import tarfile
import subprocess


# Function to download files from Google Drive
def download_file(url, output_path):
    """
    Downloads a file from Google Drive given its file ID and saves it to the specified output path.

    Parameters:
    - file_id (str): The unique identifier of the file on Google Drive.
    - output_path (str): The path where the downloaded file should be saved.

    Returns:
    - None
    """
    command = ['wget', url, '-O', output_path]
    subprocess.run(command, check=True)
    print(f"Downloaded {output_path}")

# Function to extract tar files
def extract_tar_file(tar_file_path, extract_to):
    """
    Extracts a tar file into the specified directory.

    Parameters:
    - tar_file_path (str): The path to the tar file to be extracted.
    - extract_to (str): The directory where the tar file should be extracted.

    Returns:
    - None
    """
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=extract_to)
        print(f"Extracted {tar_file_path} to {extract_to}")

# Function to download and extract all files
def download_and_prepare_data(download_dir, extract_dir, extract_only=None):
    """
    Downloads and extracts all required data files.

    Parameters:
    - download_dir (str): The directory where the tar files will be downloaded.
    - extract_dir (str): The directory where the files will be extracted.

    Returns:
    - None
    """
    # Define the Google Drive file IDs and output paths
    files_to_download = {
        'hepatic_vessel': {
            'url': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar',
            'output': os.path.join(download_dir, 'hepatic_vessel.tar')
        },
        'liver': {
            'url': 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar',
            'output': os.path.join(download_dir, 'liver.tar')
        }
    }
    if extract_only == True:
        for file_key, file_info in files_to_download.items():
            extract_tar_file(file_info['output'], extract_dir)
    
    # Download and extract the files
    else:
        for file_key, file_info in files_to_download.items():
            download_file(file_info['url'], file_info['output'])
            extract_tar_file(file_info['output'], extract_dir)


