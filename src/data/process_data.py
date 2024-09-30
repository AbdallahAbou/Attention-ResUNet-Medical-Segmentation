import nibabel as nib
import numpy as np
import os

def load_nifti_file(file_path):
    """
    Loads a NIfTI file (.nii or .nii.gz format).

    Parameters:
    - file_path: Path to the NIfTI file to be loaded.

    Returns:
    - A NumPy array containing the image data from the NIfTI file.
    """
    # Load the NIfTI file using nibabel
    nii_image = nib.load(file_path)
    
    # Extract image data as a NumPy array
    image_data = nii_image.get_fdata()

    return image_data


def normalize_ct_image(image_data):
    """
    Normalizes a CT image by rescaling its intensity values.

    Parameters:
    - image_data: A NumPy array containing the CT image data to be normalized.

    Returns:
    - A NumPy array with normalized image data where values are scaled between 0 and 1.
    """
    # Get the minimum and maximum intensity values in the image
    min_intensity = np.min(image_data)
    max_intensity = np.max(image_data)
    
    # Normalize the image data to the range [0, 1]
    normalized_data = (image_data - min_intensity) / (max_intensity - min_intensity)

    return normalized_data


def process_files(input_dir, output_dir):
    """
    Processes all NIfTI files in a given input directory: normalizes and saves them to an output directory.

    Parameters:
    - input_dir: The directory containing input .nii.gz files.
    - output_dir: The directory where processed (normalized) files will be saved.
    """
    # List all files in the input directory
    files = os.listdir(input_dir)
    count = 0
    for file in files:
        if not file.startswith('._') and file.endswith(".nii.gz"): 
            count = count + 1
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load and normalize the image
            image_data = load_nifti_file(input_path)
            normalized_data = normalize_ct_image(image_data)

            # Save normalized data
            nib.save(nib.Nifti1Image(normalized_data, affine=None), output_path)
            print(f"Processed and saved: {output_path} {count}/{len(files)}")

def process_all_data(input_dirs, output_dirs):

    # Process each input directory and save to corresponding output directory
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        process_files(input_dir, output_dir)
