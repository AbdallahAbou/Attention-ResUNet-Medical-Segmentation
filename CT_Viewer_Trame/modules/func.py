import numpy as np
from scipy.ndimage import zoom

def normalize_image(slice_data):
    """Normalize a 2D slice to grayscale [0, 255] as uint8."""
    if np.max(slice_data) == np.min(slice_data):
        return np.zeros(slice_data.shape, dtype=np.uint8)
    normalized = ((slice_data - np.min(slice_data)) /
                  (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
    return normalized

def resample_slice(slice_data, orig_spacing1, orig_spacing2):
    """
    Resample the 2D slice so that its pixels have isotropic spacing.
    orig_spacing1 and orig_spacing2 are the original voxel spacings along the two axes.
    """
    target_spacing = min(orig_spacing1, orig_spacing2)
    zoom_factor = (orig_spacing1 / target_spacing, orig_spacing2 / target_spacing)
    return zoom(slice_data, zoom=zoom_factor, order=3)