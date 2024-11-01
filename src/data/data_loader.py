import torch
from torch.utils.data import Dataset
import nibabel as nib
import os

class MedicalSliceDataset(Dataset):
    def __init__(self, image_dir, label_dir, ids_list, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.ids_list = ids_list
        self.transform = transform
        self.image_files = sorted([file for file in os.listdir(image_dir) if not file.startswith('.') and file.endswith(".nii.gz")])
        self.label_files = sorted([file for file in os.listdir(label_dir) if not file.startswith('.') and file.endswith(".nii.gz")])
        self.slices = self._get_slices()

    def _get_slices(self):
        slices = []
        ids_set = set(self.ids_list)
        idx_r = 0
        for idx in range(len(self.image_files)):
            if self.image_files[idx] in ids_set:
                image_path = os.path.join(self.image_dir, self.image_files[idx])
                image = nib.load(image_path).get_fdata()
                num_slices = image.shape[2]  
                for slice_idx in range(num_slices):
                    slices.append((idx, slice_idx))
                idx_r = idx_r + 1
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        image_idx, slice_idx = self.slices[idx]

        image_path = os.path.join(self.image_dir, self.image_files[image_idx])
        label_path = os.path.join(self.label_dir, self.label_files[image_idx])

        # Load the images and labels
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Extract the 2D slice
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]

        # Normalize the image slice
        image_slice = self.normalize_ct_image(image_slice)

        # Convert to PyTorch tensors
        image_slice = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
        label_slice = torch.tensor(label_slice, dtype=torch.long).unsqueeze(0)     # Shape: (1, H, W)

        if self.transform:
            image_slice = self.transform(image_slice)

        return image_slice, label_slice
    

    @staticmethod
    def normalize_ct_image(image_slice):
        """
        Normalizes a CT image slice by rescaling its intensity values to [0, 1].
        """
        min_intensity = image_slice.min()
        max_intensity = image_slice.max()
        # Avoid division by zero
        if max_intensity - min_intensity == 0:
            normalized_slice = image_slice - min_intensity
        else:
            normalized_slice = (image_slice - min_intensity) / (max_intensity - min_intensity)
        return normalized_slice