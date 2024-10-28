import torch
from torch.utils.data import Dataset
import nibabel as nib
import os

class MedicalSliceDataset(Dataset):
    def __init__(self, image_dir, label_dir, ids_list):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.ids_list = ids_list
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.slices = self._get_slices()

    def _get_slices(self):
        slices = []
        ids_set = set(self.ids_list)
        idx_r = 0
        for idx in range(len(self.image_files)):
            if self.image_files[idx] in ids_set:
                print('idx_r:', idx_r, 'after condition:', self.image_files[idx])
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
        ids_set = set(self.ids_list)
        image_idx, slice_idx = self.slices[idx]

        image_path = os.path.join(self.image_dir, self.image_files[image_idx])
        label_path = os.path.join(self.label_dir, self.label_files[image_idx])

        # Load the images and labels
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Extract the 2D slice
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]

        # Convert to PyTorch tensors
        image_slice = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)
        label_slice = torch.tensor(label_slice, dtype=torch.long).unsqueeze(0)     # Shape: (1, H, W)

        return image_slice, label_slice