import os
import torch
from torch.utils.data import Dataset
import nibabel as nib

class MedicalDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('._') and f.endswith('.nii.gz')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if not f.startswith('._') and f.endswith('.nii.gz')])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label as NumPy arrays
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        
        # Convert the data to tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Return the image, label, and number of slices
        return image, label, image.shape[2]


def custom_collate_fn(batch):
    all_slices = []
    all_labels = []
    
    # Iterate through each image and label in the batch
    for image, label, slices in batch:
        # Process each slice from the 3D image and its corresponding label
        for slice_idx in range(slices):
            all_slices.append(image[:, :, slice_idx].unsqueeze(0))
            all_labels.append(label[:, :, slice_idx].unsqueeze(0))
    # Stack slices into a tensor
    all_slices = torch.stack(all_slices)
    all_labels = torch.stack(all_labels)

    return all_slices, all_labels