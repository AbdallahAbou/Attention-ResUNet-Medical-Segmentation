import torch
import os
from src.data.data_loader import MedicalDataset, custom_collate_fn
from torch.utils.data import DataLoader

def preload(image_dir, label_dir, output_tensor_path):
    dataset = MedicalDataset(image_dir=image_dir, label_dir=label_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    all_images = []
    all_labels = []
    
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
    
    # Stack all images and labels into single tensors
    stacked_images = torch.cat(all_images, dim=0)
    stacked_labels = torch.cat(all_labels, dim=0)
    
    # Save tensors to disk
    torch.save({'images': stacked_images, 'labels': stacked_labels}, output_tensor_path)
    print(f"Tensors saved at {output_tensor_path}")


