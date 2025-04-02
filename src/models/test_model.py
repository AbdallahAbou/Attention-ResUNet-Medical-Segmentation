import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.data_loader import MedicalSliceDataset
from src.data.process_data import save_images_under_300_slices
from monai.networks.utils import one_hot
from src.models.attention_res_unet import AttentionResUNet
from torch.utils.data import random_split
import random
import os
import json
import numpy as np


def compute_dice(preds, targets, smooth=1e-5):
    """
    Computes per-class Dice scores averaged over the batch.

    Parameters:
    - preds: Tensor of predicted probabilities, shape (batch_size, num_classes, H, W)
    - targets: Tensor of ground truth one-hot encoded labels, shape (batch_size, num_classes, H, W)

    Returns:
    - dice_per_class: NumPy array of per-class Dice scores, shape (num_classes,)
    """
    # Ensure tensors are floats
    preds = preds.float()
    targets = targets.float()
    num_classes = preds.shape[1]

    # Flatten spatial dimensions
    preds_flat = preds.view(preds.shape[0], num_classes, -1)  # Shape: (batch_size, num_classes, H*W)
    targets_flat = targets.view(targets.shape[0], num_classes, -1)  # Shape: (batch_size, num_classes, H*W)

    # Compute intersection and union
    intersection = (preds_flat * targets_flat).sum(2)  # Shape: (batch_size, num_classes)
    union = preds_flat.sum(2) + targets_flat.sum(2)  # Shape: (batch_size, num_classes)

    # Compute per-class Dice scores for each sample
    dice = (2.0 * intersection + smooth) / (union + smooth)  # Shape: (batch_size, num_classes)

    # Average Dice score over batch for each class
    dice_per_class = dice.mean(dim=0).cpu().numpy()  # Shape: (num_classes,)

    return dice_per_class

def test_model(images_dir, labels_dir, model_path, batch_size=4, features=8, data='liver'):        
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    random.seed(3)
    json_300 = os.path.join(images_dir, 'images_under_300_slices.json') 
    #save_images_under_300_slices(images_dir, json_300, slice_threshold=80)
    json_file = json_300
    if data == 'vessel':
        json_file = os.path.join(images_dir, 'images_under_80_slices.json') 
    with open(json_file, 'r') as f:
        all_images_under = json.load(f)
    ids_list_ = all_images_under
    ids_list_ = random.sample(all_images_under, 5)

    # Dataset and DataLoader
    #ids_list_ = []
    #for file in sorted(os.listdir(images_dir)):
    #    if not file.startswith('._') and file.endswith(".nii.gz"): 
    #        ids_list_.append(file)

    test_dataset = MedicalSliceDataset(image_dir=images_dir, label_dir=labels_dir, ids_list=ids_list_)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    # Model, optimizer, and loss function

    model = AttentionResUNet(in_channels=1, out_channels=3, init_features=features).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    dice_scores_per_class = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)

            outputs_softmax = torch.softmax(outputs, dim=1)

            labels_one_hot = one_hot(labels, num_classes=3).float()

            dice_per_class = compute_dice(outputs_softmax, labels_one_hot)
            dice_scores_per_class.append(dice_per_class)

    dice_scores_per_class = np.array(dice_scores_per_class)  # Shape: (num_batches, num_classes)
    
    # Calculate average Dice score per class over all batches
    avg_dice_scores_per_class = dice_scores_per_class.mean(axis=0)  # Shape: (num_classes,)
    
    # Map class indices to class names
    class_names = {0: 'Background', 1: 'Liver', 2: 'Tumor'}
    if data == 'vessel':
        class_names = {0: 'Background', 1: 'Vessel', 2: 'Tumor'}
    # Print per-class Dice scores
    for idx, score in enumerate(avg_dice_scores_per_class):
        print(f"Dice Score for {class_names[idx]}: {score:.4f}")
    
    # Optionally, return the per-class Dice scores
    return avg_dice_scores_per_class