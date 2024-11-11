import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.data_loader import MedicalSliceDataset
from src.data.process_data import save_images_under_300_slices
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from torch.amp import autocast, GradScaler
from src.models.attention_res_unet import AttentionResUNet
from torch.utils.data import random_split
import time
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

def train_model(images_dir, labels_dir, model_save_path, batch_size=4, num_epochs=1, learning_rate=1e-4, preloaded_model_path=None, data='liver'):
    """
    Trains the AttentionResUNet model with the provided datasets.

    Parameters:
    - images_dir: str, path to the training images directory.
    - labels_dir: str, path to the training labels directory.
    - model_save_path: str, path to save the trained model.
    - num_epochs: int, the number of epochs to train for. Defaults to 10.
    - learning_rate: float, the learning rate for the optimizer. Defaults to 1e-4.
    Returns:
    - None. Prints training progress and saves the model.
    """
        
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    json_300 = os.path.join(images_dir, 'images_under_300_slices.json') 
    save_images_under_300_slices(images_dir, json_300, slice_threshold=300)
    with open(json_300, 'r') as f:
        all_images_under_300 = json.load(f)
    ids_list_ = all_images_under_300
    #ids_list_ = random.sample(all_images_under_300, 3)

    # Dataset and DataLoader
    #ids_list_ = []
    #for file in sorted(os.listdir(images_dir)):
    #    if not file.startswith('._') and file.endswith(".nii.gz"): 
    #        ids_list_.append(file)


    random.seed(42) 
    random.shuffle(ids_list_)

    train_size = int(0.8 * len(ids_list_))
    train_ids = ids_list_[:train_size]
    val_ids = ids_list_[train_size:]
    print(f"Training Set length: {len(train_ids)}/{len(ids_list_)}" )
    print(f"Validation Set length: {len(val_ids)}/{len(ids_list_)}" )
    print("Training IDs:", train_ids)
    print("Validation IDs:", val_ids)

    train_dataset = MedicalSliceDataset(image_dir=images_dir, label_dir=labels_dir, ids_list=train_ids)
    val_dataset = MedicalSliceDataset(image_dir=images_dir, label_dir=labels_dir, ids_list=val_ids)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    # Model, optimizer, and loss function
    model = AttentionResUNet(in_channels=1, out_channels=3, init_features=16).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    Dice_criterion = DiceLoss(to_onehot_y=True, softmax=True)
    Dice_CE = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
    criterion = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False, ignore_empty=False, num_classes=2)
    scaler = GradScaler('cuda')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    dice_scores = []

    if preloaded_model_path is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(preloaded_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming training from epoch {start_epoch}")
    patience = 5  # Number of epochs to wait before stopping
    trigger_times = 0

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0
        start_epoch = time.time()
        for i, (images, labels) in enumerate(train_loader):
            start_iter = time.time()
            
            images = images.to(device)
            labels = labels.to(device).long()

            #labels = (labels != 0).float()

            optimizer.zero_grad()

            with autocast('cuda'):
                outputs = model(images)
                #print(f"Outputs stats - min: {outputs.min()}, max: {outputs.max()}, mean: {outputs.mean()}")
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            end_iter = time.time()
            #print(f"Iteration {i+1}/{len(train_loader)}, Elapsed time: {end_iter - start_epoch:.4f} seconds")
        #print(f"Training loop: Image shape : {images.shape}, Labels shape : {labels.shape}")

        
        avg_train_loss = train_loss / len(train_loader)
        # Validation Loop
        model.eval()
        val_loss = 0
        dice_scores_per_class = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()

                unique_labels = torch.unique(labels)
                #print(f"Unique values in labels: {unique_labels}") 

                #labels = (labels != 0).float()
                #print(f"Validation loop: Image shape : {images.shape}, Labels shape : {labels.shape}")
                outputs = model(images)
                loss_ = criterion(outputs, labels)
                val_loss += loss_.item()

                outputs_softmax = torch.softmax(outputs, dim=1)
                unique_outputs = torch.unique(outputs_softmax)
                #print(f"Unique values in outputs (softmax): {unique_outputs}")

                labels_one_hot = one_hot(labels, num_classes=3).float()
                #print(f"Validation loop: Output softmax shape : {outputs_softmax.shape}, Labels One hot shape : {labels_one_hot.shape}")
                #print(f"Unique labels one hot: {torch.unique(labels_one_hot)} ")
                dice_per_class = compute_dice(outputs_softmax, labels_one_hot)
                dice_scores_per_class.append(dice_per_class)

        avg_val_loss = val_loss / len(val_loader)
        dice_scores_per_class = np.array(dice_scores_per_class)  # Shape: (num_batches, num_classes)
    
        # Calculate average Dice score per class over all batches
        avg_dice_scores_per_class = dice_scores_per_class.mean(axis=0)  # Shape: (num_classes,)
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
                'scaler_state_dict': scaler.state_dict(),
            }, model_save_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

        end_epoch = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {end_epoch - start_epoch:.2f}s")
        # Map class indices to class names
        class_names = {0: 'Background', 1: 'Liver', 2: 'Tumor'}
        if data == 'vessel':
            class_names = {0: 'Background', 1: 'Vessel', 2: 'Tumor'}
        # Print per-class Dice scores
        for idx, score in enumerate(avg_dice_scores_per_class):
            print(f"Dice Score for {class_names[idx]}: {score:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice_scores_per_class)

        #torch.cuda.empty_cache()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_val_loss,
        'scaler_state_dict': scaler.state_dict(),
    }, model_save_path)
    metrics = {
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'dice_scores': [float(score) for score in dice_scores],
    }
    metrics_save_path = os.path.splitext(model_save_path)[0] + '_metrics_64.json'
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to {metrics_save_path}")

