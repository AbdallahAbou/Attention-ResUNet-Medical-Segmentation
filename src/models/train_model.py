import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.data_loader import MedicalSliceDataset
from src.data.process_data import save_images_under_300_slices
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from torch.amp import autocast, GradScaler
from src.models.attention_res_unet import AttentionResUNet
from torch.utils.data import random_split
import time
import random
import os
import json

def compute_dice(preds, targets, smooth=1e-5):
    preds = preds > 0.5  
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))  
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))  
    dice = (2.0 * intersection + smooth) / (union + smooth)  
    return dice.mean().item()


def train_model(images_dir, labels_dir, model_save_path, batch_size=4, num_epochs=1, learning_rate=1e-4, preloaded_model_path=None):
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
    save_images_under_300_slices(images_dir, json_300)
    with open(json_300, 'r') as f:
        all_images_under_300 = json.load(f)
    ids_list_ = all_images_under_300
    #ids_list_ = random.sample(all_images_under_300, 15)
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
    model = AttentionResUNet(in_channels=1, out_channels=1, init_features=8).to(device)
    if preloaded_model_path is not None:
        model.load_state_dict(torch.load(preloaded_model_path))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    Dice_criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    criterion = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False, ignore_empty=False, num_classes=2)
    scaler = GradScaler('cuda')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait before stopping
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_epoch = time.time()
        for i, (images, labels) in enumerate(train_loader):
            start_iter = time.time()
            
            images = images.to(device)
            labels = labels.to(device)

            labels = (labels != 0).float()

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

        end_epoch = time.time()
        avg_train_loss = train_loss / len(train_loader)
        # Validation Loop
        model.eval()
        val_loss = 0
        dice_scores_custom = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels = (labels != 0).float()
                
                outputs = model(images)
                loss_ = criterion(outputs, labels)
                val_loss += loss_.item()

                outputs_sigmoid = torch.sigmoid(outputs)
                dice_score = compute_dice(outputs_sigmoid, labels)
                dice_scores_custom.append(dice_score)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = sum(dice_scores_custom) / len(dice_scores_custom)
        #print(f"Validation loop: Image shape : {images.shape}, Labels shape : {labels.shape}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice score: {avg_dice_score:.4f}, Time: {end_epoch - start_epoch:.2f}s")

    
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, model_save_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, model_save_path)

