import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.data_loader import MedicalSliceDataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.amp import autocast, GradScaler
from src.models.attention_res_unet import AttentionResUNet
from torch.utils.data import random_split
import time
import random
import os






def train_model(images_dir, labels_dir, model_save_path, batch_size=4, num_epochs=1, learning_rate=1e-4):
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

    # Dataset and DataLoader
    ids_list_ = []
    for file in sorted(os.listdir(images_dir)):
        if not file.startswith('._') and file.endswith(".nii.gz"): 
            ids_list_.append(file)

    random.seed(42) 
    random.shuffle(ids_list_)

    train_size = int(0.8 * len(ids_list_))
    train_ids = ids_list_[:train_size]
    val_ids = ids_list_[train_size:]
    print(f"Training Set length: {len(train_ids)}/{len(ids_list_)}" )
    print(f"Validation Set length: {len(val_ids)}/{len(ids_list_)}" )

    train_dataset = MedicalSliceDataset(image_dir=images_dir, label_dir=labels_dir, ids_list=train_ids)
    val_dataset = MedicalSliceDataset(image_dir=images_dir, label_dir=labels_dir, ids_list=val_ids)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Model, optimizer, and loss function
    model = AttentionResUNet(in_channels=1, out_channels=3, init_features=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=3)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    patience = 7  # Number of epochs to wait before stopping
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_epoch = time.time()
        for i, (images, labels) in enumerate(train_loader):
            start_iter = time.time()
            
            images = images.to(device)
            labels = labels.to(device)

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

        end_epoch = time.time()
        avg_train_loss = train_loss / len(train_loader)
        # Validation Loop
        model.eval()
        val_loss = 0
        dice_metric.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                outputs_softmax = torch.softmax(outputs, dim=1)
                #print('Label shape before one hot:', labels.shape)
                # One-hot encode labels
                labels_one_hot = torch.nn.functional.one_hot(labels.squeeze(1), num_classes=3)
                #print('Label shape after one hot:', labels_one_hot.shape)
                labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()
                #print('Label shape after permute:', labels_one_hot.shape)
                dice_metric(outputs_softmax, labels_one_hot)

        avg_val_loss = val_loss / len(val_loader)
        dice_score, not_nans = dice_metric.aggregate()#[0].item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice score: {dice_score}, Time: {end_epoch - start_epoch:.2f}s")
        
        dice_metric.reset()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0

        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break
    torch.save(model.state_dict(), model_save_path)

