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
from src.models.train_model import compute_dice
import time
import random
import os
import json

def pred_model(images_dir, model_path, batch_size=4, test_size=None, ids_list=None, features=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Initialize the test dataset and DataLoader
    test_ids = [file for file in os.listdir(images_dir) if file.endswith('.nii.gz') and not file.startswith('.')]
    random.seed(2)
    if test_size is not None:
        test_ids = random.sample(test_ids, test_size)
    test_dataset = MedicalSliceDataset(image_dir=images_dir, ids_list=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Load the trained model

    # Initialize the model and load the saved weights
    model = AttentionResUNet(in_channels=1, out_channels=3, init_features=features).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    #dice_scores_custom = []

    outputs_list = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            outputs = model(images)
            outputs_list.append(outputs.cpu())
            #outputs_softmax = torch.softmax(outputs, dim=1)

            #labels_one_hot = one_hot(labels, num_classes=3).float()
            #print(f"Validation loop: Output softmax shape : {outputs_softmax.shape}, Labels One hot shape : {labels_one_hot.shape}")
            #print(f"Unique labels one hot: {torch.unique(labels_one_hot)} ")
            #dice_score = compute_dice(outputs_softmax, labels_one_hot)
            #dice_scores_custom.append(dice_score)

        #avg_dice_score = sum(dice_scores_custom) / len(dice_scores_custom)


    outputs_all = torch.cat(outputs_list, dim=0)  

    predicted_masks = torch.argmax(outputs_all, dim=1) 

    return test_ids, predicted_masks 