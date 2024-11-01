import torch
from torch.utils.data import DataLoader
from src.data.data_loader import MedicalSliceDataset
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import os
from src.models.attention_res_unet import AttentionResUNet

def test_model(images_dir, labels_dir, model_path, batch_size=4, num_classes=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Initialize the test dataset and DataLoader
    test_dataset = MedicalSliceDataset(image_dir=images_dir, label_dir=labels_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model and load the saved weights
    model = AttentionResUNet(in_channels=1, out_channels=num_classes, init_features=8).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize the DiceMetric for evaluation
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=num_classes)
    dice_helper_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            outputs_softmax = torch.softmax(outputs, dim=1)
            labels_one_hot = torch.nn.functional.one_hot(labels.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

            # Calculate Dice Metric for each batch and aggregate
            dice_metric(outputs_softmax, labels_one_hot)
            dice_helper_score, _ = DiceMetric(include_background=False, reduction="mean", get_not_nans=True, num_classes=num_classes)(outputs_softmax, labels_one_hot)
            dice_helper_scores.append(dice_helper_score.mean().item())

    # Aggregate Dice score over all batches
    dice_score, _ = dice_metric.aggregate()

    # Average Dice score from DiceHelper (for comparison)
    avg_dice_helper_score = sum(dice_helper_scores) / len(dice_helper_scores)

    print(f"Test Dice Metric score: {dice_score.mean().item()}")
    print(f"Test DiceHelper average score: {avg_dice_helper_score:.4f}")

    # Reset metric for future evaluations
    dice_metric.reset()

if __name__ == "__main__":
    # Paths for images, labels, and saved model
    images_dir = './path/to/test/images'
    labels_dir = './path/to/test/labels'
    model_path = './path/to/saved_model.pth'

    # Run testing
    test_model(images_dir, labels_dir, model_path, batch_size=4, num_classes=3)
