import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.data_loader import MedicalDataset, custom_collate_fn
from src.models.attention_res_unet import AttentionResUNet
from src.models.loss import DiceLoss  

def train_model(train_images_dir, train_labels_dir, model_save_path, val_split=0.2, num_epochs=10, learning_rate=1e-4, pretrained_model_path=None):
    """
    Trains the AttentionResUNet model with the provided datasets.

    Parameters:
    - train_images_dir: str, path to the training images directory.
    - train_labels_dir: str, path to the training labels directory.
    - model_save_path: str, path to save the trained model.
    - val_split: float, the fraction of the dataset to use for validation. Defaults to 0.2 (20%).
    - num_epochs: int, the number of epochs to train for. Defaults to 10.
    - learning_rate: float, the learning rate for the optimizer. Defaults to 1e-4.
    - pretrained_model_path: str, optional path to a pre-trained model to continue training.   
    Returns:
    - None. Prints training progress and saves the model.
    """

    # Set device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    full_dataset = MedicalDataset(image_dir=train_images_dir, label_dir=train_labels_dir)

    # Split dataset into training and validation sets
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model, optimizer, and loss function
    model = AttentionResUNet(in_channels=1, out_channels=2).to(device)

    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))
        print(f"Loaded pre-trained model from {pretrained_model_path}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dice_loss = DiceLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Calculate loss
            loss = dice_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}")
    
    # Save the model
    torch.save(model.state_dict(), model_save_path)

