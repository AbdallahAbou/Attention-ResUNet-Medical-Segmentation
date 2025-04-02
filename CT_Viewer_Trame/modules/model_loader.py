import torch
from AttentionResUnet import AttentionResUNet

# Global variables that will be accessed from the main application
liver_model = None
vessel_model = None
device = None

def load_models(liver_model_path, vessel_model_path):
    """Load both the liver and vessel segmentation models."""
    global liver_model, vessel_model, device
    
    # Make the globals accessible to the main application
    import sys
    main_module = sys.modules['__main__']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Initialize the liver model (features=8)
        liver_model = AttentionResUNet(in_channels=1, out_channels=3, init_features=8).to(device)
        
        # Load the saved weights
        checkpoint = torch.load(liver_model_path, map_location=device)
        liver_model.load_state_dict(checkpoint['model_state_dict'])
        liver_model.eval()
        print("Liver model loaded successfully.")
        
        # Initialize the vessel model (features=16)
        vessel_model = AttentionResUNet(in_channels=1, out_channels=3, init_features=16).to(device)
        
        # Load the saved weights
        checkpoint = torch.load(vessel_model_path, map_location=device)
        vessel_model.load_state_dict(checkpoint['model_state_dict'])
        vessel_model.eval()
        print("Vessel model loaded successfully.")
        
        # Set the global references in the main module
        main_module.liver_model = liver_model
        main_module.vessel_model = vessel_model
        main_module.device = device
        
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False