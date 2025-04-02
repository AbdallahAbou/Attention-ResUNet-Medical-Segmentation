from trame.app import get_server
from trame.app.file_upload import ClientFile
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, html
import nibabel as nib
import numpy as np
import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from separate modules
from modules.AttentionResUnet import AttentionResUNet
from modules.model_loader import load_models
from modules.func import normalize_image, resample_slice

# Additional imports for encoding
from PIL import Image
import io
import base64

# Initialize server
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# Initialize state variables
state.trame__title = "CT Image Viewer with Segmentation"
state.has_image = False
state.status_message = ""
state.segmentation_enabled = False

# New state variables for segmentation visibility
state.show_liver = True
state.show_vessels = True
state.show_tumor = True

# The slider will represent a fraction (0.0–1.0)
state.current_slice_fraction = 0.5

# We also keep the maximum indices for each dimension (for reference)
state.max_axial = 0      # dimension 2
state.max_sagittal = 0   # dimension 0
state.max_coronal = 0    # dimension 1

# Module-level storage for 3D image data and affine matrix
image_data_3d = None
affine_matrix = None
liver_segmentation_data_3d = None
vessel_segmentation_data_3d = None
liver_model = None
vessel_model = None
device = None

LIVER_MODEL_PATH = "models/liver_model_classification.pth"
VESSEL_MODEL_PATH = "models/vessel_model_classification_16.pth"

def normalize_slice(slice_data):
    """Normalize a 2D slice for model input."""
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)
    if max_val - min_val == 0:
        normalized = np.zeros_like(slice_data)
    else:
        normalized = (slice_data - min_val) / (max_val - min_val)
    return normalized

def predict_segmentation(normalized_image_data_3d, model_type):
    """Run the segmentation model on the entire 3D volume."""
    global liver_model, vessel_model, device
    
    # Select the appropriate model based on the model type
    if model_type == "liver":
        model = liver_model
        model_name = "Liver"
    else:  # "vessel"
        model = vessel_model
        model_name = "Vessel"
    
    if model is None:
        state.status_message = f"Error: {model_name} model not loaded."
        return None
    
    try:
        # Create an empty array for segmentation results
        segmentation_data_3d = np.zeros(normalized_image_data_3d.shape, dtype=np.uint8)
        
        # Show progress message
        state.status_message = f"Running {model_name} segmentation... (0%)"
        
        # Process each slice
        total_slices = normalized_image_data_3d.shape[2]
        with torch.no_grad():
            for z in range(total_slices):
                # Update progress every 10% of slices
                if z % max(1, total_slices // 10) == 0:
                    progress = int((z / total_slices) * 100)
                    state.status_message = f"Running {model_name} segmentation... ({progress}%)"
                
                # Get the slice and prepare it for the model
                slice_data = normalized_image_data_3d[:, :, z]
                slice_tensor = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Run the model
                output = model(slice_tensor)
                output_softmax = torch.softmax(output, dim=1)
                prediction = torch.argmax(output_softmax, dim=1).squeeze().cpu().numpy()
                
                # Store the prediction
                segmentation_data_3d[:, :, z] = prediction
        
        state.status_message = f"{model_name} segmentation completed successfully."
        return segmentation_data_3d
        
    except Exception as e:
        state.status_message = f"Error during {model_name} segmentation: {str(e)}"
        print(f"Error during {model_name} segmentation: {str(e)}")
        return None

# File upload handler
@state.change("file_exchange")
def handle_file_upload(file_exchange, **kwargs):
    global image_data_3d, affine_matrix, liver_segmentation_data_3d, vessel_segmentation_data_3d
    
    try:
        # Reset segmentations
        liver_segmentation_data_3d = None
        vessel_segmentation_data_3d = None
        state.segmentation_enabled = False
        
        # Load the image from the uploaded file
        file = ClientFile(file_exchange)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
            tmp_file.write(file.content)
            tmp_file.flush()
            img = nib.load(tmp_file.name)
            image_data_3d = img.get_fdata()
            affine_matrix = img.affine
            os.unlink(tmp_file.name)  # Remove the temporary file

        # Set maximum slice indices for each view
        # Axial view uses the third dimension (index 2)
        state.max_axial = image_data_3d.shape[2] - 1
        # Sagittal view uses the first dimension (index 0)
        state.max_sagittal = image_data_3d.shape[0] - 1
        # Coronal view uses the second dimension (index 1)
        state.max_coronal = image_data_3d.shape[1] - 1

        # Set the slider to the middle position
        state.current_slice_fraction = 0.5
        state.has_image = True

        # Display the initial set of images
        update_display()
        
        state.status_message = "Image loaded successfully."

    except Exception as e:
        state.status_message = f"Error: {str(e)}"
        print(f"Error: {str(e)}")

# Slider update handler for the single slider
@ctrl.add("update_slice_fraction")
def update_slice_fraction(new_val):
    state.current_slice_fraction = float(new_val)
    update_display()

# Toggle handlers for each segmentation type
@ctrl.add("toggle_liver")
def toggle_liver():
    state.show_liver = not state.show_liver
    update_display()

@ctrl.add("toggle_vessels")
def toggle_vessels():
    state.show_vessels = not state.show_vessels
    update_display()

@ctrl.add("toggle_tumor")
def toggle_tumor():
    state.show_tumor = not state.show_tumor
    update_display()

# Run both segmentations with one button
@ctrl.add("run_all_segmentations")
def run_all_segmentations():
    global image_data_3d, liver_segmentation_data_3d, vessel_segmentation_data_3d, liver_model, vessel_model
    
    if image_data_3d is None:
        state.status_message = "Please load an image first."
        return
    
    try:
        # If models aren't loaded yet, load them
        if liver_model is None or vessel_model is None:
            if not load_models(LIVER_MODEL_PATH, VESSEL_MODEL_PATH):
                state.status_message = "Failed to load models."
                return
        
        # Create a normalized version of the 3D image for segmentation
        normalized_image_data_3d = np.zeros_like(image_data_3d)
        for z in range(image_data_3d.shape[2]):
            normalized_image_data_3d[:, :, z] = normalize_slice(image_data_3d[:, :, z])
        
        # Run liver segmentation
        state.status_message = "Starting liver segmentation..."
        liver_segmentation_data_3d = predict_segmentation(normalized_image_data_3d, "liver")
        
        # Run vessel segmentation
        state.status_message = "Starting vessel segmentation..."
        vessel_segmentation_data_3d = predict_segmentation(normalized_image_data_3d, "vessel")
        
        # Enable segmentation and make sure we can see it
        state.segmentation_enabled = True
        
        # Update the display with both segmentations
        state.status_message = "All segmentations completed."
        update_display()
        
    except Exception as e:
        state.status_message = f"Error in segmentation: {str(e)}"
        print(f"Error in segmentation: {str(e)}")

def encode_image(slice_array, liver_seg_slice=None, vessel_seg_slice=None):
    """
    Encode a 2D numpy array as a Base64 PNG image.
    A rotation/flip is applied to achieve an anatomically intuitive orientation.
    If segmentation slices are provided, overlay them based on visibility settings.
    """
    transformed = np.flipud(slice_array.T)
    
    # Convert grayscale to RGB
    rgb_image = np.stack([transformed] * 3, axis=-1)
    
    # If we have segmentation data and segmentation is enabled, blend it with the image
    if state.segmentation_enabled:
        # Create a color overlay for the segmentation
        overlay_img = np.zeros_like(rgb_image)
        
        # Add liver segmentation if enabled and available
        if state.show_liver and liver_seg_slice is not None:
            # Transform the segmentation to match the image orientation
            liver_seg_transformed = np.flipud(liver_seg_slice.T)
            
            # Class 1 (liver): Red overlay
            liver_mask = (liver_seg_transformed == 1)
            overlay_img[liver_mask, 0] = 255  # Red channel
            
            # Class 2 (tumor/lesion): Green overlay
            if state.show_tumor:
                tumor_mask = (liver_seg_transformed == 2)
                overlay_img[tumor_mask, 1] = 255  # Green channel
        
        # Add vessel segmentation if enabled and available
        if state.show_vessels and vessel_seg_slice is not None:
            # Transform the segmentation to match the image orientation
            vessel_seg_transformed = np.flipud(vessel_seg_slice.T)
            
            # Class 1 (vessels): Blue overlay
            vessel_mask = (vessel_seg_transformed == 1)
            overlay_img[vessel_mask, 2] = 255  # Blue channel
        
        # Blend the segmentation with the original image
        alpha = 0.4  # Opacity of the overlay
        rgb_image = (1 - alpha) * rgb_image + alpha * overlay_img
        rgb_image = rgb_image.astype(np.uint8)
    
    # Convert NumPy array to PIL Image and encode
    with io.BytesIO() as buffer:
        Image.fromarray(rgb_image.astype(np.uint8)).save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

def update_display():
    """Extract slices from the 3D volume for axial, sagittal, and coronal views,
       resample to isotropic resolution, normalize, and update the displayed images."""
    global image_data_3d, affine_matrix, liver_segmentation_data_3d, vessel_segmentation_data_3d
    if image_data_3d is None or affine_matrix is None:
        return

    # Compute voxel spacings from the affine matrix
    spacing_x = np.linalg.norm(affine_matrix[:3, 0])  # Left-Right
    spacing_y = np.linalg.norm(affine_matrix[:3, 1])  # Anterior-Posterior
    spacing_z = np.linalg.norm(affine_matrix[:3, 2])  # Superior-Inferior

    # Use the current fraction to compute the index in each dimension.
    frac = state.current_slice_fraction
    axial_index = int(frac * state.max_axial)
    sagittal_index = int(frac * state.max_sagittal)
    coronal_index = int(frac * state.max_coronal)

    print(
        f"Displaying slices: Axial {axial_index}, Sagittal {sagittal_index}, Coronal {coronal_index}"
    )  # Debug output

    # --- Axial view (Top-Down) ---
    axial_slice = image_data_3d[:, :, axial_index]
    axial_slice = normalize_image(axial_slice)
    # Axial view: pixel spacing from spacing_x and spacing_y.
    axial_resampled = resample_slice(axial_slice, spacing_x, spacing_y)
    
    # Get corresponding segmentation slices if available
    axial_liver_seg_resampled = None
    if liver_segmentation_data_3d is not None:
        axial_liver_seg_slice = liver_segmentation_data_3d[:, :, axial_index]
        axial_liver_seg_resampled = resample_slice(axial_liver_seg_slice, spacing_x, spacing_y)
    
    axial_vessel_seg_resampled = None
    if vessel_segmentation_data_3d is not None:
        axial_vessel_seg_slice = vessel_segmentation_data_3d[:, :, axial_index]
        axial_vessel_seg_resampled = resample_slice(axial_vessel_seg_slice, spacing_x, spacing_y)
    
    state.current_image_axial = encode_image(axial_resampled, axial_liver_seg_resampled, axial_vessel_seg_resampled)

    # --- Sagittal view (Side) ---
    sagittal_slice = image_data_3d[sagittal_index, :, :]
    sagittal_slice = normalize_image(sagittal_slice)
    # Sagittal view: pixel spacing from spacing_y and spacing_z.
    sagittal_resampled = resample_slice(sagittal_slice, spacing_y, spacing_z)
    
    # Get corresponding segmentation slices if available
    sagittal_liver_seg_resampled = None
    if liver_segmentation_data_3d is not None:
        sagittal_liver_seg_slice = liver_segmentation_data_3d[sagittal_index, :, :]
        sagittal_liver_seg_resampled = resample_slice(sagittal_liver_seg_slice, spacing_y, spacing_z)
    
    sagittal_vessel_seg_resampled = None
    if vessel_segmentation_data_3d is not None:
        sagittal_vessel_seg_slice = vessel_segmentation_data_3d[sagittal_index, :, :]
        sagittal_vessel_seg_resampled = resample_slice(sagittal_vessel_seg_slice, spacing_y, spacing_z)
    
    state.current_image_sagittal = encode_image(sagittal_resampled, sagittal_liver_seg_resampled, sagittal_vessel_seg_resampled)

    # --- Coronal view (Front) ---
    coronal_slice = image_data_3d[:, coronal_index, :]
    coronal_slice = normalize_image(coronal_slice)
    # Coronal view: pixel spacing from spacing_x and spacing_z.
    coronal_resampled = resample_slice(coronal_slice, spacing_x, spacing_z)
    
    # Get corresponding segmentation slices if available
    coronal_liver_seg_resampled = None
    if liver_segmentation_data_3d is not None:
        coronal_liver_seg_slice = liver_segmentation_data_3d[:, coronal_index, :]
        coronal_liver_seg_resampled = resample_slice(coronal_liver_seg_slice, spacing_x, spacing_z)
    
    coronal_vessel_seg_resampled = None
    if vessel_segmentation_data_3d is not None:
        coronal_vessel_seg_slice = vessel_segmentation_data_3d[:, coronal_index, :]
        coronal_vessel_seg_resampled = resample_slice(coronal_vessel_seg_slice, spacing_x, spacing_z)
    
    state.current_image_coronal = encode_image(coronal_resampled, coronal_liver_seg_resampled, coronal_vessel_seg_resampled)

# UI Layout
with SinglePageLayout(server) as layout:
    with layout.toolbar:
        vuetify.VToolbarTitle("CT Image Viewer with Segmentation")

    with layout.content:
        with vuetify.VContainer(fluid=True):
            # File upload section
            with vuetify.VRow(justify="center"):
                with vuetify.VCol(cols=12, sm=8, md=6):
                    with vuetify.VCard(elevation=10):
                        with vuetify.VCardTitle():
                            vuetify.VIcon("mdi-file-upload", left=True)
                            html.Span("Upload CT Image")
                        with vuetify.VCardText():
                            vuetify.VFileInput(
                                v_model=("file_exchange", None),
                                accept=".nii,.nii.gz",
                                label="Choose file",
                                outlined=True,
                                prepend_icon="mdi-file",
                            )
            
            # Segmentation controls
            with vuetify.VRow(v_if="has_image", justify="center", classes="mt-4"):
                with vuetify.VCol(cols=12, sm=8, md=6):
                    with vuetify.VCard(elevation=10):
                        with vuetify.VCardTitle():
                            vuetify.VIcon("mdi-brain", left=True)
                            html.Span("Segmentation Controls")
                        with vuetify.VCardText():
                            # Display model paths for reference
                            html.Div("Liver Model: " + LIVER_MODEL_PATH, classes="mb-2")
                            html.Div("Vessel Model: " + VESSEL_MODEL_PATH, classes="mb-4")
                            
                            # Single segmentation button
                            with vuetify.VRow():
                                with vuetify.VCol(cols=12):
                                    vuetify.VBtn(
                                        "Run All Segmentations", 
                                        block=True,
                                        color="primary",
                                        click=ctrl.run_all_segmentations
                                    )
                            
                            # Checkboxes for toggling segmentation types
                            with vuetify.VRow(v_if="segmentation_enabled", classes="mt-4"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VCheckbox(
                                        v_model=("show_liver", True),
                                        label="Show Liver",
                                        color="error",  # Red color
                                        change=ctrl.toggle_liver
                                    )
                                with vuetify.VCol(cols=4):
                                    vuetify.VCheckbox(
                                        v_model=("show_vessels", True),
                                        label="Show Vessels",
                                        color="primary",  # Blue color
                                        change=ctrl.toggle_vessels
                                    )
                                with vuetify.VCol(cols=4):
                                    vuetify.VCheckbox(
                                        v_model=("show_tumor", True),
                                        label="Show Tumor",
                                        color="success",  # Green color
                                        change=ctrl.toggle_tumor
                                    )
            
            # Status message
            with vuetify.VRow(justify="center", classes="mt-2"):
                with vuetify.VCol(cols=12, sm=8, md=6):
                    with vuetify.VAlert(type="info", v_if="status_message", outlined=True):
                        html.Span("{{ status_message }}")
            
            # Image display section
            with vuetify.VRow(v_if="has_image", justify="center", classes="mt-4"):
                with vuetify.VCol(cols=12, sm=10):
                    # Make the whole CT Image View block 80% wide and centered
                    with vuetify.VCard(elevation=10, style="width: 80%; margin: auto; padding: 16px;"):
                        with vuetify.VRow():
                            # Axial view column (larger)
                            with vuetify.VCol(cols=12, sm=8):
                                html.Div("Axial (Top-Down)", style="text-align: center; font-weight: bold;")
                                html.Img(
                                    style="width: 100%;",
                                    src=("current_image_axial", ""),
                                )
                            # Right column for sagittal and coronal (stacked)
                            with vuetify.VCol(cols=12, sm=4):
                                with vuetify.VRow():
                                    with vuetify.VCol(cols=12):
                                        html.Div("Sagittal (Side)", style="text-align: center; font-weight: bold;")
                                        html.Img(
                                            style="width: 100%;",
                                            src=("current_image_sagittal", ""),
                                        )
                                with vuetify.VRow(classes="mt-4"):
                                    with vuetify.VCol(cols=12):
                                        html.Div("Coronal (Front)", style="text-align: center; font-weight: bold;")
                                        html.Img(
                                            style="width: 100%;",
                                            src=("current_image_coronal", ""),
                                        )
                        # Single slider to control all three views via a fraction
                        vuetify.VSlider(
                            v_model=("current_slice_fraction", 0.5),
                            min=0,
                            max=1,
                            step=0.01,
                            label="Slice Position (Fraction)",
                            classes="mt-4",
                            change=(ctrl.update_slice_fraction, "[$event]"),
                        )
                        
                        # Legend for segmentation colors (only shown when segmentation is enabled)
                        with vuetify.VRow(v_if="segmentation_enabled", justify="center", classes="mt-2"):
                            with vuetify.VCol(cols=12):
                                with vuetify.VCard(flat=True):
                                    with vuetify.VCardText():
                                        html.Div("Segmentation Legend:", style="font-weight: bold;")
                                        # Liver model legend
                                        with vuetify.VRow(v_if="show_liver", align="center"):
                                            with vuetify.VCol(cols="auto"):
                                                html.Div(style="width: 20px; height: 20px; background-color: rgba(255, 0, 0, 0.4);")
                                            with vuetify.VCol():
                                                html.Span("Liver")
                                        with vuetify.VRow(v_if="show_tumor", align="center"):
                                            with vuetify.VCol(cols="auto"):
                                                html.Div(style="width: 20px; height: 20px; background-color: rgba(0, 255, 0, 0.4);")
                                            with vuetify.VCol():
                                                html.Span("Tumor")
                                        
                                        # Vessel model legend
                                        with vuetify.VRow(v_if="show_vessels", align="center"):
                                            with vuetify.VCol(cols="auto"):
                                                html.Div(style="width: 20px; height: 20px; background-color: rgba(0, 0, 255, 0.4);")
                                            with vuetify.VCol():
                                                html.Span("Vessels")

if __name__ == "__main__":
    server.start()
