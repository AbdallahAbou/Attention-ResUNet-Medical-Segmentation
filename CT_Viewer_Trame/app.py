from trame.app import get_server
from trame.app.file_upload import ClientFile
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, html
import nibabel as nib
import numpy as np
import tempfile
import os

# Initialize server
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# Initialize state
state.trame__title = "CT Image Viewer"
state.current_slice = 0
state.max_slice = 0
state.has_image = False
state.status_message = ""
state.current_image = ""

# Module-level storage for 3D image data
image_data_3d = None

# File upload handler
@state.change("file_exchange")
def handle_file_upload(file_exchange, **kwargs):
    global image_data_3d
    
    try:
        # Load the image
        file = ClientFile(file_exchange)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_file:
            tmp_file.write(file.content)
            image_data_3d = nib.load(tmp_file.name).get_fdata()
        
        # Update state
        state.max_slice = image_data_3d.shape[2] - 1
        state.current_slice = state.max_slice // 2  # Start at the middle slice
        state.has_image = True
        
        # Display the initial slice
        update_display()
        
    except Exception as e:
        state.status_message = f"Error: {str(e)}"

# Slider update handler
@ctrl.add("update_slice")
def update_slice(new_slice):
    """Update the current slice and refresh the image."""
    print(f"Updating slice to: {new_slice}")  # Debugging
    state.current_slice = int(new_slice)  # Update the slice index
    update_display()  # Refresh the image

# Image extraction and encoding
def update_display():
    """Extract the current slice and update the displayed image."""
    if image_data_3d is None:
        return

    print(f"Displaying slice: {state.current_slice}")  # Debugging

    # Extract the current slice
    slice_data = image_data_3d[:, :, state.current_slice]
    
    # Normalize the slice to grayscale (0–255)
    normalized = ((slice_data - np.min(slice_data)) / 
                 (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
    
    # Flip the image 90 degrees to the right (rotate clockwise)
    normalized = np.rot90(normalized, k=1)
    
    # Convert to Base64-encoded PNG
    from PIL import Image
    import io
    import base64
    
    with io.BytesIO() as buffer:
        Image.fromarray(normalized).save(buffer, format="PNG")
        state.current_image = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

# UI Layout
with SinglePageLayout(server) as layout:
    with layout.toolbar:
        vuetify.VToolbarTitle("CT Image Viewer")
    
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
            
            # Image display section
            with vuetify.VRow(v_if="has_image", justify="center", classes="mt-4"):
                with vuetify.VCol(cols=12, sm=10, md=8):
                    # Only change: the VCard is now styled to be 50% width
                    with vuetify.VCard(style="width: 50%; margin: auto;"):
                        with vuetify.VCardTitle():
                            html.Span("CT Image View")
                        with vuetify.VCardText():
                            html.Img(
                                style="width: 100%;",
                                src=("current_image", ""),
                            )
                            vuetify.VSlider(
                                v_model=("current_slice", 0),
                                min=0,
                                max=("max_slice", 0),
                                step=1,
                                label="Slice",
                                classes="mt-4",
                                change=(ctrl.update_slice, "[$event]"),  # Trigger on slider change
                            )

if __name__ == "__main__":
    server.start()
