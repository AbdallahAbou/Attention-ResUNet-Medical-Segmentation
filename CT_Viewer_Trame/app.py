from trame.app import get_server
from trame.app.file_upload import ClientFile
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, html
import nibabel as nib
import numpy as np
import tempfile
import os

# Additional imports for resampling and image encoding
from scipy.ndimage import zoom
from PIL import Image
import io
import base64

# Initialize server
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# Initialize state variables (using one slider controlling a fraction 0–1)
state.trame__title = "CT Image Viewer"
state.has_image = False
state.status_message = ""

# The slider will represent a fraction (0.0–1.0)
state.current_slice_fraction = 0.5

# We also keep the maximum indices for each dimension (for reference)
state.max_axial = 0      # dimension 2
state.max_sagittal = 0   # dimension 0
state.max_coronal = 0    # dimension 1

# Module-level storage for 3D image data and affine matrix
image_data_3d = None
affine_matrix = None

# File upload handler
@state.change("file_exchange")
def handle_file_upload(file_exchange, **kwargs):
    global image_data_3d, affine_matrix

    try:
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

    except Exception as e:
        state.status_message = f"Error: {str(e)}"

# Slider update handler for the single slider
@ctrl.add("update_slice_fraction")
def update_slice_fraction(new_val):
    state.current_slice_fraction = float(new_val)
    update_display()

def normalize_image(slice_data):
    """Normalize a 2D slice to grayscale [0, 255] as uint8."""
    if np.max(slice_data) == np.min(slice_data):
        return np.zeros(slice_data.shape, dtype=np.uint8)
    normalized = ((slice_data - np.min(slice_data)) /
                  (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
    return normalized

def resample_slice(slice_data, orig_spacing1, orig_spacing2):
    """
    Resample the 2D slice so that its pixels have isotropic spacing.
    orig_spacing1 and orig_spacing2 are the original voxel spacings along the two axes.
    """
    target_spacing = min(orig_spacing1, orig_spacing2)
    zoom_factor = (orig_spacing1 / target_spacing, orig_spacing2 / target_spacing)
    return zoom(slice_data, zoom=zoom_factor, order=3)

def encode_image(slice_array):
    """
    Encode a 2D numpy array as a Base64 PNG image.
    A rotation/flip is applied to achieve an anatomically intuitive orientation.
    """
    transformed = np.flipud(slice_array.T)
    with io.BytesIO() as buffer:
        Image.fromarray(transformed).save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

def update_display():
    """Extract slices from the 3D volume for axial, sagittal, and coronal views,
       resample to isotropic resolution, normalize, and update the displayed images."""
    global image_data_3d, affine_matrix
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
    state.current_image_axial = encode_image(axial_resampled)

    # --- Sagittal view (Side) ---
    sagittal_slice = image_data_3d[sagittal_index, :, :]
    sagittal_slice = normalize_image(sagittal_slice)
    # Sagittal view: pixel spacing from spacing_y and spacing_z.
    sagittal_resampled = resample_slice(sagittal_slice, spacing_y, spacing_z)
    state.current_image_sagittal = encode_image(sagittal_resampled)

    # --- Coronal view (Front) ---
    coronal_slice = image_data_3d[:, coronal_index, :]
    coronal_slice = normalize_image(coronal_slice)
    # Coronal view: pixel spacing from spacing_x and spacing_z.
    coronal_resampled = resample_slice(coronal_slice, spacing_x, spacing_z)
    state.current_image_coronal = encode_image(coronal_resampled)

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
            # Image display section: axial image large on the left/center, 
            # sagittal (side) and coronal (front) stacked in a smaller column on the right.
            with vuetify.VRow(v_if="has_image", justify="center", classes="mt-4"):
                with vuetify.VCol(cols=12, sm=10):
                    # Only change: make the whole CT Image View block 50% wide and centered.
                    with vuetify.VCard(elevation=10, style="width: 50%; margin: auto; padding: 16px;"):
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

if __name__ == "__main__":
    server.start()
