import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors

def visualize_slices(image_data, label_data):
    """
    Visualizes slices of a given 3D image and its corresponding label with a slider to navigate through slices.

    Parameters:
    - image_data: 3D NumPy array representing the image.
    - label_data: 3D NumPy array representing the corresponding label data.
    """
    # Define the custom colormap for the labels
    cmap = colors.ListedColormap(['black', 'red', 'yellow'])  # Adjust colors as needed
    bounds = [0, 1, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Enable interactive plotting mode
    plt.ion()
    
    slice_index = 0  # Start from the first slice

    # Initialize the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Display the initial image and label overlay
    image_slice = image_data[:, :, slice_index].T  # Transpose for correct orientation
    label_slice = label_data[:, :, slice_index].T

    img_display = ax.imshow(image_slice, cmap='gray', origin='lower')
    label_display = ax.imshow(label_slice, cmap=cmap, norm=norm, alpha=0.5, origin='lower')

    ax.set_title(f'Slice {slice_index}')
    ax.axis('off')

    # Slider setup for navigating through slices
    axcolor = 'lightgoldenrodyellow'
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slice_slider = Slider(ax_slice, 'Slice', 0, image_data.shape[2] - 1, valinit=slice_index, valfmt='%0.0f')

    # Update function for the slider
    def update(val):
        slice_idx = int(slice_slider.val)
        image_slice = image_data[:, :, slice_idx].T
        label_slice = label_data[:, :, slice_idx].T

        img_display.set_data(image_slice)
        label_display.set_data(label_slice)

        ax.set_title(f'Slice {slice_idx}')
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slice_slider.on_changed(update)

    plt.show()

def analyze_image_statistics(liver_data, vessel_data):
    """
    Analyzes and prints the statistics (min, max, mean, std) of liver and vessel CT images.
    Also plots histograms for the intensity distribution of both images.

    Parameters:
    - liver_data: 3D NumPy array representing the liver image.
    - vessel_data: 3D NumPy array representing the vessel image.
    """
    # Analyze liver CT image
    liver_min_intensity = np.min(liver_data)
    liver_max_intensity = np.max(liver_data)
    liver_mean_intensity = np.mean(liver_data)
    liver_std_intensity = np.std(liver_data)

    print(f"Liver CT Image - Min: {liver_min_intensity}, Max: {liver_max_intensity}, Mean: {liver_mean_intensity}, Std: {liver_std_intensity}")

    # Plot histogram for liver image
    plt.figure(figsize=(10, 4))
    plt.hist(liver_data.flatten(), bins=100, color='blue', alpha=0.7)
    plt.title('Intensity Distribution of Liver CT Image')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.show()

    # Analyze vessel CT image
    vessel_min_intensity = np.min(vessel_data)
    vessel_max_intensity = np.max(vessel_data)
    vessel_mean_intensity = np.mean(vessel_data)
    vessel_std_intensity = np.std(vessel_data)

    print(f"Vessel CT Image - Min: {vessel_min_intensity}, Max: {vessel_max_intensity}, Mean: {vessel_mean_intensity}, Std: {vessel_std_intensity}")

    # Plot histogram for vessel image
    plt.figure(figsize=(10, 4))
    plt.hist(vessel_data.flatten(), bins=100, color='green', alpha=0.7)
    plt.title('Intensity Distribution of Vessel CT Image')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.show()

# Example usage (for testing):
# visualize_slices(liver_data, liver_label_data)
# analyze_image_statistics(liver_data, vessel_data)
