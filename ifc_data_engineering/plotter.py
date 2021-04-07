"""Set of methods for easy plotting of datasets."""
import numpy as np
import matplotlib.pyplot as plt


def minmax(image_array: np.ndarray) -> np.ndarray:
    """Perform minmax normalisation on a single cell image."""
    max_pixel_value = image_array.max()
    min_pixel_value = image_array.min()
    return (image_array - min_pixel_value) / (max_pixel_value - min_pixel_value)


def plot_image_grid(image_array: np.ndarray, side_length: int, channel: int, filepath: str) -> None:
    """Plot grid of images selected from give numpy array."""
    count = 0
    row_list = []
    for i in range(side_length):
        temp_row = []
        for j in range(side_length):
            image = minmax(image_array[count, :, :, channel])
            temp_row.append(image)
            count += 1
        row_list.append(np.concatenate(temp_row, axis=1))

    grid = np.concatenate(row_list, axis=0)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="gray")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filepath, dpi=500)
