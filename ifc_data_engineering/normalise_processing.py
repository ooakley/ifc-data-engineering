"""
Take raw numpy stack of image files, normalise according to parameters generated from experiment.

Example:
python normalise_dataset.py AIS-001
"""
from __future__ import annotations
import os
import numpy as np
import json
import argparse

from sklearn import mixture

EDGE_STD_THRESHOLD = 10
GRAD_RMS_THRESHOLD = 20
FLUOR_ARTIFACT_THRESHOLD = 60000
MIN_FLUORESCENCE = 0.03
MAX_SAT_PIXEL_COUNT = 200


def edge_std_filter(array: np.ndarray) -> np.ndarray:
    """Remove images with high standard deviation along their edge pixels - for centred images."""
    top = array[:, 0, :, 0]
    bottom = array[:, -1, :, 0]
    left = array[:, :, 0, 0]
    right = array[:, :, -1, 0]
    edges = np.stack([top, bottom, left, right], axis=2)
    std = np.std(edges, axis=1)
    max_std = np.max(std, axis=1)
    return array[max_std < EDGE_STD_THRESHOLD]


def grad_rms_filter(array: np.ndarray) -> np.ndarray:
    """Remove images with low gradients in brightfield values - for in-focus images."""
    bf_array = array[:, :, :, 0]
    # Calculating gradients:
    grad_list: list[np.ndarray] = []
    for i in range(bf_array.shape[0]):
        grad_list.append(np.gradient(bf_array[i, :, :])[0])
    grad_array = np.stack(grad_list)
    sq_grad_array = grad_array**2
    mean_sq_array = np.mean(np.reshape(sq_grad_array, (sq_grad_array.shape[0], 64*64)), axis=1)
    rms_array = np.sqrt(mean_sq_array)
    final_array = array[rms_array > GRAD_RMS_THRESHOLD, :, :, :]
    return final_array


def fluor_artifact_filter(array: np.ndarray) -> np.ndarray:
    """Remove images with artifacting in the fluorescence channel."""
    return array[(array[:, :, :, 1].max(axis=(1, 2)) < FLUOR_ARTIFACT_THRESHOLD)]


def fluorescence_min_filter(array: np.ndarray, min_fluor: float = MIN_FLUORESCENCE) -> np.ndarray:
    """Filter normalised array to remove images with low signal:noise ratios."""
    mask = array[:, :, :, 1].mean(axis=(1, 2)) > min_fluor
    array = array[mask]
    return array


def saturated_pixel_filter(array: np.ndarray, max_sat: int = MAX_SAT_PIXEL_COUNT) -> np.ndarray:
    """Filter normalised array to remove images with high saturation."""
    sat_pixel_count = np.count_nonzero(array[:, :, :, 1].squeeze() == 1, axis=(1, 2))
    return array[sat_pixel_count < 200]


class DatasetGenerator:

    def __init__(self, filepath_list: str) -> None:
        """Load numpy array into interal dataset dictionary."""
        self.filepath_list = filepath_list
        self.dataset_dict: dict = {}
        self.init_condition_numbers: dict = {}

    def _load_filter_dataset(self) -> None:
        """Filter raw datasets."""
        for filepath in self.filepath_list:
            print("Loading numpy dataset from:", filepath)
            filename = os.path.basename(filepath)
            dot_index = filename.index(".")
            condition = filename[:dot_index]
            self.dataset_dict[condition] = np.load(filepath)
            self.init_condition_numbers[condition] = self.dataset_dict[condition].shape[0]

            print("--- Condition:", condition, "---")
            filtered_array = edge_std_filter(self.dataset_dict[condition])
            filtered_array = grad_rms_filter(filtered_array)
            filtered_array = fluor_artifact_filter(filtered_array)
            self.dataset_dict[condition] = filtered_array

    def _calculate_normalisation_statistics(self) -> tuple:
        """Calculate statistics for normalisation across entire dataset."""
        # Constructing selected brightfield and fluorescence arrays:
        bf_array_list: list = []
        fluor_array_list: list = []
        for condition in self.dataset_dict:
            bf_array_list.append(self.dataset_dict[condition][:1000, :, :, 0])
            fluor_array_list.append(self.dataset_dict[condition][:1000, :, :, 1])
        bf_array = np.concatenate(bf_array_list, axis=0)
        fluor_array = np.concatenate(fluor_array_list, axis=0)

        # BRIGHTFIELD NORM:
        # Constructing cell mask:
        print("- Constructing cell mask...")
        stdev_array = np.std(bf_array, axis=0)

        # Calculating background and cell contents stats:
        print("- Calculating median and standard deviation...")
        background_pix = bf_array[:, stdev_array < 50]
        cell_pix = bf_array[:, stdev_array > 50]
        median_background = np.median(background_pix)
        stdev_cell = np.std(cell_pix)

        # FLUORESCENT NORM:
        # Taking log of values as pixel values follow log distribution:
        log_values = np.log(fluor_array.flatten().reshape(-1, 1) + 1e-20)

        # Fitting mixture model:
        print("Fitting gaussian mixture model...")
        mixture_model = mixture.GaussianMixture(n_components=3, covariance_type='full',
                                                verbose=2, verbose_interval=5, tol=1e-4)
        mixture_model.fit(log_values)
        component_params = np.vstack(
            (np.squeeze(mixture_model.means_), np.squeeze(mixture_model.covariances_))
            )
        means_list = np.sort(component_params).T

        # Printing results:
        print("Component 1 Mean/StdDev (Fluorescent Background):", means_list[0])
        print("Component 2 Mean/StdDev (Cell Contents):", means_list[1])
        print("Component 3 Mean/StdDev (LC3 Punctae):", means_list[2])

        return median_background, stdev_cell, means_list

    def _apply_normalisation_statistics(
            self, median_background: np.ndarray, stdev_cell: np.ndarray, means_list: list
            ) -> None:
        for condition in self.dataset_dict:
            # Normalising brightfield:
            bf_stack = self.dataset_dict[condition][:, :, :, 0]
            bf_centred = bf_stack - median_background
            bf_normed = bf_centred / stdev_cell
            bf_clipped = np.clip(bf_normed, -3, 3)
            bf_final = (bf_clipped + 3) / 6

            # Normalising fluorescent:
            fluor_stack = np.log(self.dataset_dict[condition][:, :, :, 0] + 1e-20)
            fluor_centred = fluor_stack - means_list[2][0]  # Subtracting punctae mean
            fluor_normed = fluor_centred / means_list[2][1]  # Dividing by punctae stdev
            fluor_clipped = np.clip(fluor_normed, -2, 2)
            fluor_final = (fluor_clipped + 2) / 4

            # Concatenating and overwriting:
            self.dataset_dict[condition] = np.stack([bf_final, fluor_final], axis=3)

    def _apply_post_norm_filters(self) -> None:
        for condition in self.dataset_dict:
            self.dataset_dict[condition] = fluorescence_min_filter(self.dataset_dict[condition])
            self.dataset_dict[condition] = saturated_pixel_filter(self.dataset_dict[condition])

    def run_normalisation(self) -> None:
        """Load datasets to internal dict, and run normalisation and quality filtering."""
        self._load_filter_dataset()
        median_background, stdev_cell, means_list = self._calculate_normalisation_statistics()
        self._apply_normalisation_statistics(median_background, stdev_cell, means_list)
        self._apply_post_norm_filters()

    def save_normalised_dataset(self, dirpath: str) -> None:
        """Save normalised dataset to directory specified by dirpath."""
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
    
        # Saving dataset:
        array_list = list(self.dataset_dict.values())
        array_path = os.path.join(dirpath, "dataset.npy")
        with open(array_path, "wb") as f:
            array = np.concatenate(array_list, axis=0)
            print("Final dataset size:", array.shape[0])
            np.save(f, array)
        
        # Saving dataset label array:
        count: int = 0
        label_list: list = []
        label_dict: dict = []
        for condition in self.dataset_dict:
            label = self.dataset_dict[condition].shape[0]

        # Save label : condition dict
        # Save metadata:
        # Date created
        # Number of images filtered
        # Filter hyperparameters
        # Normalisation statistics
        # Dataset hash
        # Random seed (for gaussian init)
        # Grid images of normalised and filtered data
        # Pixel value and basic image feature histograms

    stats_dict = {
        "punctae_mean": means_list[2][0],
        "punctae_stdev": means_list[2][1],
        "median_background": median_background,
        "stdev_cell": stdev_cell,
        "bf_clip": (-3, 3),
        "fluor_clip": (-1, 2)
        }

