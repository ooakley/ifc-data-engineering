"""
Converts .tgz files from a given ImageStream experiment to raw numpy dataset.

Images are padded based on a basic statistical model of the intensity distribution of their edge
pixels, in addition to being filtered if they don't conform to set of constraints on image
dimension.

Example:
python tarball_to_dataset.py AIS-001_F91S
"""
from __future__ import annotations
from datetime import datetime
import json
import os

from tifffile import TiffFile
from io import BytesIO
import tarfile
import numpy as np

from .plotter import plot_image_grid

MAX_DIM_SIZE = 90
MIN_DIM_SIZE = 20
MAX_DIM_RATIO = 1.4


class TarHandler:
    """Handles reading of tarfiles into numpy array and subsequent processing."""

    def __init__(self, filepath: str) -> None:
        """Initialise with path to tarfile."""
        self.filepath = filepath
        self.metadata: dict = {}
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["timestamp"] = timestamp

    def _extract_filenames_from_tar(self, tar: tarfile.TarFile) -> tuple:
        """Read relevant tarfile metadata into a python list."""
        # Sorting file list such that different channels of the same image are sequentially ordered:
        raw_list = tar.getnames()
        raw_list.sort()
        trunc_list = [name[len(raw_list[0])+1:] for name in raw_list][1:]

        # Filtering tar dir just for ome_tiff file in channels of interest:
        ch1_list = [
            raw_list[0] + "/" + name for name in trunc_list if (
                name.find("Thumbs") == -1 and
                name.find(".") != 0 and
                name.find("Ch1") != -1 and
                name.find("Ch11") == -1
            )
        ]
        ch2_list = [
            raw_list[0] + "/" + name for name in trunc_list if (
                name.find("Thumbs") == -1 and
                name.find(".") != 0 and
                name.find("Ch2") != -1 and
                name.find("Ch12") == -1
            )
        ]

        # Assert all images are paired:
        assert len(ch1_list) == len(ch2_list), (
            "Name extraction from tar dir failed, not all images paired."
        )

        return ch1_list, ch2_list

    def _extract_images_from_tar(
            self, tar: tarfile.TarFile, ch1_list: list, ch2_list: list
            ) -> list:
        """
        Extract tiff images from tar archive and convert to numpy arrays.

        We need to firstly get the tarinfo object. As the tifffile module requires BytesIO buffers
        to parse the tifffile into a numpy array, we read the tarinfo object as a buffer before
        converting it to a numpy array and storing it in a list.
        """
        # Pulling image data from tar directory into list:
        print("Dataset size:", int(len(ch1_list)))
        dataset_size = int(len(ch1_list))
        self.metadata["dataset_size"] = dataset_size
        raw_numpy_list = []

        # Looping over dataset:
        for i in range(dataset_size):
            ch_one = tar.getmember(ch1_list[i])
            ch_two = tar.getmember(ch2_list[i])
            with BytesIO(tar.extractfile(ch_one).read()) as file:
                with TiffFile(file) as tif:
                    array_one = tif.asarray()
            with BytesIO(tar.extractfile(ch_two).read()) as file:
                with TiffFile(file) as tif:
                    array_two = tif.asarray()
                    stack_list = (array_one, array_two)
                    raw_numpy_list.append(np.stack(stack_list, axis=2))
            if (i + 1) % 1000 == 0:
                print(i + 1, " cells read...")

        return raw_numpy_list

    def _filter_on_dimension(self, dataset_list: list) -> list:
        """Filter cell images based solely on their dimensions."""
        filtered_data = []
        MAX_DIM = 120
        MIN_DIM = 30

        for img_array in dataset_list:
            rows = img_array.shape[0]
            cols = img_array.shape[1]
            if max(rows, cols) > MAX_DIM:
                continue
            if min(rows, cols) < MIN_DIM:
                continue
            if max(rows, cols) / min(rows, cols) > MAX_DIM_RATIO:
                continue
            filtered_data.append(img_array)

        print("Images filtered:", len(dataset_list) - len(filtered_data))
        self.metadata["images_filtered"] = len(dataset_list) - len(filtered_data)

        return filtered_data

    def _generate_edge_padding_array(
            self, image_array_list: list, output_rows: int, output_cols: int
            ) -> list:
        """Generate noise sampled from distribution of edge pixel intensities."""
        background_array_list = []

        for image_array in image_array_list:
            channels_list = []
            for i in range(image_array.shape[-1]):
                # Selecting image channel:
                single_channel = image_array[:, :, i]

                # Deriving stats from edge pixels:
                edge_pixel_list = [
                    single_channel[0, :], single_channel[-1, :],
                    single_channel[1:-1, 0], single_channel[1:-1, -1]
                ]
                edge_pixel_array = np.concatenate(edge_pixel_list)
                edge_mean = edge_pixel_array.mean()
                edge_var = edge_pixel_array.var()

                # Removing pixels that probably aren't background:
                excl_cond = (edge_pixel_array - edge_mean)**2 > 3 * edge_var
                filtered_values = np.ma.masked_array(
                    edge_pixel_array, np.where(excl_cond, True, False)
                )
                filt_mean = filtered_values.mean()
                filt_std = filtered_values.std()

                # Generating background into which img_array can be inserted:
                temp_background_array = np.random.normal(
                    filt_mean, filt_std, size=(output_rows, output_cols)
                ).astype('uint16')

                channels_list.append(temp_background_array)

            full_background_array = np.stack(channels_list, axis=2)
            background_array_list.append(full_background_array)

            selection_fluor = [array[:, :, 0].ravel() for array in background_array_list[::10]]
            selection_bf = [array[:, :, 1].ravel() for array in background_array_list[::10]]

            self.metadata["avg_fluor_intensity"] = np.mean(np.concatenate(selection_fluor))
            self.metadata["avg_bf_intensity"] = np.mean(np.concatenate(selection_bf))

        return background_array_list

    def _pad_image_arrays(
            self, image_list: list, background_array_list: list, output_rows: int, output_cols: int
            ) -> list:

        padded_image_list: list = []

        for i in range(len(image_list)):
            rows = image_list[i].shape[0]
            cols = image_list[i].shape[1]

            # Cropping image if too tall:
            if rows > output_rows:
                top = (rows - output_rows) // 2
                bottom = top + output_rows
                rows = output_rows

            if rows <= output_rows:
                top = 0
                bottom = rows

            # Cropping image if too wide:
            if cols > output_cols:
                left = (cols - output_cols) // 2
                right = left + output_cols
                cols = output_cols

            if cols <= output_cols:
                left = 0
                right = cols

            temp_array = image_list[i][top:bottom, left:right]

            # Placing image into generated background (if either dimension is too small):
            pad_top = (output_rows - rows) // 2
            pad_bottom = pad_top + rows
            pad_left = (output_cols - cols) // 2
            pad_right = pad_left + cols

            padded_image = background_array_list[i]

            padded_image[pad_top:pad_bottom, pad_left:pad_right] = temp_array

            padded_image_list.append(padded_image)

        return padded_image_list

    def process_file(self) -> None:
        """Take tarfile, extract tiff images, processes resultant numpy arrays."""
        with tarfile.open(self.filepath, "r:gz") as tar:
            # Tarfiles are effectively compressed file directories, so the
            # metadata containing the filenames of contained files is extracted first:
            ch1_list, ch2_list = self._extract_filenames_from_tar(tar)

            # The filenames are then used to extract the contained .tiff files into numpy arrays:
            raw_numpy_list = self._extract_images_from_tar(tar, ch1_list, ch2_list)

        # Filter these arrays based on dimension:
        filtered_list = self._filter_on_dimension(raw_numpy_list)

        # Pad/crop these arrays to uniform dimension:
        background_array_list = self._generate_edge_padding_array(filtered_list, 64, 64)
        padded_image_list = self._pad_image_arrays(filtered_list, background_array_list, 64, 64)

        # Concatenate arrays into numpy dataset:
        dataset = np.stack(padded_image_list, axis=0)
        print("Final dataset shape:", dataset.shape)

        self.dataset = dataset

    def save_raw_dataset(self) -> None:
        """Save dataset and metadata to parent directory of input file."""
        parent_dir = os.path.dirname(self.filepath)
        numpy_path = os.path.join(parent_dir, "raw_dataset.npy")
        metadata_path = os.path.join(parent_dir, "raw_metadata.json")
        fluor_image_path = os.path.join(parent_dir, "fluor_grid.png")
        bf_image_path = os.path.join(parent_dir, "bf_grid.png")

        with open(metadata_path, "w") as write_file:
            json.dump(self.metadata, write_file, indent=4)

        with open(numpy_path, "wb") as write_file:
            np.save(write_file, self.dataset)

        plot_image_grid(self.dataset, 10, 0, fluor_image_path)
        plot_image_grid(self.dataset, 10, 1, bf_image_path)
