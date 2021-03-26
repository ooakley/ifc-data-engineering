"""
Converts .tgz files from a given ImageStream experiment to raw numpy dataset.

Images are padded based on a basic statistical model of the intensity distribution of their edge
pixels, in addition to being filtered if they don't conform to set of constraints on image
dimension.

Example:
python tarball_to_dataset.py AIS-001_F91S
"""
from tifffile import TiffFile
import os
from io import BytesIO
import tarfile
import numpy as np
import argparse

MAX_DIM_SIZE = 90
MIN_DIM_SIZE = 20
MAX_DIM_RATIO = 1.4


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Experiment number and condition, for example AIS-001_WTCTRL")
    parser.add_argument('-s', '--size', default='64x64',
                        help="output size as {rows}x{columnss}, e.g. 64x64")
    parser.add_argument('--max_dim_ratio', help="The maximum ratio of input dimensions",
                        default=MAX_DIM_RATIO, type=float)
    parser.add_argument('--min_dim_size', help="The minimum input dimension",
                        default=MIN_DIM_SIZE, type=int)
    parser.add_argument('--max_dim_size', help="The maximum input dimension",
                        default=MAX_DIM_SIZE, type=int)
    return parser.parse_args()


def data_from_tar(tar):
    """Pull data from tarball into a python list."""
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

    assert len(ch1_list) == len(ch2_list), (
        "Name extraction from tar dir failed, not all images paired."
    )
    dataset_size = int(len(ch1_list))
    print("Dataset size:", dataset_size)

    # Pulling image data from tar directory into list:
    data = [None] * dataset_size
    for i in range(dataset_size):
        ch_one = tar.getmember(ch1_list[i])
        ch_two = tar.getmember(ch2_list[i])
        with BytesIO(tar.extractfile(ch_one).read()) as file:
            with TiffFile(file) as tif:
                array_one = tif.asarray()
                data[i] = array_one
        with BytesIO(tar.extractfile(ch_two).read()) as file:
            with TiffFile(file) as tif:
                array_two = tif.asarray()
                stack_list = (data[i], array_two)
                data[i] = np.stack(stack_list, axis=2)
        if (i + 1) % 1000 == 0:
            print(i + 1, " cells read...")
    return data


def filter_data(dataset_list):
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
    return filtered_data


def edge_padding_array(img_array, output_rows, output_cols):
    """Generate noise sampled from distribution of edge pixel intensities."""
    # Deriving stats from edge pixels:
    edge_pixels = [img_array[0, :], img_array[-1, :],
                   img_array[1:-1, 0], img_array[1:-1, -1]]
    edge_pixels = np.concatenate(edge_pixels)
    edge_mean = edge_pixels.mean()
    edge_var = edge_pixels.var()

    # Removing pixels that probably aren't background:
    excl_cond = (edge_pixels - edge_mean)**2 > 3 * edge_var
    filtered_values = np.ma.masked_array(edge_pixels, np.where(excl_cond, True, False))
    filt_mean = filtered_values.mean()
    filt_std = filtered_values.std()

    # Generating background into which img_array can be inserted:
    bkgr_array = np.random.normal(
        filt_mean, filt_std, size=(output_rows, output_cols)
    ).astype('uint16')
    return bkgr_array


def pad_img_array(img_array, output_rows, output_cols):
    """Generate padding, then pad/crop image to desired dimensions as appropriate."""
    bkgr_array = edge_padding_array(img_array, output_rows, output_cols)
    rows = img_array.shape[0]
    cols = img_array.shape[1]

    # Cropping image if too tall:
    if rows > output_rows:
        top = (rows - output_rows) // 2
        bottom = top + output_rows
        img_array = img_array[top:bottom, :]
        rows = output_rows

    # Cropping image if too wide:
    if cols > output_cols:
        left = (cols - output_cols) // 2
        right = left + output_cols
        img_array = img_array[:, left:right]
        cols = output_cols

    # Placing image into generated background:
    top = (output_rows - rows) // 2
    bottom = top + rows
    left = (output_cols - cols) // 2
    right = left + cols
    bkgr_array[top:bottom, left:right] = img_array
    return bkgr_array


def tgz_to_dataset(path):
    """Generate .npy dataset from path to tarfile."""
    # Opening tarfile into list:
    print("Reading cell images...")
    with tarfile.open(path, 'r:gz') as tar:
        dataset_list = data_from_tar(tar)
    print("Cell images read:", len(dataset_list))

    # Filter based on image dimensions:
    filtered_dataset = filter_data(dataset_list)
    print("Images filtered out due to their shape:", len(dataset_list) - len(filtered_dataset))
    print("Final raw dataset size:", len(filtered_dataset))

    # Pad (or crop) images to defined dimensions:
    padded_dataset = np.zeros((len(filtered_dataset), 64, 64, 2))
    for i in range(len(filtered_dataset)):
        padarray_ch1 = pad_img_array(filtered_dataset[i][:, :, 0], 64, 64)
        padarray_ch2 = pad_img_array(filtered_dataset[i][:, :, 1], 64, 64)
        stack = np.stack((padarray_ch1, padarray_ch2), axis=2)
        padded_dataset[i, :, :, :] = stack
    return padded_dataset


def main():
    """Parse arguments, load in .tgz, save/upload resultant raw dataset."""
    # Parsing arguments:
    args = _parse_args()

    # Setting up args, paths, & directories:
    rows, cols = [int(dim) for dim in args.size.split('x')]
    expt_no, expt_cond = args.name.split('_')
    tarball_path = os.path.join("files", "tarballs")
    expt_list = os.listdir(tarball_path)
    dir_name = [dir_name for dir_name in expt_list if expt_no in dir_name]
    assert len(dir_name) == 1, "Duplicate experiment numbers in tarball dir."
    filepath = os.path.join("files", "tarballs", dir_name[0], expt_cond + ".tgz")

    # Saving dataset locally & uploading:
    raw_np_path = os.path.join("files", "raw_numpy", dir_name[0])
    if not os.path.exists(raw_np_path):
        os.makedirs(raw_np_path)
    save_path = os.path.join(raw_np_path, expt_cond + ".npy")
    np.save(save_path, padded_dataset)
