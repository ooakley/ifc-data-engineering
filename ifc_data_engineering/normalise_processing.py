"""
Take raw numpy stack of image files, normalise according to parameters generated from experiment.

Example:
python normalise_dataset.py AIS-001
"""
import os
import numpy as np
import json
import argparse

from sklearn import mixture

MIN_FLUORESCENCE = 0.03
MAX_SAT_PIXEL_COUNT = 200
EDGE_STD_THRESHOLD = 10
GRAD_RMS_THRESHOLD = 20
FLUOR_ARTIFACT_THRESHOLD = 60000


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Experiment number and condition, for example AIS-001_WTCTRL")
    return parser.parse_args()


def fluorescence_min_filter(array, min_fluor=MIN_FLUORESCENCE):
    """Filter normalised array to remove low information images."""
    print("Raw shape:", array.shape[0])
    mask = array[:, :, :, 1].mean(axis=(1, 2)) > min_fluor
    array = array[mask]
    print("Filtered shape:", array.shape[0])
    return array


def edge_std_filter(array):
    """Remove images with high standard deviation along their edge pixels  - for centred images."""
    top = array[:, 0, :, 0]
    bottom = array[:, -1, :, 0]
    left = array[:, :, 0, 0]
    right = array[:, :, -1, 0]
    edges = np.stack([top, bottom, left, right], axis=2)
    std = np.std(edges, axis=1)
    max_std = np.max(std, axis=1)
    return array[max_std < EDGE_STD_THRESHOLD]


def grad_rms_filter(full_array):
    """Remove images with low gradients in brightfield values  - for in-focus images."""
    array = full_array[:, :, :, 0]
    # Calculating gradients:
    grad_arrays = []
    for i in range(array.shape[0]):
        grad_arrays.append(np.gradient(array[i, :, :])[0])
    grad_arrays = np.stack(grad_arrays)
    sq_array = grad_arrays**2
    mean_sq_array = np.mean(np.reshape(sq_array, (sq_array.shape[0], 64*64)), axis=1)
    rms_array = np.sqrt(mean_sq_array)
    final_array = full_array[rms_array > GRAD_RMS_THRESHOLD, :, :, :]
    return final_array


def fluor_artifact_filter(array):
    """Remove images with artifacting in the fluorescence channel."""
    return array[(array[:, :, :, 1].max(axis=(1, 2)) < FLUOR_ARTIFACT_THRESHOLD)]


def calculate_brightfield(bf_array):
    """Calculate normalisation statistics for the brightfield channel."""
    # Constructing cell mask:
    print("- Constructing cell mask...")
    stdev_array = np.std(bf_array, axis=0)

    # Calculating background and cell contents stats:
    print("- Calculating median and standard deviation...")
    background_pix = bf_array[:, stdev_array < 50]
    cell_pix = bf_array[:, stdev_array > 50]
    median_background = np.median(background_pix)
    stdev_cell = np.std(cell_pix)

    return median_background, stdev_cell


def calculate_fluorescence(fl_array):
    """Calculate normalisation statistics for the fluorescence channel."""
    # Decimating array - reshaping is necessary for fitting of Gaussian mixture:
    if fl_array.shape[0] > 5000:
        intercalation_factor = int(fl_array.shape[0] // 5000)
        dec_array = fl_array[::intercalation_factor, :, :]
    else:
        dec_array = fl_array
    log_values = np.log(dec_array.flatten().reshape(-1, 1) + 1e-20)

    # Fitting mixture model:
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

    return means_list


def normalise_fluorescence(img_stack, punctae_mean, punctae_stdev, fluor_clip):
    """Normalise fluorescent channel of images in numpy array format."""
    log_stack = np.log(img_stack + 1e-20)
    gaussian_centred = log_stack - punctae_mean
    gaussian_normed = gaussian_centred / punctae_stdev
    gaussian_clipped = np.clip(gaussian_normed, *fluor_clip)
    arr_max = np.max(gaussian_clipped)
    arr_min = np.min(gaussian_clipped)
    gaussian_final = (gaussian_clipped - arr_min) / (arr_max - arr_min)
    return gaussian_final


def normalise_brightfield(img_stack, median_background, stdev_cell, bf_clip):
    """Normalise brightfield channel of images in numpy array format."""
    bf_centred = img_stack - median_background
    bf_normed = bf_centred / stdev_cell
    bf_clipped = np.clip(bf_normed, *bf_clip)
    arr_max = np.max(bf_clipped)
    arr_min = np.min(bf_clipped)
    bf_final = (bf_clipped - arr_min) / (arr_max - arr_min)
    return bf_final


def generate_normalised_array(array, stats_dict):
    """Normalise full brightfield/fluorescence array."""
    # Normalising channels:
    normed_stack = np.zeros_like(array)
    normed_stack[:, :, :, 0] = normalise_brightfield(array[:, :, :, 0],
                                                     stats_dict["median_background"],
                                                     stats_dict["stdev_cell"],
                                                     stats_dict["bf_clip"])
    normed_stack[:, :, :, 1] = normalise_fluorescence(array[:, :, :, 1],
                                                      stats_dict["punctae_mean"],
                                                      stats_dict["punctae_stdev"],
                                                      stats_dict["fluor_clip"])
    return normed_stack


def main():
    """Take padded numpy image arrays, normalise brightfield/fluorescent channels, save file."""
    # Parsing arguments:
    args = _parse_args()
    expt_name = args.name

    # Parsing arguments:
    raw_path = os.path.join("files", "raw_numpy")
    expt_list = os.listdir(raw_path)
    dir_name = [dir_name for dir_name in expt_list if expt_name in dir_name]
    assert len(dir_name) == 1, "Duplicate experiments in filepath."
    dir_path = os.path.join("files", "raw_numpy", dir_name[0])
    filename_list = os.listdir(dir_path)
    image_array_dict = {}

    # Loading image arrays of raw numpy files:
    print("Loading images...")
    for filename in filename_list:
        full_path = os.path.join(dir_path, filename)
        image_array_dict[filename] = np.load(full_path)

    # Filter datasets for junk images:
    print("Filtering images...")
    for arr_name in image_array_dict:
        print("--- Condition:", arr_name, "---")
        print("Raw size:", image_array_dict[arr_name].shape[0])
        filtered_array = edge_std_filter(image_array_dict[arr_name])
        filtered_array = grad_rms_filter(filtered_array)
        filtered_array = fluor_artifact_filter(filtered_array)
        image_array_dict[arr_name] = filtered_array
        print("Filtered size:", image_array_dict[arr_name].shape[0])

    # Concatenate datasets for calculation of norm stats:
    concat_array = []
    for arr_name in image_array_dict:
        if image_array_dict[arr_name].shape[0] > 5000:
            decimate_factor = image_array_dict[arr_name].shape[0] // 5000
            concat_array.append(image_array_dict[arr_name][::decimate_factor])
        else:
            concat_array.append(image_array_dict[arr_name])
    concat_array = np.concatenate(concat_array, axis=0)
    print("Dataset size for generating norm stats:", concat_array.shape)

    # Calculating normalisation statistics:
    print("Calculating brightfield stats...")
    median_background, stdev_cell = calculate_brightfield(concat_array[:, :, :, 0])
    print("Calculating fluorescence stats...")
    means_list = calculate_fluorescence(concat_array[:, :, :, 1])
    stats_dict = {
        "punctae_mean": means_list[2][0],
        "punctae_stdev": means_list[2][1],
        "median_background": median_background,
        "stdev_cell": stdev_cell,
        "bf_clip": (-3, 3),
        "fluor_clip": (-1, 2)
        }

    # Apply normalisation to individual conditions:
    print("Applying normalisation stats to...")
    for arr_name in image_array_dict:
        print(arr_name)
        image_array_dict[arr_name] = generate_normalised_array(
            image_array_dict[arr_name], stats_dict)
        image_array_dict[arr_name] = fluorescence_min_filter(image_array_dict[arr_name])

    # Saving normalised numpy arrays:
    print("Saving files...")
    normed_path = os.path.join("files", "normed_dataset", dir_name[0])
    if not os.path.exists(normed_path):
        os.makedirs(normed_path)
    for arr_name in image_array_dict:
        save_path = os.path.join(normed_path, arr_name)
        np.save(save_path, image_array_dict[arr_name])

    # Saving normalisation statistics:
    print("Saving norm stats...")
    dirpath = os.path.join("files", "norm_stats")
    stats_filepath = os.path.join(dirpath, expt_name + ".json")
    if not os.path.exists(dirpath):
        print("Generating path", dirpath)
        os.makedirs(dirpath)
    with open(stats_filepath, 'w') as data_file:
        json.dump(stats_dict, data_file)


if __name__ == "__main__":
    main()
