"""Process raw image data in the form of a .tgz file."""
from .tarball_processing import tgz_to_dataset


def main():
    """Run main program logic."""
    padded_dataset = tgz_to_dataset(filepath)
    
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
    return None


main()
