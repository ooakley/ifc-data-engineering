"""Process raw image data in the form of a .tgz file."""
import argparse
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .tarball_handler import TarHandler


def parse_args() -> Any:
    """Define and parse arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_tar",
        type=str, help="Path to input .tgz file to be processed."
    )
    parser.add_argument(
        "--numpy_list", 
        type=str, nargs='+',
        help="list of paths to raw numpy files for processing"
    )
    parser.add_argument(
        "--path_to_output",
        type=str, help="Path to directory where normalised .npy file will be saved."
    )
    parser.add_argument(
        "-n", "--do_normalisation",
        action="store_true", help="Whether to normalise image data."
    )
    parser.add_argument(
        "-f", "--do_filtering",
        action="store_true", help="Whether to filter image data."
    )
    return parser.parse_args()



def main() -> None:
    """Run main program logic."""
    args = parse_args()

    # Processing tarballs:
    if args.path_to_tar is not None:
        print("Processing:", args.path_to_tar)
        handler = TarHandler(args.path_to_tar)
        handler.process_file()
        handler.save_raw_dataset()

    if args.numpy_list is not None:
        do_thing()


main()
