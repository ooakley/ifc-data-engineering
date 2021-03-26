"""Tests for functional data loading capacity."""

import ifc_data_engineering.__main__ as ifc_main

TEST_ARGS = ["-i", "../../raw_data/20190821", "-o", "../../processed_data/20190821"]


def test_argparse() -> None:
    """Test for argparser functioning as expected."""
    args = ifc_main.parse_args(TEST_ARGS)
    assert args.path_to_dataset == "../../raw_data/20190821"


def test_tarfile_loading() -> None:
    pass
