# src/download_datasets.py

import os
import requests
from requests.exceptions import RequestException
import bz2
import urllib3


def download_file(url: str, save_path: str) -> None:
    """
    Download a file from the specified URL and save it to the given path,
    bypassing SSL verification.
    """
    # Disable the InsecureRequestWarning that will be printed to the console
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        print(f"Downloading {url} to {save_path}...")
        # Add verify=False to ignore SSL certificate errors
        with requests.get(url, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {url} successfully.")
    except RequestException as e:
        print(f"Error downloading {url}: {e}")
        raise e


def decompress_bz2_file(file_path: str, decompressed_path: str) -> None:
    """
    Decompress a .bz2 file.

    Parameters
    ----------
    file_path : str
        The path to the .bz2 file.
    decompressed_path : str
        The path where the decompressed file will be saved.

    Raises
    ------
    Exception
        If an error occurs during decompression.
    """
    try:
        with (
            bz2.BZ2File(file_path, "rb") as f_in,
            open(decompressed_path, "wb") as f_out,
        ):
            print(f"Decompressing {file_path} to {decompressed_path}...")
            for data in iter(
                lambda: f_in.read(100 * 1024), b""
            ):  # Read in chunks of 100KB
                f_out.write(data)
            print(f"Decompressed {file_path} successfully.")
    except Exception as e:
        print(f"Error decompressing {file_path}: {e}")
        raise e


def download_datasets() -> None:
    """
    Download and save datasets needed for the experiments.

    Downloads datasets from specified URLs, saves them locally, and
    decompresses them if necessary.
    """
    datasets = {
        # "YearPredictionMSD.bz2": (
        #     "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
        #     "regression/YearPredictionMSD.bz2"
        # ),
        "cpusmall": (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
            "regression/cpusmall"
        ),
        "cadata": (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
            "regression/cadata"
        ),
        "space_ga": (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
            "regression/space_ga"
        ),
        "abalone": (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"
            "regression/abalone"
        ),
    }
    data_dir = "data"  # The directory where datasets will be saved

    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    for filename, url in datasets.items():
        save_path = os.path.join(data_dir, filename)

        # Download the dataset if it doesn't already exist
        if not os.path.exists(save_path):
            try:
                download_file(url, save_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue
        else:
            print(f"{filename} already exists. Skipping download.")

        # If the file is compressed, decompress it
        if filename.endswith(".bz2"):
            decompressed_filename = filename[:-4]  # Remove '.bz2' extension
            decompressed_path = os.path.join(data_dir, decompressed_filename)
            if not os.path.exists(decompressed_path):
                try:
                    decompress_bz2_file(save_path, decompressed_path)
                except Exception as e:
                    print(f"Failed to decompress {filename}: {e}")
                    continue
            else:
                print(f"{decompressed_filename} already exists, skipping.")

    print("All datasets downloaded and ready.")


if __name__ == "__main__":
    download_datasets()
