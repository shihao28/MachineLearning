import os
import pytest
from conf import Config
from pathlib import Path
from src.data_loading import RawDataHandler

# Loading environment variables (for local use only)
from dotenv import load_dotenv
load_dotenv()


def test_raw_data_handler():

    # Initialisation
    expected_hash = "c26c91"

    # Clearing folder of any possible data
    files_list = os.listdir(Config.RAW_DATA_HANDLER['export_settings']['export_filepath'])
    matching_files = [x for x in files_list if expected_hash in x]
    for file in matching_files:
        os.remove(Path(Config.RAW_DATA_HANDLER['export_settings']['export_filepath'], file))

    # Running download_data
    RawDataHandler.download_data(hash_method='sha256')

    # Check if hash code exists in file directory
    files_list = os.listdir(Config.RAW_DATA_HANDLER['export_settings']['export_filepath'])
    matching_files = [x for x in files_list if expected_hash in x]
    assert len(matching_files) > 0, f"File was not downloaded successfully. Expected hash {expected_hash} in " \
                                    f"files_list but these were found: {files_list}"
