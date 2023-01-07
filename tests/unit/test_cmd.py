import os
import pytest
from conf import Config
from pathlib import Path

# Loading environment variables (for local use only)
from dotenv import load_dotenv
load_dotenv()


def test_cmd_download(expected_hash):
    # Clearing folder of any possible data
    files_list = os.listdir(Config.RAW_DATA_HANDLER['export_settings']['export_filepath'])
    matching_files = [x for x in files_list if expected_hash in x]
    for file in matching_files:
        os.remove(Path(Config.RAW_DATA_HANDLER['export_settings']['export_filepath'], file))

    # Running download_data
    cli_command = "python main.py download --hash_method sha256"

    os.system(cli_command)

    # Check if hash code exists in file directory
    files_list = os.listdir(Config.RAW_DATA_HANDLER['export_settings']['export_filepath'])
    matching_files = [x for x in files_list if expected_hash in x]
    assert len(matching_files) > 0, f"File was not downloaded successfully. Expected hash {expected_hash} in " \
                                    f"files_list but these were found: {files_list}"


def test_cmd_train(mlflow_setup):

    mlflow_file_path = os.getenv('ARTIFACT_URI')

    # Checking initial number of models in ./mlruns/ folder
    ori_total_mlruns_files = sum([len(files) for r, d, files in os.walk(mlflow_file_path)])
    # Executing cmd script
    cli_command = "python main.py --unit_test train --model regression --algorithms LR RFR " \
                  "--numeric-features longitude latitude housing_median_age total_rooms total_bedrooms population " \
                  "households median_income " \
                  "--categorical-features ocean_proximity --register ut_above_median_house_value "\
                  "--variables above_median_house_value"

    os.system(cli_command)
    # Checking final number of models in ./mlruns/ folder
    new_total_mlruns_files = sum([len(files) for r, d, files in os.walk(mlflow_file_path)])
    assert new_total_mlruns_files > ori_total_mlruns_files, f"No new files detected in {mlflow_file_path} folder."


@pytest.mark.last
def test_cmd_predict(mlflow_setup):

    # Executing cmd script
    new_data_filepath = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    export_filename = "test_predict.csv"

    cli_command = f"python main.py --unit_test predict --na_treatment drop " \
                  f"--import_from csv --data_filepath {new_data_filepath} " \
                  f"--export_to csv --export_filepath . --export_filename {export_filename} " \
                  f"--model_name ut_above_median_house_value --model_version latest"

    os.system(cli_command)

    assert os.path.exists(f"{export_filename}"), f"Prediction failed, {export_filename} not found."
    os.remove(f"{export_filename}")


