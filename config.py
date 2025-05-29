import pandas as pd 

from datetime import datetime, UTC
from dotenv import load_dotenv, find_dotenv
#from pydantic_settings import BaseSettings, SettingsConfigDict

#from src.setup.paths import PARENT_DIR
#from src.feature_pipeline.data_sourcing import Year
import os 
from pathlib import Path 

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
HIDDEN_DIM = 512


'''env_file_path = PARENT_DIR.joinpath(".env") 
_ = load_dotenv(env_file_path) 


class GeneralConfig(BaseSettings):

    _ = SettingsConfigDict(
        env_file=str(env_file_path),
        env_file_encoding="utf-8", 
        extra="allow"
    )

    years: list[Year] = [
        Year(value=2024, offset=9),
        Year(value=2025, offset=0)
    ]

    n_features: int = 672

    # Hopsworks
    backfill_days: int = 210 
    feature_group_version: int = 1
    feature_view_version: int = 1

    current_hour: datetime = pd.to_datetime(datetime.now(tz=UTC)).floor("H")
    displayed_scenario_names: dict[str, str] = {"start": "Departures", "end": "Arrivals"} 

    email: str
    # Comet
    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    hopsworks_api_key: str
    hopsworks_project_name: str

    database_public_url: str


config = GeneralConfig()'''

PARENT_DIR = Path("_file_").parent.resolve()

#IMAGES_DIR = Path.joinpath(PARENT_DIR, "images")
DATA_DIR = Path.joinpath(PARENT_DIR, "data")

RAW_DATA_DIR = DATA_DIR/"raw"

'''MODELS_DIR = PARENT_DIR/"models"
LOCAL_SAVE_DIR = MODELS_DIR/"locally_created"
COMET_SAVE_DIR = MODELS_DIR/"comet_downloads"

PARQUETS = RAW_DATA_DIR/"Parquets"

CLEANED_DATA = DATA_DIR/"cleaned"
TRANSFORMED_DATA = DATA_DIR/"transformed"
GEOGRAPHICAL_DATA = DATA_DIR/"geographical"

ROUNDING_INDEXER = GEOGRAPHICAL_DATA / "rounding_indexer"
MIXED_INDEXER = GEOGRAPHICAL_DATA / "mixed_indexer"

TIME_SERIES_DATA = TRANSFORMED_DATA/"time_series"
TRAINING_DATA = TRANSFORMED_DATA/"training_data"
INFERENCE_DATA = TRANSFORMED_DATA/"inference"


def make_fundamental_paths() -> None:
    for path in [
        DATA_DIR, CLEANED_DATA, RAW_DATA_DIR, PARQUETS, GEOGRAPHICAL_DATA, TRANSFORMED_DATA, TIME_SERIES_DATA, 
        IMAGES_DIR, TRAINING_DATA, INFERENCE_DATA, MODELS_DIR, LOCAL_SAVE_DIR, COMET_SAVE_DIR, ROUNDING_INDEXER,
        MIXED_INDEXER
    ]: 
        if not Path(path).exists():
            os.mkdir(path)


if __name__ == "__main__":
    os.chdir(PARENT_DIR)
    print(PARENT_DIR)
    '''
