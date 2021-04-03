from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from src.preparation.constants import (
    DF_STM_VIAJES_COLS,
    FILE_STM_HORARIOS_BUSES_PARADAS,
    FILE_STM_PARADAS,
    FILE_STM_RECORRIDOS,
    FILE_STM_VIAJES_PREFIX,
    MONTH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.preparation.typer_messages import msg_done, msg_load, msg_write


def write_file_from_response(response, output: str):
    with open(str(output), "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
        file.flush()


def write_spatial(gdf: gpd.GeoDataFrame, output: str):
    gdf.to_file(f"{output}.geojson", driver="GeoJSON")
    msg_write(f"  To: {output}")


def load_stm_bus_data(month: str = MONTH, sample: int = None) -> pd.DataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_VIAJES_PREFIX}{month}.csv"
    msg_load(f"Loading {file_path}...")
    df = pd.read_csv(
        file_path,
        usecols=DF_STM_VIAJES_COLS,
        nrows=sample,
    )
    msg_done()
    return df


def load_stm_bus_time_by_bus_stop() -> pd.DataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_HORARIOS_BUSES_PARADAS}.zip"
    msg_load(f"Loading {file_path}...")
    df = pd.read_csv(file_path, delimiter=";", compression="zip")
    msg_done()
    return df


def load_stm_bus_stops() -> gpd.GeoDataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_PARADAS}.geojson"
    msg_load(f"Loading {file_path}...")
    gdf = gpd.read_file(file_path)
    msg_done()
    return gdf


def load_stm_bus_line_track() -> gpd.GeoDataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_RECORRIDOS}.geojson"
    msg_load(f"Loading {file_path}...")
    gdf = gpd.read_file(file_path)
    msg_done()
    return gdf


def save_pickle_file(df: pd.DataFrame, filename: str):
    file_path = Path(PROCESSED_DATA_PATH) / f"{filename}.pkl"
    df.to_pickle(file_path)
    msg_done(f"File saved to {file_path}")


def load_pickle_file(filename: str, dtypes: Dict[str, object] = None) -> pd.DataFrame:
    file_path = Path(PROCESSED_DATA_PATH) / f"{filename}.pkl"
    msg_load(f"Loading {file_path}...")
    df = pd.read_pickle(file_path)
    msg_done()
    if dtypes is not None:
        return df.astype(dtypes)
    else:
        return df
