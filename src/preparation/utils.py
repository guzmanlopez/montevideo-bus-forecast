from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from src.preparation.constants import (
    DF_STM_VIAJES_COLS,
    FILE_ADYACENCY_MATRIX,
    FILE_BUS_STOP_ORDERED,
    FILE_BUS_STOP_PROC,
    FILE_BUS_STOP_SNAP,
    FILE_BUS_TRACK_ORDERED,
    FILE_BUS_TRACK_PROC,
    FILE_FROM_TO_WEIGHT,
    FILE_STM_HORARIOS_BUSES_PARADAS,
    FILE_STM_PARADAS,
    FILE_STM_RECORRIDOS,
    FILE_STM_VIAJES_PREFIX,
    MONTH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.preparation.typer_messages import (
    msg_bus,
    msg_bus_stop,
    msg_bus_track,
    msg_load,
    msg_write,
)


def write_file_from_response(response, output: str):
    with open(str(output), "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
        file.flush()


def write_spatial(gdf: gpd.GeoDataFrame, output: str):
    gdf.to_file(f"{output}.geojson", driver="GeoJSON")
    msg_write(f"Saved to: {output}.geojson\n")


def load_stm_bus_data(month: str = MONTH, sample: int = None) -> pd.DataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_VIAJES_PREFIX}{month}.csv"
    msg_load(f"Loading {file_path}...")
    df = pd.read_csv(
        file_path,
        usecols=DF_STM_VIAJES_COLS,
        nrows=sample,
    )
    return df


def load_stm_bus_time_by_bus_stop() -> pd.DataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_HORARIOS_BUSES_PARADAS}.zip"
    msg_load(f"Loading {file_path}...")
    df = pd.read_csv(file_path, delimiter=";", compression="zip")
    return df


def load_stm_bus_stops() -> gpd.GeoDataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_PARADAS}.geojson"
    msg_load(f"Loading {file_path}...")
    gdf = gpd.read_file(file_path)
    return gdf


def load_stm_bus_line_track() -> gpd.GeoDataFrame:
    file_path = Path(RAW_DATA_PATH) / f"{FILE_STM_RECORRIDOS}.geojson"
    msg_load(f"Loading {file_path}...")
    gdf = gpd.read_file(file_path)
    return gdf


def load_spatial_data(bus_line: str, type: str = "bus_stop") -> gpd.GeoDataFrame:
    msg_bus(bus_line)
    if type == "bus_stop":
        file_path = (
            Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_PROC}_{bus_line}.geojson"
        )
        msg_bus_stop("Bus stops")

    elif type == "bus_stop_ordered":
        file_path = (
            Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_ORDERED}_{bus_line}.geojson"
        )
        msg_bus_stop("Bus stops")

    elif type == "bus_stop_snap":
        file_path = (
            Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_SNAP}_{bus_line}.geojson"
        )
        msg_bus_stop("Bus stops")

    elif type == "bus_line":
        file_path = (
            Path(PROCESSED_DATA_PATH)
            / "bus_tracks"
            / f"{FILE_BUS_TRACK_PROC}_busline_{bus_line}.geojson"
        )
        msg_bus_track("Bus tracks")

    elif type == "bus_track_ordered":
        file_path = (
            Path(PROCESSED_DATA_PATH)
            / "bus_tracks"
            / f"{FILE_BUS_TRACK_ORDERED}_{bus_line}.geojson"
        )
        msg_bus_track("Bus tracks")

    msg_load(f"Loading file {file_path}...")
    gdf = gpd.read_file(file_path)
    return gdf


def load_adyacency_data():
    file_path = Path(PROCESSED_DATA_PATH) / f"{FILE_ADYACENCY_MATRIX}.csv"
    msg_load(f"Loading {file_path}...")
    df = pd.read_csv(file_path, index_col=0)
    return df


def load_edges_data():
    file_path = Path(PROCESSED_DATA_PATH) / f"{FILE_FROM_TO_WEIGHT}.csv"
    msg_load(f"Loading {file_path}...")
    df = pd.read_csv(file_path, index_col=0)
    return df


def save_pickle_file(df: pd.DataFrame, filename: str):
    file_path = Path(PROCESSED_DATA_PATH) / f"{filename}.pkl"
    df.to_pickle(file_path)
    msg_write(f"File saved to {file_path}\n")


def save_df_to_csv(df: pd.DataFrame, filename: str):
    file_path = Path(PROCESSED_DATA_PATH) / f"{filename}.csv"
    df.to_csv(file_path)
    msg_write(f"File saved to {file_path}\n")


def load_pickle_file(filename: str, dtypes: Dict[str, object] = None) -> pd.DataFrame:
    file_path = Path(PROCESSED_DATA_PATH) / f"{filename}.pkl"
    msg_load(f"Loading {file_path}...")
    df = pd.read_pickle(file_path)
    if dtypes is not None:
        return df.astype(dtypes)
    else:
        return df
