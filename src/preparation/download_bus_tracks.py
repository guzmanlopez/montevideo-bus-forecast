from pathlib import Path

import geopandas as gpd
import pretty_errors  # noqa
import typer
from src.preparation.constants import (
    FILE_STM_RECORRIDOS,
    RAW_DATA_PATH,
    URL_STM_RECORRIDOS_OMNIBUS,
)
from src.preparation.typer_messages import msg_done, msg_download
from src.preparation.utils import write_spatial


def download_data(url: str = URL_STM_RECORRIDOS_OMNIBUS):
    output = Path(RAW_DATA_PATH) / FILE_STM_RECORRIDOS
    msg_download(f"  STM bus line tracks from: {url}")
    gdf = gpd.read_file(url)
    write_spatial(gdf, output)
    msg_done()


def main():
    download_data()


if __name__ == "__main__":
    typer.run(main)
