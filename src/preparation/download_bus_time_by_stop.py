from pathlib import Path

import pretty_errors  # noqa
import requests
import typer
from src.preparation.constants import (
    FILE_STM_HORARIOS_BUSES_PARADAS,
    RAW_DATA_PATH,
    URL_HORARIOS_OMNIBUS_POR_PARADAS,
)
from src.preparation.typer_messages import msg_done, msg_download, msg_write
from src.preparation.utils import write_file_from_response


def download_data(url: str = URL_HORARIOS_OMNIBUS_POR_PARADAS):
    output = Path(RAW_DATA_PATH) / f"{FILE_STM_HORARIOS_BUSES_PARADAS}.zip"
    msg_download(f"  STM bus time data by bus stop from: {url}")
    with requests.get(
        url,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as response:
        response.raise_for_status()
        msg_write(f"  To: {output}")
        write_file_from_response(response, output)
        msg_done()


def main():
    download_data()


if __name__ == "__main__":
    typer.run(main)
