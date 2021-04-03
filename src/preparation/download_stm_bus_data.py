from pathlib import Path

import pretty_errors  # noqa
import requests
import typer
from src.preparation.constants import (
    FILE_STM_VIAJES_PREFIX,
    RAW_DATA_PATH,
    URL_STM_VIAJES_DICIEMBRE,
    URL_STM_VIAJES_NOVIEMBRE,
    URL_STM_VIAJES_OCTUBRE,
)
from src.preparation.typer_messages import msg_done, msg_download, msg_write
from src.preparation.utils import write_file_from_response


def download_data(url: str = URL_STM_VIAJES_OCTUBRE, month: str = "octubre"):
    output = Path(RAW_DATA_PATH) / f"{FILE_STM_VIAJES_PREFIX}{month}.csv"
    msg_download(f"  STM bus data for: {month.title()} from: {url}")
    with requests.get(
        url,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as response:
        response.raise_for_status()
        msg_write(f"  To: {output}")
        write_file_from_response(response, output)
        msg_done()


def main(month: str = "octubre"):
    if month == "noviembre":
        url = URL_STM_VIAJES_NOVIEMBRE
    elif month == "diciembre":
        url = URL_STM_VIAJES_DICIEMBRE
    else:
        url = URL_STM_VIAJES_OCTUBRE
    download_data(url, month)


if __name__ == "__main__":
    typer.run(main)
