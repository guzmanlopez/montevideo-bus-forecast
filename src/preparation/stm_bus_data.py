from pathlib import Path

import pretty_errors
import requests
import typer
from src.preparation.constants import (
    URL_STM_VIAJES_DICIEMBRE,
    URL_STM_VIAJES_NOVIEMBRE,
    URL_STM_VIAJES_OCTUBRE,
)
from src.preparation.typer_messages import msg_done, msg_download, msg_writte
from src.preparation.utils import write_csv_from_response


def download_data(url: str = URL_STM_VIAJES_OCTUBRE, month: str = "octubre"):
    output = Path("data") / "raw" / f"stm_viajes_{month}.csv"
    msg_download(f"Downloading [{month}] bus data from: {url}...")
    with requests.get(
        url,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as response:
        response.raise_for_status()
        msg_writte(f"Writting data to: {output}")
        write_csv_from_response(response, output)
        msg_done("Done!")


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
