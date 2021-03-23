from pathlib import Path

import pandas as pd


def write_csv_from_response(response, output: str):
    with open(str(output), "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
        file.flush()


def load_stm_bus_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(Path("data") / "raw" / filename)
    # df = df.astype({})
    return df
