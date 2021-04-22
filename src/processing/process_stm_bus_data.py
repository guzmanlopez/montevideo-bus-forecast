import pandas as pd
import pretty_errors  # noqa
import typer
from src.preparation.constants import DAY_NAME_MAPPING, PROCESSED_FILE
from src.preparation.typer_messages import msg_done, msg_process
from src.preparation.utils import load_stm_bus_data, save_pickle_file


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    msg_process()
    df.loc[:, "fecha_evento"] = pd.to_datetime(
        df["fecha_evento"], infer_datetime_format=True, utc=False
    )
    df.set_index("fecha_evento", inplace=True)

    # Add columns with month and day name
    df.loc[:, "mes"] = df.index.month
    df.loc[:, "nombre_dia"] = df.index.day_name()
    df.loc[:, "nombre_dia"].replace(DAY_NAME_MAPPING, inplace=True)
    df.loc[:, "hora"] = df.index.hour

    # Filter
    df = df.loc[df["cantidad_pasajeros"] > 0, :]

    msg_done()
    return df


def main(month: str = typer.Option("octubre")):
    df = load_stm_bus_data(month)
    df_proc = pre_process_data(df)
    save_pickle_file(df_proc, PROCESSED_FILE)


if __name__ == "__main__":
    typer.run(main)
