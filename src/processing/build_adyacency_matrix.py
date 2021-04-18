import pretty_errors  # noqa
import typer
from src.preparation.constants import FILE_ADYACENCY_MATRIX, FILE_FROM_TO_WEIGHT
from src.preparation.utils import save_df_to_csv
from src.processing.utils import build_adyacency_matrix


def main(
    control: bool = typer.Option(True),
    diff_value: int = typer.Option(50),
    bus_stop_code_control_1: int = typer.Option(1283),
    bus_stop_code_control_2: int = typer.Option(1284),
):
    df_adyacency_matrix, df_from_to_weight = build_adyacency_matrix(
        control, diff_value, bus_stop_code_control_1, bus_stop_code_control_2
    )
    save_df_to_csv(df_adyacency_matrix, FILE_ADYACENCY_MATRIX)
    save_df_to_csv(df_from_to_weight, FILE_FROM_TO_WEIGHT)


if __name__ == "__main__":
    typer.run(main)
