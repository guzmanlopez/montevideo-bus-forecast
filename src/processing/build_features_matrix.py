import pretty_errors  # noqa
import typer
from src.preparation.constants import FILE_FEATURES_MATRIX
from src.preparation.typer_messages import msg_done, msg_info, msg_process
from src.preparation.utils import save_df_to_csv
from src.processing.utils import get_features_matrix


def main(
    freq: str = typer.Option("1H"),
):
    msg_process("Building feature matrix\n")
    msg_info(f"Resample time {freq}\n")
    df_features = get_features_matrix(freq=freq)
    save_df_to_csv(df_features, FILE_FEATURES_MATRIX)
    msg_done()


if __name__ == "__main__":
    typer.run(main)
