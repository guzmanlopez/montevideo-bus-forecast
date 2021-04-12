import pretty_errors  # noqa
import typer
from src.preparation.constants import BUFFER, BUS_LINES, PROCESSED_FILE
from src.preparation.utils import load_pickle_file, load_stm_bus_line_track, load_stm_bus_stops
from src.processing.utils import build_bus_line_tracks_and_stops


def main(bus_line: str = "103", all_lines: bool = False, write: bool = True):
    bus_lines = BUS_LINES if all_lines else [bus_line]

    # Load processed file
    df_proc = load_pickle_file(PROCESSED_FILE)

    # Load bus stops
    gdf_bus_stops = load_stm_bus_stops()

    # Load bus tracks
    gdf_bus_tracks = load_stm_bus_line_track()

    # Add fix to line 183 because part of it track was not found
    gdf_bus_tracks.loc[gdf_bus_tracks["COD_VAR_01"] == 7603, "DESC_VARIA"] = "A"
    gdf_bus_tracks.loc[gdf_bus_tracks["COD_VAR_01"] == 7603, "COD_VAR_01"] = 8401

    for bus_line in bus_lines:
        # Build tracks
        df_proc_filtered = df_proc.loc[df_proc["dsc_linea"] == bus_line, :]
        sevar_codigo = df_proc_filtered["sevar_codigo"].unique().tolist()
        build_bus_line_tracks_and_stops(
            gdf_bus_stops, gdf_bus_tracks, sevar_codigo, BUFFER, write, bus_line
        )


if __name__ == "__main__":
    typer.run(main)