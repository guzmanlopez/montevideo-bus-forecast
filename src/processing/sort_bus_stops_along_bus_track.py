import pretty_errors  # noqa
import typer
from src.preparation.constants import (
    BUS_LINES,
    LINE_DENSIFY,
    LINE_LENGTH_THRESHOLD,
    METHOD,
    NEIGHBORS,
    PROCESSED_FILE,
    TOLERANCE_DIST,
)
from src.preparation.utils import load_pickle_file, load_spatial_line
from src.processing.utils import get_order_of_bus_stops_along_track


def main(bus_line: str = "103", all_lines: bool = False):
    # Load processed file
    df_proc = load_pickle_file(PROCESSED_FILE)
    bus_lines = BUS_LINES if all_lines else [bus_line]

    for bus_line in bus_lines:
        # Read bus stops by bus line from geojson file
        gdf_stops = load_spatial_line(bus_line, type="bus_stop")

        # Read bus tracks by bus line from geojson files
        gdf_track = load_spatial_line(bus_line, type="bus_line")
        gdf_track["line"] = bus_line

        # Filter bus stops that are present in the main data df_proc
        cod_paradas = df_proc.loc[
            df_proc["dsc_linea"] == bus_line, "codigo_parada_origen"
        ].unique()
        gdf_stops = gdf_stops.loc[gdf_stops["COD_UBIC_P"].isin(cod_paradas), :]

        # bus_line_track_params = {
        #     "103": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "G": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "183": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "185": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "306": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "145": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "163": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "137": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "405": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        #     "110": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        # }
        # "103": {"method": "pca", "neighbors": None, "densify": 10, "length": 30},
        # "183": {"method": "pca", "neighbors": None, "densify": 10, "length": 30},

        get_order_of_bus_stops_along_track(
            gdf_stops,
            gdf_track,
            method=METHOD,
            neighbors=NEIGHBORS,
            line_densify=LINE_DENSIFY,
            line_length_threshold=LINE_LENGTH_THRESHOLD,
            simplify_tolerance_dist=TOLERANCE_DIST,
            write=True,
        )


if __name__ == "__main__":
    typer.run(main)
