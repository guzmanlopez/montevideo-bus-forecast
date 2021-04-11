import pretty_errors  # noqa
import typer
from src.preparation.constants import PROCESSED_FILE, TOLERANCE_DIST
from src.preparation.utils import load_pickle_file, load_spatial_line
from src.processing.utils import get_order_of_bus_stops_along_track


def main(bus_line: str = "103"):
    # Load processed file
    df_proc = load_pickle_file(PROCESSED_FILE)

    # Read bus stops by bus line from geojson file
    gdf_stops = load_spatial_line(bus_line, type="bus_stop")

    # Read bus tracks by bus line from geojson files
    gdf_track = load_spatial_line(bus_line, type="bus_line")
    gdf_track["line"] = bus_line

    # Filter bus stops that are present in the main data df_proc
    cod_paradas = df_proc.loc[df_proc["dsc_linea"] == bus_line, "codigo_parada_origen"].unique()
    gdf_stops = gdf_stops.loc[gdf_stops["COD_UBIC_P"].isin(cod_paradas), :]

    bus_line_track_params = {
        # "103": {"method": "pca", "neighbors": None, "densify": 10, "length": 30},
        "G": {"method": "nn", "neighbors": 3, "densify": 5, "length": 5},
        # "183": {"method": "pca", "neighbors": None, "densify": 10, "length": 30},
        "185": {"method": "nn", "neighbors": 5, "densify": 25, "length": 50},
        "306": {"method": "nn", "neighbors": 8, "densify": 10, "length": 20},
        "145": {"method": "nn", "neighbors": 8, "densify": 10, "length": 20},
        "163": {"method": "nn", "neighbors": 8, "densify": 10, "length": 20},
        "137": {"method": "nn", "neighbors": 8, "densify": 10, "length": 20},
        "405": {"method": "nn", "neighbors": 8, "densify": 10, "length": 20},
        "110": {"method": "nn", "neighbors": 8, "densify": 10, "length": 20},
    }

    get_order_of_bus_stops_along_track(
        gdf_stops,
        gdf_track,
        method=bus_line_track_params.get(bus_line).get("method"),
        neighbors=bus_line_track_params.get(bus_line).get("neighbors"),
        line_densify=bus_line_track_params.get(bus_line).get("densify"),
        line_length_threshold=bus_line_track_params.get(bus_line).get("length"),
        simplify_tolerance_dist=TOLERANCE_DIST,
        write=True,
    )


if __name__ == "__main__":
    typer.run(main)
