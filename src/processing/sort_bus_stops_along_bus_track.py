from typing import List

import pretty_errors  # noqa
import typer
from src.preparation.constants import BUS_LINE_TRACK_PARS, BUS_LINES, METHOD, PROCESSED_FILE
from src.preparation.utils import load_pickle_file, load_spatial_data
from src.processing.utils import fix_bus_stop_order_length, get_order_of_bus_stops_along_track


def main(bus_lines: List[str] = typer.Option(BUS_LINES)):
    # Load processed file
    df_proc = load_pickle_file(PROCESSED_FILE)

    for bus_line in bus_lines:
        # Read bus stops by bus line from geojson file
        gdf_stops = load_spatial_data(bus_line, type="bus_stop")

        # Read bus tracks by bus line from geojson files
        gdf_track = load_spatial_data(bus_line, type="bus_line")
        gdf_track["line"] = bus_line

        # Filter bus stops that are present in the main data df_proc
        cod_paradas = df_proc.loc[
            df_proc["dsc_linea"] == bus_line, "codigo_parada_origen"
        ].unique()
        gdf_stops = gdf_stops.loc[gdf_stops["COD_UBIC_P"].isin(cod_paradas), :]

        get_order_of_bus_stops_along_track(
            gdf_stops,
            gdf_track,
            method=METHOD,
            neighbors=BUS_LINE_TRACK_PARS.get(bus_line).get("neighbors"),
            line_densify=BUS_LINE_TRACK_PARS.get(bus_line).get("densify"),
            line_length_threshold=BUS_LINE_TRACK_PARS.get(bus_line).get("length"),
            simplify_tolerance_dist=BUS_LINE_TRACK_PARS.get(bus_line).get("tolerance"),
            write=True,
        )
        fix_bus_stop_order_length(bus_line)


if __name__ == "__main__":
    typer.run(main)
