from pathlib import Path

import geopandas as gpd
from shapely import ops
from shapely.geometry import LineString, Point
from src.preparation.constants import POINTS_IN_TRACK_FILE, PROCESSED_DATA_PATH, SNAP_FILE
from src.preparation.utils import write_spatial


def snap_points2lines(
    gdf_points: gpd.GeoDataFrame, gdf_lines: gpd.GeoDataFrame, write: bool = True
):
    gdf_snap_points = gdf_points.drop_duplicates().reset_index(drop=True).copy()
    gdf_snap_points["geometry"] = gdf_snap_points.apply(
        lambda row: gdf_lines["geometry"].interpolate(
            gdf_lines["geometry"].project(row["geometry"])
        ),
        axis=1,
    )
    gdf_snap_points.set_geometry(col="geometry", inplace=True)

    if write:
        output = Path(PROCESSED_DATA_PATH) / SNAP_FILE
        write_spatial(gdf_snap_points, output)

    return gdf_snap_points


def get_longest_track_from_bus_line(gdf_bus_tracks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Select lowest ordinal origin and highest ordinal destination
    min_ordinal = gdf_bus_tracks["ORDINAL_OR"].min()
    max_ordinal = gdf_bus_tracks["ORDINAL_DE"].max()

    # Get track that start more at the begining
    gdf_origin = gdf_bus_tracks.loc[gdf_bus_tracks["ORDINAL_OR"] == min_ordinal, :]

    # From the tracks that start more at the begining, get the track that ends more at the end
    if gdf_origin.shape[0] > 1:
        longest_from_origin = gdf_origin["ORDINAL_DE"].max()
        gdf_origin = gdf_origin.loc[gdf_origin["ORDINAL_DE"] == longest_from_origin, :]
        # In case there is an odd select first one:
        if gdf_origin.shape[0] > 1:
            gdf_origin = gdf_origin.iloc[[0]]

    # Get track that ends more at the end
    gdf_destination = gdf_bus_tracks.loc[gdf_bus_tracks["ORDINAL_DE"] == max_ordinal, :]

    # From the tracks that ends more at the end, get the track that starts more at the begining
    if gdf_destination.shape[0] > 1:
        # Get track that start more at the begining - highest range
        longest_from_destination = gdf_destination["ORDINAL_OR"].min()
        gdf_destination = gdf_destination.loc[
            gdf_destination["ORDINAL_OR"] == longest_from_destination, :
        ]
        # In case there is an odd select first one:
        if gdf_destination.shape[0] > 1:
            gdf_destination = gdf_destination.iloc[[0]]

    # Bind tracks
    gdf_union = gpd.overlay(gdf_origin, gdf_destination, how="union")
    gdf_union = gpd.GeoDataFrame(geometry=[gdf_union.unary_union], crs=32721)

    return gdf_union


def cut(line: LineString, distance: float):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:]),
            ]


def cut_tracks_by_bus_stops(
    gdf_bus_stops: gpd.GeoDataFrame,
    gdf_bus_tracks: gpd.GeoDataFrame,
    crs: int = 32721,
    write: bool = True,
):
    # Get bus line name
    bus_line = "_".join(gdf_bus_stops["DESC_LINEA"].unique().tolist())

    # Merge all points and lines
    points = gdf_bus_stops["geometry"].unary_union
    multilinestring = ops.linemerge(gdf_bus_tracks["geometry"].unary_union)

    gdf_lines = gpd.GeoDataFrame()

    # Get intersections and cut linestrings using distance
    for i, line in enumerate(multilinestring):
        line._crs = crs
        for j, point in enumerate(points):
            point._crs = crs
            d = line.project(point)
            split_line = cut(line, d)
            if len(split_line) > 1:
                gdf_line = gpd.GeoDataFrame(
                    {"line_index": i, "point_index": j, "geometry": split_line}
                )
            else:
                gdf_line = gpd.GeoDataFrame(
                    {"line_index": i, "point_index": None, "geometry": split_line}
                )
            gdf_lines = gdf_lines.append(gdf_line)

    gdf_lines = gdf_lines.set_crs(crs)
    gdf_lines.reset_index(drop=True, inplace=True)

    if write:
        write_spatial(gdf_lines, f"{POINTS_IN_TRACK_FILE}_{bus_line}")

    return gdf_lines
