from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import ops
from shapely.geometry import LineString, Point
from shapely.geometry.multilinestring import MultiLineString
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from src.preparation.constants import (
    FILE_BUS_STOP_ORDERED,
    FILE_BUS_STOP_PROC,
    FILE_BUS_TRACK_PROC,
    PROCESSED_DATA_PATH,
    SNAP_FILE,
)
from src.preparation.typer_messages import msg_bus, msg_done, msg_process
from src.preparation.utils import write_spatial


def snap_points2lines(
    gdf_points: gpd.GeoDataFrame, gdf_lines: gpd.GeoDataFrame, write: bool = True
):
    bus_line = gdf_points["DESC_LINEA"][0]
    msg_bus(bus_line)
    msg_process("snap_points2lines")

    gdf_snap_points = gdf_points.copy()
    line_union = ops.linemerge(gdf_lines["geometry"].unary_union)
    gdf_snap_points["geometry"] = gdf_snap_points.apply(
        lambda row: line_union.interpolate(line_union.project(row["geometry"])),
        axis=1,
    )
    gdf_snap_points.set_geometry(col="geometry", inplace=True)

    if write:
        output = Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{SNAP_FILE}_{bus_line}"
        write_spatial(gdf_snap_points, output)

    msg_done()
    return gdf_snap_points


def get_longest_track_from_bus_line(gdf_bus_tracks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    msg_process("get_longest_track_from_bus_line")

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

    msg_done()
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
) -> gpd.GeoDataFrame:
    # Get bus line name
    bus_line = "_".join(gdf_bus_stops["DESC_LINEA"].unique().tolist())
    msg_process("cut_tracks_by_bus_stops for bus line")

    # Merge all points and lines
    points = gdf_bus_stops["geometry"].unary_union
    multilinestring = ops.linemerge(gdf_bus_tracks["geometry"].unary_union)

    gdf_lines = gpd.GeoDataFrame()

    if multilinestring.__class__ is not MultiLineString:
        multilinestring._crs = crs
        for j, point in enumerate(points):
            point._crs = crs
            d = multilinestring.project(point)
            split_line = cut(multilinestring, d)
            if len(split_line) > 1:
                gdf_line = gpd.GeoDataFrame(
                    {"line_index": 0, "point_index": j, "geometry": split_line}
                )
            else:
                gdf_line = gpd.GeoDataFrame(
                    {"line_index": 0, "point_index": None, "geometry": split_line}
                )
            gdf_lines = gdf_lines.append(gdf_line)
    else:
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

    # Set CRS and remove duplicates
    gdf_lines = gdf_lines.set_crs(crs)
    gdf_lines.drop_duplicates(inplace=True)
    gdf_lines.reset_index(drop=True, inplace=True)

    if write:
        write_spatial(
            gdf_lines,
            Path(PROCESSED_DATA_PATH) / "bus_tracks" / f"{FILE_BUS_TRACK_PROC}_busline_{bus_line}",
        )

    msg_done()
    return gdf_lines


def build_bus_line_tracks_and_stops(
    gdf_bus_stops: gpd.GeoDataFrame,
    gdf_bus_tracks: gpd.GeoDataFrame,
    sevar_codigo: List[int],
    buffer: float = 40,
    write: bool = True,
    bus_line: str = "",
) -> Tuple:

    msg_bus(bus_line)
    msg_process("build_bus_line_tracks_and_stops")

    # Filter bus track by sevar_codigo
    gdf_bus_tracks_filtered = gdf_bus_tracks.copy()
    gdf_bus_tracks_filtered = gdf_bus_tracks.loc[
        (gdf_bus_tracks["COD_VAR_01"].isin(sevar_codigo)) & (gdf_bus_tracks["DESC_VARIA"] == "A"),
        :,
    ]
    cod_varian = gdf_bus_tracks_filtered["COD_VAR_01"].unique()

    # Get longest track
    gdf_bus_tracks_filtered = get_longest_track_from_bus_line(gdf_bus_tracks_filtered)

    # Filter bus stops by sevar_codigo
    gdf_bus_stops_filtered = gdf_bus_stops.copy()
    gdf_bus_stops_filtered = gdf_bus_stops.loc[gdf_bus_stops["COD_VARIAN"].isin(cod_varian), :]
    gdf_bus_stops_filtered = (
        gdf_bus_stops_filtered[["DESC_LINEA", "COD_UBIC_P", "geometry"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Cut tracks by bus stations
    gdf_bus_tracks_by_stops = cut_tracks_by_bus_stops(
        gdf_bus_stops=gdf_bus_stops_filtered, gdf_bus_tracks=gdf_bus_tracks_filtered, write=write
    )

    # Remove not matched bus stops by using a buffer from bus track
    line_buff = gpd.GeoDataFrame(geometry=gdf_bus_tracks_by_stops.buffer(buffer), crs=32721)
    bus_stops = gpd.overlay(gdf_bus_stops_filtered, line_buff)["COD_UBIC_P"].unique()
    gdf_bus_stops_filtered = gdf_bus_stops_filtered.loc[
        gdf_bus_stops_filtered["COD_UBIC_P"].isin(bus_stops), :
    ]

    if write:
        write_spatial(
            gdf_bus_stops_filtered,
            Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_PROC}_{bus_line}",
        )

    return gdf_bus_stops_filtered, gdf_bus_tracks_by_stops


def sort_points_along_line_nn(
    bus_line_track: gpd.GeoDataFrame,
    n_neighbors: int = 5,
) -> gpd.GeoDataFrame:

    bus_line = bus_line_track["line"][0]
    msg_bus(bus_line)
    msg_process("sort_points_along_line")

    # Get line segments from bus line track
    list_of_lines = [line for line in bus_line_track["geometry"]]

    # Get points coordinates for each line segment
    x, y = [], []

    for line in list_of_lines:
        x_coords, y_coords = line.coords.xy
        x += x_coords
        y += y_coords

    df_xy = pd.DataFrame({"x": x, "y": y}).drop_duplicates().reset_index(drop=True)

    x, y = df_xy["x"].values, df_xy["y"].values
    points = np.c_[x, y]

    # Create n-NN graph between nodes
    clf = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    # Find the path with smallest cost from all sources
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]

    mindist = np.inf
    minidx = 0

    for i in range(len(points)):
        p = paths[i]  # order of nodes
        ordered = points[p]  # ordered nodes
        # Find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i

    # Reconstruct the order
    opt_order = paths[minidx]
    xx = x[opt_order]
    yy = y[opt_order]

    # Convert to spatial
    df_xy_sort = pd.DataFrame({"id": range(0, len(xx)), "x": xx, "y": yy})
    gdf_xy_sort = gpd.GeoDataFrame(
        df_xy_sort[["id"]],
        geometry=gpd.points_from_xy(df_xy_sort["x"], df_xy_sort["y"]),
        crs=32721,
    )

    msg_done()
    return gdf_xy_sort


def sort_points_along_line_pca(bus_line_track: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bus_line = bus_line_track["line"][0]
    msg_bus(bus_line)
    msg_process("sort_points_along_line_pca")

    # Get line segments from bus line track
    list_of_lines = [line for line in bus_line_track["geometry"]]

    # Get points coordinates for each line segment
    x, y = [], []

    for line in list_of_lines:
        x_coords, y_coords = line.coords.xy
        x += x_coords
        y += y_coords

    df_xy = pd.DataFrame({"x": x, "y": y}).drop_duplicates().reset_index(drop=True)

    x, y = df_xy["x"].values, df_xy["y"].values
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    # Make PCA object
    pca = PCA(2)
    # Fit on data
    pca.fit(xy)

    # Transform into PCA space
    xypca = pca.transform(xy)
    newx = xypca[:, 0]
    newy = xypca[:, 1]

    # Sort
    indexsort = np.argsort(x)
    newx = newx[indexsort]
    newy = newy[indexsort]

    # Return back to old coordinates:
    xyclean = pca.inverse_transform(
        np.concatenate((newx.reshape(-1, 1), newy.reshape(-1, 1)), axis=1)
    )
    xc, yc = xyclean[:, 0], xyclean[:, 1]

    # Convert to spatial
    df_xy_sort = pd.DataFrame({"id": range(0, len(xc)), "x": xc, "y": yc})
    gdf_xy_sort = gpd.GeoDataFrame(
        df_xy_sort[["id"]],
        geometry=gpd.points_from_xy(df_xy_sort["x"], df_xy_sort["y"]),
        crs=32721,
    )

    msg_done()
    return gdf_xy_sort


def get_order_of_bus_stops_along_track(
    gdf_bus_stops: gpd.GeoDataFrame,
    gdf_bus_track: gpd.GeoDataFrame,
    mode: str = "pca",
    nn: int = 5,
    write: bool = True,
) -> gpd.GeoDataFrame:
    bus_line = gdf_bus_stops["DESC_LINEA"][0]
    msg_bus(bus_line)
    msg_process("get_order_of_bus_stops_along_track\n")
    gdf_stops_snap = snap_points2lines(gdf_bus_stops, gdf_bus_track, write=write)

    if mode == "pca":
        gdf_track_sorted = sort_points_along_line_pca(gdf_bus_track)
    elif mode == "nn":
        gdf_track_sorted = sort_points_along_line_nn(gdf_bus_track, nn)

    gdf_stops_sorted = (
        gpd.overlay(
            gdf_stops_snap,
            gpd.GeoDataFrame(gdf_track_sorted, geometry=gdf_track_sorted.buffer(0.01), crs=32721),
            "identity",
        )
        .sort_values("id", ascending=False)
        .reset_index(drop=True)
    )
    gdf_stops_sorted["id"] = gdf_stops_sorted.index

    if write:
        write_spatial(
            gdf_stops_sorted,
            Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_ORDERED}_{bus_line}",
        )

    msg_done()
    return gdf_stops_sorted
