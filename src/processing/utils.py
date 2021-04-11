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
    CRS,
    FILE_BUS_STOP_ORDERED,
    FILE_BUS_STOP_PROC,
    FILE_BUS_STOP_SNAP,
    FILE_BUS_TRACK_ORDERED,
    FILE_BUS_TRACK_PROC,
    PROCESSED_DATA_PATH,
)
from src.preparation.typer_messages import msg_bus, msg_done, msg_info, msg_process, msg_warn
from src.preparation.utils import write_spatial


def snap_points2lines(
    gdf_points: gpd.GeoDataFrame, gdf_tracks: gpd.GeoDataFrame, write: bool = True
):
    bus_line = gdf_points["DESC_LINEA"].unique()[0]
    msg_process("Snap bus stations to bus tracks")

    track = gdf_tracks["geometry"].unary_union
    gdf_snap_points = gdf_points.copy()
    gdf_snap_points["geometry"] = gdf_snap_points.apply(
        lambda row: track.interpolate(track.project(row["geometry"])),
        axis=1,
    )
    gdf_snap_points.set_geometry(col="geometry", inplace=True)
    gdf_snap_points = gdf_snap_points.set_crs(CRS)

    if write:
        output = Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_SNAP}_{bus_line}"
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
    gdf_union = gpd.GeoDataFrame(geometry=[gdf_union.unary_union], crs=CRS)

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
    crs: int = CRS,
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
    line_buff = gpd.GeoDataFrame(geometry=gdf_bus_tracks_by_stops.buffer(buffer), crs=CRS)
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
    bus_line_track: MultiLineString,
    neighbors: int = 5,
) -> gpd.GeoDataFrame:

    msg_process("Sort points along a path using Nearest Neighbors and minimizing path cost")
    msg_info(f"Using {neighbors} neighbors")

    # Get points coordinates for each line segment
    x, y = [], []

    for line in bus_line_track:
        x_coords, y_coords = line.coords.xy
        x += x_coords
        y += y_coords

    df_xy = pd.DataFrame({"x": x, "y": y}).drop_duplicates().reset_index(drop=True)
    x, y = df_xy["x"].values, df_xy["y"].values
    points = np.c_[x, y]

    # Create n-NN graph between nodes
    clf = NearestNeighbors(n_neighbors=neighbors, algorithm="brute", n_jobs=-1).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G, create_using=nx.DiGraph)

    # Find the path with smallest cost from all sources
    paths = [
        list(nx.dfs_preorder_nodes(G=T, source=i, depth_limit=len(T))) for i in range(len(points))
    ]
    max_len = len(max(paths, key=len))
    msg = f"{max_len}/{points.shape[0]} nodes used to calculate shortest path"

    if (max_len / points.shape[0]) == 1:
        msg_info(msg)
    else:
        msg_warn(msg)

    min_dist = np.inf
    min_idx = 0

    for i in range(len(points)):
        node_order = paths[i]
        if len(node_order) == max_len:
            ordered = points[node_order]
            # Find cost of order by the sum of euclidean distances between points (i) and (i+1)
            cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
            if cost < min_dist:
                min_dist = cost
                min_idx = i

    opt_order = paths[min_idx]
    xx = x[opt_order]
    yy = y[opt_order]

    # Convert to spatial
    df_xy_sort = pd.DataFrame({"id": range(0, len(xx)), "x": xx, "y": yy})
    gdf_xy_sort = gpd.GeoDataFrame(
        df_xy_sort[["id"]],
        geometry=gpd.points_from_xy(df_xy_sort["x"], df_xy_sort["y"], crs=CRS),
        crs=CRS,
    )

    msg_done()
    return gdf_xy_sort


def sort_points_along_line_pca(
    bus_line_track: MultiLineString,
) -> gpd.GeoDataFrame:
    bus_line = bus_line_track["line"][0]
    msg_bus(bus_line)
    msg_process("Sort bus stops along bus tracks using PCA")

    # Get points coordinates for each line segment
    x, y = [], []

    for line in bus_line_track:
        x_coords, y_coords = line.coords.xy
        x += x_coords
        y += y_coords

    df_xy = pd.DataFrame({"x": x, "y": y}).drop_duplicates().reset_index(drop=True)

    x, y = df_xy["x"].values, df_xy["y"].values
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    print(f"{xy.size} points")

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
        geometry=gpd.points_from_xy(df_xy_sort["x"], df_xy_sort["y"], crs=CRS),
        crs=CRS,
    )

    msg_done()
    return gdf_xy_sort


def get_order_of_bus_stops_along_track(
    gdf_bus_stops: gpd.GeoDataFrame,
    gdf_bus_track: gpd.GeoDataFrame,
    method: str = "nn",
    neighbors: int = 8,
    line_densify: int = 10,
    line_length_threshold: int = None,
    simplify_tolerance_dist: float = 10,
    write: bool = True,
) -> gpd.GeoDataFrame:
    bus_line = gdf_bus_stops["DESC_LINEA"].unique()[0]
    msg_bus(bus_line)
    msg_process("Get order of bus stops along bus track")
    msg_warn("This operation can take some time \n")

    linestrings = simplify_linestring(gdf_bus_track, simplify_tolerance_dist)

    if line_length_threshold is not None:
        msg_process("Densyfing bus track by adding points at equal intervals")
        msg_info(f"Densify line lengths equal or above {line_length_threshold} meters")
        msg_info(f"Add points every {line_densify} meters")
        initial_number_of_points = sum([len(line.coords) for line in linestrings])
        # Get line segments from bus line track
        linestrings = [
            densify_linestring(line, line_densify)
            if line.length >= line_length_threshold
            else line
            for line in linestrings
        ]
        linestrings = MultiLineString(linestrings)
        final_number_of_points = sum([len(line.coords) for line in linestrings])
        points_added = final_number_of_points - initial_number_of_points
        perc_added = 100 * (points_added / initial_number_of_points)
        msg_info(
            f"Added {points_added} new points of {initial_number_of_points} initial points ({round(perc_added, 1)}%) \n"
        )

    if method == "pca":
        gdf_track_sorted = sort_points_along_line_pca(linestrings)
    elif method == "nn":
        gdf_track_sorted = sort_points_along_line_nn(linestrings, neighbors)

    track_sorted_lines = []

    for i in range(0, (len(gdf_track_sorted) - 1)):
        p_start = gdf_track_sorted.iloc[[i]]["geometry"].values[0]
        p_end = gdf_track_sorted.iloc[[i + 1]]["geometry"].values[0]
        line = LineString([p_start, p_end])
        track_sorted_lines.append(line)

    gdf_track_sorted_lines = gpd.GeoDataFrame(geometry=track_sorted_lines, crs=CRS)
    gdf_track_sorted_lines["id"] = gdf_track_sorted_lines.index

    # Snap bus stations to new bus track
    gdf_stops_snap = snap_points2lines(gdf_bus_stops, gdf_track_sorted_lines, write=True)

    # Add buffer to bus track to match the snap of bus stops
    gdf_track_sorted_lines_buffer = gdf_track_sorted_lines.copy()
    gdf_track_sorted_lines_buffer = gpd.GeoDataFrame(
        gdf_track_sorted_lines_buffer,
        geometry=gdf_track_sorted_lines_buffer.buffer(0.01, cap_style=2),
        crs=CRS,
    )

    gdf_stops_sorted = gpd.sjoin(
        gdf_stops_snap, gdf_track_sorted_lines_buffer, how="left", op="intersects"
    )
    gdf_stops_sorted = gdf_stops_sorted.sort_values("id").reset_index(drop=True)
    gdf_stops_sorted["idx"] = gdf_stops_sorted.index
    gdf_stops_sorted.drop(columns="index_right")

    if write:
        write_spatial(
            gdf_stops_sorted,
            Path(PROCESSED_DATA_PATH) / "bus_stops" / f"{FILE_BUS_STOP_ORDERED}_{bus_line}",
        )
        write_spatial(
            gdf_track_sorted_lines,
            Path(PROCESSED_DATA_PATH) / "bus_tracks" / f"{FILE_BUS_TRACK_ORDERED}_{bus_line}",
        )

    msg_done()
    return gdf_stops_sorted


def densify_linestring(line: LineString, x: int = 10) -> LineString:
    # Create a linespace of points every x meters
    lenspace = np.linspace(0, line.length, int(line.length / x) + 1)
    new_points = []
    for space in lenspace:
        new_point = line.interpolate(space)
        new_points.append(new_point)
    dense_line = LineString(new_points)
    return dense_line


def simplify_linestring(
    gdf_line: gpd.GeoDataFrame, simplify_tolerance_dist: float
) -> MultiLineString:
    msg_process("Simplifying bus track")
    msg_info(f"Using {simplify_tolerance_dist} meters of tolerance")
    union_line = gdf_line["geometry"].unary_union
    simple_line = ops.linemerge(union_line).simplify(simplify_tolerance_dist)
    # Messages
    initial_number_of_points = sum([len(line.coords) for line in union_line])
    final_number_of_points = sum([len(line.coords) for line in simple_line])
    points_removed = initial_number_of_points - final_number_of_points
    perc_removed = 100 * (points_removed / initial_number_of_points)
    msg_info(
        f"Removed {points_removed} points of {initial_number_of_points} ({round(perc_removed, 1)}%) \n"
    )
    return simple_line
