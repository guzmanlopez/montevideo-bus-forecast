# %% [markdown]
# # Análisis Exploratorio de Datos

# %%

import geopandas as gpd
import pandas as pd
from notebooks.eda.plots import (
    plot_boardings_by_day_name,
    plot_boardings_by_hour_and_day_name,
    plot_boardings_by_time,
)
from src.preparation.constants import BUS_LINES, CRS, DAY_NAME_MAPPING, PROCESSED_FILE
from src.preparation.utils import (
    load_pickle_file,
    load_spatial_line,
    load_stm_bus_data,
    load_stm_bus_line_track,
    load_stm_bus_stops,
    save_pickle_file,
)
from src.processing.process_stm_bus_data import pre_process_data
from src.processing.utils import (
    build_bus_line_tracks_and_stops,
    get_order_of_bus_stops_along_track,
    sort_points_along_line_nn,
    sort_points_along_line_pca,
)

# %%
# Load data and pre-process data
df = pre_process_data(load_stm_bus_data())

# %%[markdown]
# ## Plots

# %%
df_hourly = df.groupby([pd.Grouper(freq="1H")])["cantidad_pasajeros"].sum().reset_index()
plot_boardings_by_time(df_hourly)

# %%
# Daily sum of boardings and median by day name
df_day_name = (
    df.groupby([pd.Grouper(freq="1D"), "nombre_dia"])["cantidad_pasajeros"]
    .sum()
    .groupby(["nombre_dia"])
    .median()
    .reset_index()
)
plot_boardings_by_day_name(df_day_name)

# %%
# Hourly sum of boardings and median by day name
df_hourly_day_name = df_hourly.copy()
df_hourly_day_name.set_index("fecha_evento", inplace=True)
df_hourly_day_name.loc[:, "nombre_dia"] = df_hourly_day_name.index.day_name()
df_hourly_day_name.loc[:, "nombre_dia"].replace(DAY_NAME_MAPPING, inplace=True)
df_hourly_day_name.loc[:, "hora"] = df_hourly_day_name.index.hour
df_hourly_day_name = df_hourly_day_name.groupby(["hora", "nombre_dia"]).median().reset_index()

plot_boardings_by_hour_and_day_name(df_hourly_day_name)

# %%
# Get top buses lines per day of the week
df_weekly_by_day_name_and_line = (
    df.groupby([pd.Grouper(freq="1D"), "nombre_dia", "dsc_linea"])["cantidad_pasajeros"]
    .sum()
    .groupby(["dsc_linea", "nombre_dia"])
    .median()
    .groupby("dsc_linea")
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

df_weekly_by_day_name_and_line["decile_rank"] = pd.qcut(
    df_weekly_by_day_name_and_line["cantidad_pasajeros"], 10, labels=False
)

# Contribution of each decile
df_decile_rank_prop = df_weekly_by_day_name_and_line.groupby("decile_rank").sum().reset_index()
df_decile_rank_prop["proportion"] = (
    df_decile_rank_prop["cantidad_pasajeros"] / df_decile_rank_prop["cantidad_pasajeros"].sum()
)

# Select bus lines from the 9th decile
df_bus_lines = df_weekly_by_day_name_and_line.loc[
    df_weekly_by_day_name_and_line["decile_rank"] == 9, :
]
df_bus_lines = df_bus_lines.sort_values("cantidad_pasajeros", ascending=False)

# %%
# Filter by bus line
df_proc = df.loc[df["dsc_linea"].isin(df_bus_lines["dsc_linea"].tolist()), :]

df_proc = (
    df_proc.groupby(
        [
            pd.Grouper(freq="1H"),
            "nombre_dia",
            "codigo_parada_origen",
            "dsc_linea",
            "sevar_codigo",
            "ordinal_de_tramo",
        ]
    )["cantidad_pasajeros"]
    .sum()
    .reset_index()
    .sort_values("fecha_evento", ascending=True)
    .reset_index(drop=True)
)

save_pickle_file(df_proc, PROCESSED_FILE)

# %% [markdown]
# ## Build bus line tracks

# %%
# Load processed file
df_proc = load_pickle_file(PROCESSED_FILE)

# Load bus stops
# gdf_bus_stops = load_stm_bus_stops()

# Load bus tracks
# gdf_bus_tracks = load_stm_bus_line_track()

# Add fix to 183 because part of it track was not found
# gdf_bus_tracks.loc[gdf_bus_tracks["COD_VAR_01"] == 7603, "DESC_VARIA"] = "A"
# gdf_bus_tracks.loc[gdf_bus_tracks["COD_VAR_01"] == 7603, "COD_VAR_01"] = 8401

# %%
# Build tracks
for bus_line in BUS_LINES:
    df_proc_filtered = df_proc.loc[df_proc["dsc_linea"] == bus_line, :]
    sevar_codigo = df_proc_filtered["sevar_codigo"].unique().tolist()
    build_bus_line_tracks_and_stops(
        gdf_bus_stops, gdf_bus_tracks, sevar_codigo, 40, True, bus_line
    )

# %%
# Read all bus stops by bus line from geojson files
all_bus_stops = gpd.GeoDataFrame()
for bus_line in BUS_LINES:
    all_bus_stops = all_bus_stops.append(load_spatial_line(bus_line, type="bus_stop"))
all_bus_stops = all_bus_stops.set_crs(CRS)

# Read all bus tracks by bus line from geojson files
all_bus_tracks = gpd.GeoDataFrame()
for bus_line in BUS_LINES:
    df = load_spatial_line(bus_line, type="bus_line")
    df["line"] = bus_line
    all_bus_tracks = all_bus_tracks.append(df)
all_bus_tracks = all_bus_tracks.set_crs(CRS)

# %%
# Get order of points in track
# for bus_line in BUS_LINES:
#     gdf_stops = all_bus_stops.loc[all_bus_stops["DESC_LINEA"] == bus_line, :]
#     gdf_track = all_bus_tracks.loc[all_bus_tracks["line"] == bus_line, :]

#     if bus_line in [""]:
#         get_order_of_bus_stops_along_track(gdf_stops, gdf_track, mode="nn", nn=8, write=True)
#     elif bus_line in [""]:
#         get_order_of_bus_stops_along_track(gdf_stops, gdf_track, mode="pca", write=True)


# %%
bus_line = "G"
gdf_stops = all_bus_stops.loc[all_bus_stops["DESC_LINEA"] == bus_line, :]
gdf_track = all_bus_tracks.loc[all_bus_tracks["line"] == bus_line, :]

# Filter bus stops that are present in the main data df_proc
cod_paradas = df_proc.loc[df_proc["dsc_linea"] == bus_line, "codigo_parada_origen"].unique()
gdf_stops = gdf_stops.loc[gdf_stops["COD_UBIC_P"].isin(cod_paradas), :]

# %%
# TODO: overlay gdf_bus_track / gdf_track_sorted y medir distancias entre paradas sumando distancias entre puntos
# Distance between bus stations along track
bus_line = "G"
gdf_bus_track = load_spatial_line(bus_line, type="bus_stop_ordered")
gdf_track = all_bus_tracks.loc[all_bus_tracks["line"] == bus_line, :]
gdf_track_sorted = sort_points_along_line_pca(gdf_track)
gdf_track_sorted = sort_points_along_line_nn(gdf_track, 8)

gdf_track_sorted.to_file("./data/processed/test.geojson", "GeoJSON")

# %%


# %%
shared_bus_stops = (
    all_bus_stops.groupby(["COD_UBIC_P"])
    .agg(lines=("DESC_LINEA", "|".join), number_of_lines=("DESC_LINEA", len))
    .round(0)
    .sort_values("COD_UBIC_P", ascending=True)
    .reset_index()
    .astype({"COD_UBIC_P": int})
)

shared_bus_stops.loc[shared_bus_stops["number_of_lines"] > 1, :]["lines"].unique()


# %%
bus_line = "103"

df_bus_line = (
    df.loc[df["dsc_linea"] == bus_line, :][
        ["codigo_parada_origen", "sevar_codigo", "ordinal_de_tramo"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
    .groupby("codigo_parada_origen")
    .agg({"ordinal_de_tramo": "min"})
    .sort_values("ordinal_de_tramo", ascending=True)
    .reset_index()
)

# Create the dictionary that defines the order for sorting
sorter_list = df_bus_line["codigo_parada_origen"].unique()
sorter_index = dict(zip(sorter_list, range(len(sorter_list))))

stops_103 = all_bus_stops.loc[all_bus_stops["DESC_LINEA"] == "103", :]
track_103 = all_bus_tracks.loc[all_bus_tracks["line"] == "103", :]

stops_103 = stops_103.assign(
    bus_stop_order=stops_103["COD_UBIC_P"].map(sorter_index).fillna(-1).astype(int).values
)
stops_103 = stops_103.sort_values("bus_stop_order", ascending=True).reset_index(drop=True)
stops_103.to_file("test.geojson", "GeoJSON")

# %%
# Build distance matrix between bus stops
all_bus_stops_unique = all_bus_stops.copy()
all_bus_stops_unique = (
    all_bus_stops_unique[["COD_UBIC_P", "geometry"]]
    .drop_duplicates()
    .sort_values("COD_UBIC_P", ascending=True)
    .reset_index(drop=True)
)

distance_matrix = all_bus_stops_unique.geometry.apply(lambda x: all_bus_stops_unique.distance(x))
distance_matrix = distance_matrix.round(0).astype(int)

names = all_bus_stops_unique["COD_UBIC_P"].round(0).astype(int).tolist()
distance_matrix.columns = names
distance_matrix.index = names

# %%
# Build connection matrix between bus stops
connection_matrix = distance_matrix.copy()
connection_matrix.loc[:, :] = 0

for bus_stop in shared_bus_stops["COD_UBIC_P"]:
    bus_lines = (
        shared_bus_stops.loc[shared_bus_stops["COD_UBIC_P"] == bus_stop, "lines"]
        .tolist()[0]
        .split("|")
    )
    for bus_line in bus_lines:
        linked_bus_stops = shared_bus_stops.loc[
            shared_bus_stops["lines"].str.contains(bus_line), "COD_UBIC_P"
        ].tolist()

        connection_matrix.loc[bus_stop, linked_bus_stops] = 1
        connection_matrix.loc[linked_bus_stops, bus_stop] = 1

# %%
linked_bus_stops = shared_bus_stops.loc[
    shared_bus_stops["lines"].str.contains("405"), "COD_UBIC_P"
].tolist()


connection_matrix.loc[linked_bus_stops, linked_bus_stops]


# %%
import geopandas as gpd

points = load_spatial_line("G", "bus_stop_snap")
lines = load_spatial_line("G", "bus_line_ordered")

# Add buffer to bus track to match again the snap of bus stops
line_buffer = lines.copy()
line_buffer = gpd.GeoDataFrame(line_buffer, geometry=line_buffer.buffer(0.01), crs=CRS)

gdf_stops_sorted = gpd.sjoin(points, line_buffer, how="left", op="within")
gdf_stops_sorted = gdf_stops_sorted.dropna()

gdf_stops_sorted = gdf_stops_sorted.sort_values("id").reset_index(drop=True)
gdf_stops_sorted["id_new"] = gdf_stops_sorted.index
gdf_stops_sorted.drop(columns="index_right")

# %%
