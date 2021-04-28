# %%[markdown]
# # Análisis Exploratorio Datos

# %%
import altair as alt
import geopandas as gpd
import networkx as nx
import pandas as pd
from notebooks.eda.plots import (
    network_bokeh_plot,
    plot_boardings_by_day_name,
    plot_boardings_by_hour_and_day_name,
    plot_boardings_by_time,
)
from src.preparation.constants import BUS_LINES, CRS, DAY_NAME_MAPPING, FILE_GRAPH, PROCESSED_FILE
from src.preparation.utils import (
    load_pickle_file,
    load_spatial_data,
    load_stm_bus_line_track,
    load_stm_bus_stops,
)

alt.renderers.enable("colab")
# %%
# Load data pre-processed data
df_proc = load_pickle_file(PROCESSED_FILE)
df_proc
# %%
df_hourly = df_proc.groupby([pd.Grouper(freq="1H")])["cantidad_pasajeros"].sum().reset_index()
plot_boardings_by_time(df_hourly)
# %%
# Daily sum of boardings and median by day name
df_day_name = (
    df_proc.groupby([pd.Grouper(freq="1D"), "nombre_dia"])["cantidad_pasajeros"]
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
    df_proc.groupby([pd.Grouper(freq="1D"), "nombre_dia", "dsc_linea"])["cantidad_pasajeros"]
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
df_decile_rank_prop
# %%
# Select bus lines from the highest decile
df_bus_lines = df_weekly_by_day_name_and_line.loc[
    df_weekly_by_day_name_and_line["decile_rank"] == 9, :
]
df_bus_lines = df_bus_lines.sort_values("cantidad_pasajeros", ascending=False)
df_bus_lines
# %%
# Load bus stops
gdf_bus_stops = load_stm_bus_stops()

# Load bus tracks
gdf_bus_tracks = load_stm_bus_line_track()

# Load ordered bus stops and bus tracks
all_bus_stops, all_bus_tracks = gpd.GeoDataFrame(), gpd.GeoDataFrame()

for bus_line in BUS_LINES:
    # Read all bus stops by bus line from geojson files
    all_bus_stops = all_bus_stops.append(load_spatial_data(bus_line, type="bus_stop_ordered"))

    # Read all bus tracks by bus line from geojson files
    df = load_spatial_data(bus_line, type="bus_track_ordered")
    df["line"] = bus_line
    all_bus_tracks = all_bus_tracks.append(df)

all_bus_stops = all_bus_stops.set_crs(CRS)
all_bus_tracks = all_bus_tracks.set_crs(CRS)
# %%
# Check shared bus stops by lines
shared_bus_stops = (
    all_bus_stops.groupby(["COD_UBIC_P"])
    .agg(lines=("DESC_LINEA", "|".join), number_of_lines=("DESC_LINEA", len))
    .round(0)
    .sort_values("COD_UBIC_P", ascending=True)
    .reset_index()
    .astype({"COD_UBIC_P": int})
)

print("Paradas compartidas entre las líneas de ómnibus seleccionadas:\n")
(
    shared_bus_stops.loc[shared_bus_stops["number_of_lines"] > 1, :]["lines"]
    .drop_duplicates()
    .reset_index(drop=True)
)
# %%
# Cargar grafo
G = nx.read_gpickle(FILE_GRAPH)
# %%
network_bokeh_plot(
    G,
    title="Grafo de paradas de ómnibus",
    colorby="in_degree",
    add_labels=False,
    save=False,
)
# %%
