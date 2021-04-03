# %% [markdown]
# # Análisis Exploratorio de Datos

# %%
import pandas as pd
from src.preparation.constants import DAY_NAME_MAPPING, PROCESSED_FILE
from src.preparation.utils import (
    load_pickle_file,
    load_stm_bus_data,
    load_stm_bus_line_track,
    load_stm_bus_stops,
    save_pickle_file,
)
from src.processing.process_stm_bus_data import pre_process_data
from src.processing.utils import cut_tracks_by_bus_stops, get_longest_track_from_bus_line

from plots import (
    plot_boardings_by_day_name,
    plot_boardings_by_hour_and_day_name,
    plot_boardings_by_time,
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

# %%
# Create adyacency matrix
df_proc = load_pickle_file(PROCESSED_FILE)
df_proc = df_proc.loc[df_proc["dsc_linea"] == "103", :]
sevar_codigo = df_proc["sevar_codigo"].unique().tolist()

# %%
# Load bus tracks and filter by sevar_codigo
gdf_bus_tracks = load_stm_bus_line_track()
gdf_bus_tracks = gdf_bus_tracks.loc[
    (gdf_bus_tracks["COD_VAR_01"].isin(sevar_codigo)) & (gdf_bus_tracks["DESC_VARIA"] == "A"), :
]
cod_varian = gdf_bus_tracks["COD_VAR_01"].unique()
gdf_bus_tracks = get_longest_track_from_bus_line(gdf_bus_tracks)

# %%
# Load bus stops and filter by sevar_codigo
gdf_bus_stops = load_stm_bus_stops()
gdf_bus_stops = gdf_bus_stops.loc[gdf_bus_stops["COD_VARIAN"].isin(cod_varian), :]
gdf_bus_stops = (
    gdf_bus_stops[["DESC_LINEA", "COD_UBIC_P", "geometry"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# %%
gdf_bus_tracks_by_stops = cut_tracks_by_bus_stops(gdf_bus_stops, gdf_bus_tracks)

# %%
