# Download data sources
URL_STM_VIAJES_OCTUBRE = "https://imnube.montevideo.gub.uy/share/s/RqMZ1vycRBOiOfrjcylMkQ"
URL_STM_VIAJES_NOVIEMBRE = "https://imnube.montevideo.gub.uy/share/s/rvjAZX4AQAaymDiye9SXPQ"
URL_STM_VIAJES_DICIEMBRE = "https://imnube.montevideo.gub.uy/share/s/AZo_3bX6RHy2CmJTa3wzXw"
URL_HORARIOS_OMNIBUS_POR_PARADAS = (
    "http://www.montevideo.gub.uy/sites/default/files/datos/uptu_pasada_variante.zip"
)
URL_STM_PARADAS_OMNIBUS = "https://intgis.montevideo.gub.uy/sit/tmp/v_uptu_paradas.zip"
URL_STM_RECORRIDOS_OMNIBUS = (
    "https://intgis.montevideo.gub.uy/sit/tmp/uptu_variante_no_maximal.zip"
)

# Files
FILE_STM_VIAJES_PREFIX = "stm_viajes_"
FILE_STM_HORARIOS_BUSES_PARADAS = "stm_horarios_buses_paradas"
FILE_STM_PARADAS = "stm_paradas"
FILE_STM_RECORRIDOS = "stm_recorridos"
RAW_DATA_PATH = "./data/raw/"
PROCESSED_DATA_PATH = "./data/processed/"
PROCESSED_FILE = "df_stm_bus_proc"
SNAP_FILE = "snap_bus_stops_to_bus_track"
POINTS_IN_TRACK_FILE = "points_in_track"

# Parameters
MONTH = "octubre"

# Process data
DF_STM_VIAJES_COLS = [
    "fecha_evento",
    "dsc_linea",
    "sevar_codigo",
    "cantidad_pasajeros",
    "codigo_parada_origen",
    "ordinal_de_tramo",
]

DAY_NAME_MAPPING = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo",
}
