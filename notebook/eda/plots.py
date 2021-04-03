import altair as alt
import pandas as pd


def plot_boardings_by_time(
    df: pd.DataFrame, x: str = "fecha_evento", y: str = "cantidad_pasajeros"
):
    plot = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X(f"{x}:T", title="Fecha"),
            alt.Y(f"{y}:Q", title="Cantidad de pasajeros"),
        )
        .properties(width=700, height=400)
    )
    return plot


def plot_boardings_by_day_name(
    df: pd.DataFrame, x: str = "nombre_dia", y: str = "cantidad_pasajeros"
):

    plot = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            alt.X(
                f"{x}:O",
                title="Día de la semana",
                sort=[
                    "Lunes",
                    "Martes",
                    "Miércoles",
                    "Jueves",
                    "Viernes",
                    "Sábado",
                    "Domingo",
                ],
            ),
            alt.Y(f"{y}:Q", title="Cantidad de pasajeros"),
        )
        .properties(width=700, height=400)
    )
    return plot


def plot_boardings_by_hour_and_day_name(
    df: pd.DataFrame, x: str = "hora", y: str = "cantidad_pasajeros"
):
    plot = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            alt.X(
                f"{x}:O",
                title="Hora del día",
            ),
            alt.Y(f"{y}:Q", title="Cantidad de pasajeros"),
            alt.Color(
                "nombre_dia:N",
                title="Día de la semana",
                sort=[
                    "Lunes",
                    "Martes",
                    "Miércoles",
                    "Jueves",
                    "Viernes",
                    "Sábado",
                    "Domingo",
                ],
            ),
        )
        .properties(width=700, height=400)
    )
    return plot
