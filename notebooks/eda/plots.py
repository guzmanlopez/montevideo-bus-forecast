from typing import List, Tuple

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.io import output_notebook, show
from bokeh.models import Circle, ColumnDataSource, LabelSet, MultiLine, StaticLayoutProvider, Title
from bokeh.palettes import Viridis8
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap


def plot_boardings_by_time(
    df: pd.DataFrame,
    x: str = "fecha_evento",
    y: str = "cantidad_pasajeros",
):
    plot = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X(f"{x}:T", title="Fecha"),
            alt.Y(f"{y}:Q", title="Cantidad de pasajeros"),
            tooltip=[x] + [y],
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
            tooltip=[x] + [y],
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
            tooltip=[x] + [y] + ["nombre_dia"],
        )
        .properties(width=700, height=400)
    )
    return plot


def network_bokeh_plot(
    G: nx.Graph = None,
    hover_tooltips: List[Tuple] = [("Parada", "@bus_stop"), ("Líneas", "@in_degree")],
    title: str = "",
    width: int = 800,
    height: int = 800,
    colorby: str = "",
    palette: Tuple = Viridis8,
    node_size: int = 12,
    nodes_alpha: float = 0.55,
    edges_alpha: float = 0.5,
    add_labels: bool = True,
    save: bool = False,
):
    output_notebook()

    # Create a graph renderer object
    graph_renderer = from_networkx(G, nx.spring_layout)

    # Set geographic positions to nodes
    fixed_layout = dict()

    for node in G.nodes:
        fixed_layout[node] = [G.nodes[node]["x"], G.nodes[node]["y"]]
    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=fixed_layout)

    # Create a plot
    plot = figure(
        tooltips=hover_tooltips,
        tools="pan,wheel_zoom,save,reset",
        active_scroll="wheel_zoom",
        plot_width=width,
        plot_height=height,
    )
    plot.add_layout(Title(text=title), "above")

    # Set node sizes and colors according to node indegree
    minval_color = min(graph_renderer.node_renderer.data_source.data[colorby])
    maxval_color = max(graph_renderer.node_renderer.data_source.data[colorby])

    graph_renderer.node_renderer.glyph = Circle(
        size=node_size,
        fill_alpha=nodes_alpha,
        fill_color=linear_cmap(colorby, palette, minval_color, maxval_color),
    )

    # Set edge width and alpha
    graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=edges_alpha)
    graph_renderer.edge_renderer.data_source.data["line_width"] = [
        np.log10(G[u][v]["weight"] + 1) for u, v in G.edges()
    ]
    graph_renderer.edge_renderer.glyph.line_width = {"field": "line_width"}

    plot.renderers.append(graph_renderer)

    if add_labels:
        x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
        node_labels = list(G.nodes())
        source = ColumnDataSource(
            {"x": x, "y": y, "name": [node_labels[i] for i in range(len(x))]}
        )
        labels = LabelSet(
            x="x",
            y="y",
            text="name",
            source=source,
            background_fill_color="white",
            text_font_size="10px",
            background_fill_alpha=0.75,
        )
        plot.renderers.append(labels)

    show(plot)

    if save:
        save(plot, filename=f"{title}.html")
