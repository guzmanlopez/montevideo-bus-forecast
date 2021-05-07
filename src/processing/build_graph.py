import json
from pathlib import Path
from typing import List

import networkx as nx
import pretty_errors  # noqa
import typer
from networkx.readwrite import json_graph
from src.preparation.constants import (
    FILE_GRAPH,
    FILE_GRAPH_FEATURES,
    PROCESSED_DATA_PATH,
)
from src.processing.utils import (
    add_features_to_graph,
    add_target_to_graph,
    get_networkx_graph,
)


def main(target: str = typer.Option("y"), features: List[str] = typer.Option(["time_index"])):
    G = get_networkx_graph()
    nx.write_gpickle(G, Path(PROCESSED_DATA_PATH) / f"{FILE_GRAPH}.pkl")
    with open(Path(PROCESSED_DATA_PATH) / f"{FILE_GRAPH}.json", "w") as f:
        f.write(json.dumps(json_graph.node_link_data(G)))
    G = add_target_to_graph(G, target)
    G = add_features_to_graph(G, features)
    nx.write_gpickle(G, Path(PROCESSED_DATA_PATH) / f"{FILE_GRAPH_FEATURES}.pkl")
    with open(Path(PROCESSED_DATA_PATH) / f"{FILE_GRAPH_FEATURES}.json", "w") as f:
        f.write(json.dumps(json_graph.node_link_data(G)))
    return G


if __name__ == "__main__":
    typer.run(main)
