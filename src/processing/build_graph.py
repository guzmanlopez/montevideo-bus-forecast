import json
from pathlib import Path

import networkx as nx
import pretty_errors  # noqa
import typer
from networkx.readwrite import json_graph
from src.preparation.constants import FILE_GRAPH, PROCESSED_DATA_PATH
from src.preparation.typer_messages import msg_process
from src.processing.utils import get_networkx_graph


def main():
    msg_process("Building Networkx Directed graph\n")
    G = get_networkx_graph()
    nx.write_gpickle(G, Path(PROCESSED_DATA_PATH) / f"{FILE_GRAPH}.pkl")
    with open(Path(PROCESSED_DATA_PATH) / f"{FILE_GRAPH}.json", "w") as f:
        f.write(json.dumps(json_graph.node_link_data(G)))
    return G


if __name__ == "__main__":
    typer.run(main)
