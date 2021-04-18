import networkx as nx
import pandas as pd
import pretty_errors  # noqa
import typer
from src.preparation.constants import FILE_GRAPH
from src.preparation.typer_messages import msg_info, msg_process
from src.preparation.utils import load_adyacency_data, save_pickle_file


def main():
    msg_process("Building Networkx Directed graph\n")
    df_from_to_weight = load_adyacency_data(adyacency=False)
    G = nx.from_pandas_edgelist(
        df_from_to_weight, source="from", target="to", edge_attr="weight", create_using=nx.DiGraph
    )
    G.name = "Bus lines of Montevideo"
    msg_info(f"\n{nx.info(G)}\n")
    save_pickle_file(df_from_to_weight, FILE_GRAPH)


if __name__ == "__main__":
    typer.run(main)
