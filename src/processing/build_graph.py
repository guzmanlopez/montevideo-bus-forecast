import numpy as np
import pretty_errors  # noqa
import typer
from src.preparation.typer_messages import msg_process
from src.processing.utils import get_networkx_graph


def main():
    msg_process("Building Networkx Directed graph\n")
    G = get_networkx_graph()
    return G


if __name__ == "__main__":
    typer.run(main)
