import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import matplotlib.pyplot as plt
import networkx as nx
from cut_strategy.utils import Utils as utils
from cut_strategy.remove_edges import EdgeRemover as remove_edges
import time as t
from cut_strategy.graph import GraphManager


def main_delete_edge(process_data):
    new_graph = GraphManager()
    network_graph = new_graph.create_graph(
        process_data["current"], process_data["future"]
    )
    original_graph = network_graph.copy()

    init = t.time()
    probability_distributions = utils.create_probability_distributions(
        process_data["file"]
    )
    network_found = remove_edges.remove_edges(
        network_graph, probability_distributions, process_data
    )

    finish = t.time()

    print("--------- Results ---------")
    print(f"Loss value: {network_found.loss_value}")
    print(f"Removed edges: {network_found.removed_edges}")
    print(f"Edges Result: {network_found.edges(data=True)}")
    print(f"Components: {list(nx.connected_components(network_found))}")
    print(f"Probability distributions: \n {network_found.table_probability}")
    print(f"Time: {round(finish - init, 5)} \n\n")

    utils.graph_result(original_graph, network_found)


def execute(data_to_process):
    if data_to_process["method"] == "delete_edges":
        main_delete_edge(data_to_process)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(base_dir, "../cut_strategy/data/tablex6.json")

    data_to_process = {
        "file": json_file_path,
        "future": "ABCD",
        "current": "ABCD",
        "state": "1001",
        "channels": "ABCDEF",  # 100010
        "method": "delete_edges",
    }
    execute(data_to_process)
