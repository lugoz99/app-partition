import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx


def create_sub_table(data_frame, colum_extract):
    if colum_extract == "":
        return data_frame

    new_table = data_frame[[colum_extract]].copy()
    new_colum = colum_extract + "0"
    new_table.insert(0, new_colum, 1 - data_frame[colum_extract])

    return new_table


def create_probability_distributions(json_file):
    probability_distributions = {}
    with open(json_file, "r") as archivo:
        datos = json.load(archivo)

    for channel, dist_prob in datos.items():
        probability_distributions[channel] = pd.DataFrame(dist_prob).T

    print("--------- Probability Distributions ---------")
    for channel, dist_prob in probability_distributions.items():
        print(f"{channel}: \n{dist_prob}\n")
    return probability_distributions


def get_type_nodes(node1, node2):
    if "'" in node1:
        return node1, node2

    return node2, node1


def graph_result(original_graph, neteork_found):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    nx.draw(original_graph, with_labels=True, ax=ax1)
    ax1.set_title("Original Network")

    pos = nx.spring_layout(neteork_found)
    edge_labels = nx.get_edge_attributes(G=neteork_found, name="weight")
    edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}
    nx.draw(neteork_found, pos, with_labels=True, ax=ax2)
    nx.draw_networkx_edge_labels(
        neteork_found, pos, edge_labels=edge_labels, font_size=8, ax=ax2
    )
    ax2.set_title("Network Best Partition")
    text = "EMD: " + str(neteork_found.loss_value)
    ax2.text(0, 0, text, verticalalignment="center", transform=ax2.transAxes)

    plt.show()


def grapho(graph):
    pos = nx.spring_layout(graph)
    edge_labels = nx.get_edge_attributes(G=graph, name="weight")
    edge_labels = {k: f"{v:.4f}" for k, v in edge_labels.items()}
    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.show()
