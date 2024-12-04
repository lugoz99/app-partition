import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx


class Utils:
    """
    Encapsulates utility functions for data manipulation and graph visualization.
    """

    @staticmethod
    def create_sub_table(data_frame: pd.DataFrame, column_extract: str) -> pd.DataFrame:
        """
        Creates a sub-table based on the specified column by adding a complementary column.

        Args:
            data_frame (pd.DataFrame): The input DataFrame.
            column_extract (str): The column to extract.

        Returns:
            pd.DataFrame: The resulting DataFrame with a complementary column.
        """
        if column_extract == "":
            return data_frame

        new_table = data_frame[[column_extract]].copy()
        new_column = column_extract + "0"
        new_table.insert(0, new_column, 1 - data_frame[column_extract])

        return new_table

    @staticmethod
    def create_probability_distributions(json_file: str) -> dict:
        """
        Loads a JSON file and converts it into a dictionary of probability distributions.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            dict: A dictionary containing probability distributions as DataFrames.
        """
        probability_distributions = {}
        with open(json_file, "r") as file:
            data = json.load(file)

        for channel, dist_prob in data.items():
            probability_distributions[channel] = pd.DataFrame(dist_prob).T

        print("--------- Probability Distributions ---------")
        for channel, dist_prob in probability_distributions.items():
            print(f"{channel}: \n{dist_prob}\n")

        return probability_distributions

    @staticmethod
    def get_type_nodes(node1: str, node2: str) -> tuple:
        """
        Determines the type of two nodes (future or current).

        Args:
            node1 (str): The first node.
            node2 (str): The second node.

        Returns:
            tuple: A tuple containing the future node and the current node.
        """
        if "'" in node1:
            return node1, node2

        return node2, node1

    @staticmethod
    def graph_result(original_graph: nx.Graph, network_found: nx.Graph):
        """
        Visualizes the original graph and the resulting partitioned graph.

        Args:
            original_graph (nx.Graph): The original graph.
            network_found (nx.Graph): The graph after finding the best partition.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        nx.draw(original_graph, with_labels=True, ax=ax1)
        ax1.set_title("Original Network")

        pos = nx.spring_layout(network_found)
        edge_labels = nx.get_edge_attributes(G=network_found, name="weight")
        edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}
        nx.draw(network_found, pos, with_labels=True, ax=ax2)
        nx.draw_networkx_edge_labels(
            network_found, pos, edge_labels=edge_labels, font_size=8, ax=ax2
        )
        ax2.set_title("Network Best Partition")
        text = "EMD: " + str(network_found.loss_value)
        ax2.text(0, 0, text, verticalalignment="center", transform=ax2.transAxes)

        plt.show()

    @staticmethod
    def grapho(graph: nx.Graph):
        """
        Visualizes a graph with edge weights.

        Args:
            graph (nx.Graph): The graph to visualize.
        """
        pos = nx.spring_layout(graph)
        edge_labels = nx.get_edge_attributes(G=graph, name="weight")
        edge_labels = {k: f"{v:.4f}" for k, v in edge_labels.items()}
        nx.draw(graph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
        plt.show()
