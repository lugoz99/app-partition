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
        Visualizes the original graph and the resulting partitioned graph with proportional layouts.

        Args:
            original_graph (nx.Graph): The original graph.
            network_found (nx.Graph): The graph after finding the best partition.
        """
        # Adjust the layouts for proportional distribution
        pos_orig = nx.spring_layout(original_graph, seed=42, k=0.5, iterations=50)
        pos_found = nx.spring_layout(network_found, seed=42, k=0.5, iterations=50)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#f9f9f9")

        # Plot Original Graph
        nx.draw(
            original_graph,
            pos_orig,
            with_labels=True,
            node_color="skyblue",
            edge_color="gray",
            node_size=700,
            font_size=10,
            ax=ax1,
        )
        ax1.set_title("Original Network", fontsize=14, fontweight="bold")

        # Plot Network Found
        edge_labels = nx.get_edge_attributes(network_found, "weight")
        edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}

        nx.draw(
            network_found,
            pos_found,
            with_labels=True,
            node_color="lightgreen",
            edge_color="black",
            node_size=700,
            font_size=10,
            ax=ax2,
        )
        nx.draw_networkx_edge_labels(
            network_found, pos_found, edge_labels=edge_labels, font_size=8, ax=ax2
        )
        ax2.set_title("Network Best Partition", fontsize=14, fontweight="bold")

        # Add text for EMD
        emd_text = f"EMD Loss: {getattr(network_found, 'loss_value', 'N/A')}"
        ax2.text(
            0.5,
            -0.1,
            emd_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight="bold",
            transform=ax2.transAxes,
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def grapho(graph: nx.Graph):
        """
        Visualizes a graph with edge weights and proportional layout.

        Args:
            graph (nx.Graph): The graph to visualize.
        """
        pos = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
        edge_labels = nx.get_edge_attributes(graph, "weight")
        edge_labels = {k: f"{v:.4f}" for k, v in edge_labels.items()}

        plt.figure(figsize=(10, 8), facecolor="#f9f9f9")
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="lightcoral",
            edge_color="black",
            node_size=700,
            font_size=12,
        )
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
        plt.title("Graph Visualization", fontsize=16, fontweight="bold")
        plt.show()
