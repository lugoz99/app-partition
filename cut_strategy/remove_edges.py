from networkx import Graph
import pandas as pd
import networkx as nx
from cut_strategy.find_partition import PartitionFinder as find_best_partition
from cut_strategy.marginalize import Marginalizer as get_marginalize_channel
from cut_strategy.probability import ProbabilityCalculator as get_original_probability
from cut_strategy.utils import Utils as utils
from cut_strategy.emd_calculation import EMDEvaluator as emd
from cut_strategy.graph import GraphManager as Graph


class EdgeRemover:
    """
    Encapsulates functionality for removing edges and evaluating graphs.
    """

    @staticmethod
    def remove_edges(network: Graph, probabilities: dict, process_data: dict) -> Graph:
        """
        Removes edges from a graph based on EMD value and connectivity.

        Args:
            network (Graph): The input graph.
            probabilities (dict): Probability tables for the graph.
            process_data (dict): Process data including current, future channels, and state.

        Returns:
            Graph: The modified graph after edge removal and evaluation.
        """
        original_prob = get_original_probability.get_original_probability(
            probabilities,
            process_data["current"],
            process_data["future"],
            process_data["channels"],
        )
        network.table_probability = probabilities
        original_graph = network.copy()

        for removed_edge in original_graph.edges():
            new_tables_prob = EdgeRemover.calcule_table_probabilities(
                network, process_data, removed_edge
            )
            emd_value = emd.calculate_emd(
                new_tables_prob, process_data["state"], original_prob
            )
            network.loss_value = emd_value

            original_graph[removed_edge[0]][removed_edge[1]]["weight"] = emd_value
            network[removed_edge[0]][removed_edge[1]]["weight"] = emd_value

            if network.loss_value == 0:
                process_data["channels"] = process_data["current"]
                network.remove_edge(*removed_edge)
                info_edge_removed = (removed_edge[0], removed_edge[1], emd_value)
                network.removed_edges.append(info_edge_removed)
                network.table_probability = new_tables_prob

            if not nx.is_connected(network):
                return network

        graph_found = find_best_partition(network, process_data, original_prob)

        return graph_found

    @staticmethod
    def create_new_graph(network: Graph, removed_edge: tuple) -> Graph:
        """
        Creates a new graph by removing a specified edge.

        Args:
            network (Graph): The input graph.
            removed_edge (tuple): The edge to remove.

        Returns:
            Graph: A new graph with the specified edge removed.
        """
        original_graph = network.copy()
        original_graph.remove_edge(removed_edge[0], removed_edge[1])

        graph_processor = Graph()
        graph_processor.add_edges_to_graph(original_graph.edges(data=True))
        info_edge_removed = (
            removed_edge[0],
            removed_edge[1],
            removed_edge[2]["weight"],
        )
        graph_processor.removed_edges.append(info_edge_removed)

        return graph_processor

    @staticmethod
    def calcule_table_probabilities(
        graph: Graph, process_data: dict, node_delete: tuple
    ) -> dict:
        """
        Calculates probability tables after removing an edge.

        Args:
            graph (Graph): The input graph.
            process_data (dict): Process data including channels and states.
            node_delete (tuple): The edge to evaluate for removal.

        Returns:
            dict: Updated probability tables.
        """
        current_channels = process_data["current"]
        future_channels = process_data["future"]
        channels = process_data["channels"]
        tables_result = {}
        node_future, node_current = utils.get_type_nodes(node_delete[0], node_delete[1])
        marg_express = EdgeRemover.get_marginalize_expression(
            current_channels, future_channels, node_delete
        )

        for future, current in marg_express.items():
            future_tab = future.replace("'", "")
            marginalize_table = get_marginalize_channel.get_marginalize_channel(
                graph.table_probability[future_tab], current, channels
            )

            if future == node_future:
                new_complete_table = EdgeRemover.complete_table_prob(
                    marginalize_table, node_delete, current
                )
                tables_result[future_tab] = new_complete_table
                continue

            tables_result[future_tab] = marginalize_table

        return tables_result

    @staticmethod
    def get_marginalize_expression(
        current_channels: str, future_channels: str, node_delete: tuple
    ) -> dict:
        """
        Generates marginalization expressions for probability tables.

        Args:
            current_channels (str): Current channel string.
            future_channels (str): Future channel string.
            node_delete (tuple): Nodes to remove.

        Returns:
            dict: Expressions for marginalizing probability tables.
        """
        future_node, current_node = utils.get_type_nodes(node_delete[0], node_delete[1])
        expression_marginalize = {}

        for future in future_channels:
            future = future + "'"
            expression_marginalize[future] = current_channels

            if future == future_node:
                new_current = current_channels.replace(current_node, "")
                expression_marginalize[future] = new_current

        return expression_marginalize

    @staticmethod
    def complete_table_prob(
        probability_table: pd.DataFrame, node_delete: tuple, channels_current: str
    ) -> pd.DataFrame:
        """
        Completes a probability table by adding missing values for marginalization.

        Args:
            probability_table (pd.DataFrame): Probability table to complete.
            node_delete (tuple): Nodes to remove.
            channels_current (str): Current channel string.

        Returns:
            pd.DataFrame: Completed probability table.
        """
        _, current = utils.get_type_nodes(node_delete[0], node_delete[1])
        position_change = EdgeRemover.calcule_position_modify_index(
            channels_current, current
        )
        new_table = EdgeRemover.modify_table_probability(
            probability_table, position_change
        )

        return new_table

    @staticmethod
    def modify_table_probability(
        probability_table: pd.DataFrame, position: int
    ) -> pd.DataFrame:
        """
        Modifies a probability table by updating its indices.

        Args:
            probability_table (pd.DataFrame): Table to modify.
            position (int): Index position to update.

        Returns:
            pd.DataFrame: Updated probability table.
        """
        copy_probability_table = probability_table.copy()

        probability_table.index = [
            EdgeRemover.modify_index(index, position, "0")
            for index in probability_table.index
        ]
        copy_probability_table.index = [
            EdgeRemover.modify_index(index, position, "1")
            for index in copy_probability_table.index
        ]

        new_probability_table = pd.concat([probability_table, copy_probability_table])

        return new_probability_table

    @staticmethod
    def modify_index(index: str, position: int, value: str) -> str:
        """
        Modifies an index string by inserting a value at a specific position.

        Args:
            index (str): The index to modify.
            position (int): The position to insert the value.
            value (str): The value to insert.

        Returns:
            str: Modified index string.
        """
        index = index[:position] + value + index[position:]
        return index

    @staticmethod
    def calcule_position_modify_index(channels: str, node: str) -> int:
        """
        Calculates the position to modify in a probability table index.

        Args:
            channels (str): Current channels.
            node (str): Node to evaluate.

        Returns:
            int: Position index for modification.
        """
        str_channels = channels + node
        sorted_channels = "".join(sorted(str_channels))
        index_node = sorted_channels.index(node)

        return index_node
