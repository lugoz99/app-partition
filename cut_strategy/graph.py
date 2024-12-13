import networkx as nx
from itertools import product


class GraphManager(nx.Graph):
    """
    Encapsulates functionality for managing and manipulating graphs.
    """

    def __init__(self):
        super().__init__()
        self.removed_edges = []
        self.table_probability = None
        self.loss_value = -1
        self.evaluated = False

    def create_graph(self, current_nodes: str, future_nodes: str):
        """
        Creates a graph with nodes and edges based on current and future nodes.

        Args:
            current_nodes (str): String representing current nodes.
            future_nodes (str): String representing future nodes.

        Returns:
            GraphManager: The created graph.
        """
        self.create_nodes(current_nodes, future_nodes)
        new_edges = self.create_edges(current_nodes, future_nodes)
        self.add_edges_from(new_edges)

        return self

    def create_edges(self, current_nodes, future_nodes):
        """
        Creates edges between current and future nodes.

        Args:
            current_nodes (str): Current nodes.
            future_nodes (str): Future nodes.

        Returns:
            list: List of edges.
        """
        current_nodes = list(current_nodes)
        future_nodes = [f_node + "'" for f_node in future_nodes]

        return list(product(current_nodes, future_nodes))

    def create_nodes(self, current_nodes: str, future_nodes: str):
        """
        Adds nodes to the graph based on current and future nodes.

        Args:
            current_nodes (str): Current nodes.
            future_nodes (str): Future nodes.
        """
        futures = [f_node + "'" for f_node in future_nodes]
        currents = list(current_nodes)

        self.add_nodes_from(currents)
        self.add_nodes_from(futures)

    def add_edges_to_graph(self, edges: list):
        """
        Adds edges to the graph.

        Args:
            edges (list): List of edges to add.
        """
        self.add_edges_from(edges)

    def convert_edges_to_probability_expression(self):
        """
        Converts edges to a probability expression for marginalization.

        Returns:
            dict: Probability expression based on edges.
        """
        exp_edges_prob = {}

        for edge in self.edges():
            node1, node2 = edge

            type_nodes = self._get_type_nodes(node1, node2)
            future = type_nodes["future"]
            current = type_nodes["current"]

            if future not in exp_edges_prob:
                exp_edges_prob[future] = ""

            exp_edges_prob[future] += current

        return exp_edges_prob

    def _get_type_nodes(self, node1, node2):
        """
        Determines the type of nodes (current or future).

        Args:
            node1 (str): First node.
            node2 (str): Second node.

        Returns:
            dict: Dictionary with 'future' and 'current' node types.
        """
        if "'" in node1:
            return {"future": node1, "current": node2}

        return {"future": node2, "current": node1}
