import networkx as nx
from itertools import product

# Clase grafo hereda de networkx para aprovechar sus metodos
# 
class Graph(nx.Graph):
    def __init__(self):
        super().__init__()
        self.removed_edges = [] 
        self.table_probability = None
        self.loss_value = -1
        self.evaluated = False


    def create_graph(self, current_nodes: str, future_nodes: str):
        self.create_nodes(current_nodes, future_nodes)
        new_edges = self.create_edges(current_nodes, future_nodes)
        self.add_edges_from(new_edges)

        # self.create_nodes(current_nodes, future_nodes)

        return self

    def create_edges(self, current_nodes, future_nodes):
        current_nodes = list(current_nodes)
        future_nodes = [f_node + "'" for f_node in future_nodes]

        edges = list(product(current_nodes, future_nodes))

        return edges

    def create_nodes(self, current_nodes: str, future_nodes: str):
        futures = [f_node + "'" for f_node in future_nodes]
        currents = list(current_nodes)

        self.add_nodes_from(currents)
        self.add_nodes_from(futures)

    
    def add_edges_to_graph(self, edges: list):
        self.add_edges_from(edges)

    # Apartir de las aristas actuales, optiene una expresion de probabilidad
    # que puede comprender los metodos de marginalizacion
    def conver_edges_to_probability_expression(self):
        exp_edges_prob = {}

        for edges in self.edges():
            node1, node2 = edges

            type_nodes= self._get_type_nodes(node1, node2)
            future = type_nodes['future']
            current = type_nodes['current']

            if future not in exp_edges_prob:
                exp_edges_prob[future] = ''

            exp_edges_prob[future] = exp_edges_prob[future] + current

        return exp_edges_prob
    
    def _get_type_nodes(self, node1, node2):
        if "'" in node1:
            return {
                'future': node1,
                'current': node2
            }
        
        return {
            'future': node2,
            'current': node1
        }

