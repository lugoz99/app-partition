from networkx import Graph
import pandas as pd
import networkx as nx
from cut_strategy.find_partition import find_best_partition
from cut_strategy.marginalize import get_marginalize_channel
from cut_strategy.probability import get_original_probability
from cut_strategy.utils import get_type_nodes
from emd_calculation import calcule_emd as emd
from graph import Graph


def remove_edges(network: Graph, probabilities, proccess_data):
    original_prob = get_original_probability(
        probabilities,
        proccess_data["current"],
        proccess_data["future"],
        proccess_data["channels"],
    )
    network.table_probability = probabilities

    original_graph = network.copy()

    for removed_edge in original_graph.edges():
        new_tables_prob = calcule_table_probabilities(
            network, proccess_data, removed_edge
        )
        emd_value = emd.calcule_emd(
            new_tables_prob, proccess_data["state"], original_prob
        )
        network.loss_value = emd_value

        original_graph[removed_edge[0]][removed_edge[1]]["weight"] = emd_value
        network[removed_edge[0]][removed_edge[1]]["weight"] = emd_value

        if network.loss_value == 0:
            proccess_data["channels"] = proccess_data["current"]
            network.remove_edge(*removed_edge)
            info_edge_removed = (removed_edge[0], removed_edge[1], emd_value)
            network.removed_edges.append(info_edge_removed)
            network.table_probability = new_tables_prob

        if not nx.is_connected(network):
            return network

    graph_found = find_best_partition(network, proccess_data, original_prob)

    return graph_found


def create_new_graph(network: Graph, removed_edge):
    original_graph = network.copy()
    original_graph.remove_edge(removed_edge[0], removed_edge[1])

    graph_processor = Graph()
    graph_processor.add_edges_to_graph(original_graph.edges(data=True))
    info_edge_removed = (removed_edge[0], removed_edge[1], removed_edge[2]["weight"])
    graph_processor.removed_edges.append(info_edge_removed)

    return graph_processor


def calcule_table_probabilities(graph: Graph, proccess_data, node_delete):
    current_channels = proccess_data["current"]
    future_channels = proccess_data["future"]
    channels = proccess_data["channels"]
    tables_result = {}
    node_future, node_current = get_type_nodes(node_delete[0], node_delete[1])
    marg_express = get_marginalize_expression(
        current_channels, future_channels, node_delete
    )

    for future, current in marg_express.items():
        future_tab = future.replace("'", "")
        marginalize_table = get_marginalize_channel(
            graph.table_probability[future_tab], current, channels
        )

        if future == node_future:
            new_complete_table = complete_table_prob(
                marginalize_table, node_delete, current
            )
            tables_result[future_tab] = new_complete_table
            continue

        tables_result[future_tab] = marginalize_table

    return tables_result


def get_marginalize_expression(current_channels, future_channels, node_delete):
    future_node, current_node = get_type_nodes(node_delete[0], node_delete[1])
    expression_marginalize = {}

    for future in future_channels:
        future = future + "'"
        expression_marginalize[future] = current_channels

        if future == future_node:
            new_current = current_channels.replace(current_node, "")
            expression_marginalize[future] = new_current

    return expression_marginalize


# Calcula la tabla de probabilidad cuando la expresion de probabilidad no encuentra
# una expresion para el nodo eliminado, este caso equivale al futuro vacio | current
def get_table_future_empty(node_delete, probabilities, proccess_data):
    future, current = node_delete
    currents = proccess_data["current"].replace(current, "")
    exp_prob = {}

    chat_future = future.replace("'", "")
    marginalize_table = get_marginalize_channel(
        probabilities[chat_future], currents, proccess_data["channels"]
    )

    exp_prob[future] = currents

    return marginalize_table, exp_prob


# Rellena el valor marginalizado de la tabla, "Copiando" el valor de probabilidad
# marguinalizado a la tabla como si tuviera todos sus valores originales.
# @ param probabilities: diccionario con las tablas de probabilidad
# @ param node_delete: tupla con los nodos a eliminar
# @ param probability_exp: diccionario con las expresiones de probabilidad, del grafo sin los nodos eliminados
def complete_table_prob(probability_table, node_delete, channels_current):
    _, current = get_type_nodes(node_delete[0], node_delete[1])
    position_change = calcule_position_modify_index(channels_current, current)
    new_table = modify_table_probability(probability_table, position_change)

    return new_table


# Copia la tabla de probabilidad con el resultado de la marginalizacion
# @ param probability_table: tabla de probabilidad
# @ param position: posicion a modificar
# El posicion es el indice del caracter a modificar
def modify_table_probability(probability_table, position):
    copy_probability_table = probability_table.copy()

    probability_table.index = [
        modify_index(index, position, "0") for index in probability_table.index
    ]
    copy_probability_table.index = [
        modify_index(index, position, "1") for index in copy_probability_table.index
    ]

    new_probability_table = pd.concat([probability_table, copy_probability_table])

    return new_probability_table


def modify_index(index, position, value):
    index = index[:position] + value + index[position:]
    return index


# Dados todos los current channels, y el nodo eliminado, determina en que posicion de la cadena
# debe el el caracter a agregar en la indice de la tabla a modificar
def calcule_position_modify_index(chanels, node):
    str_chanels = chanels + node
    sorted_channels = "".join(sorted(str_chanels))
    index_node = sorted_channels.index(node)

    return index_node
