from graph.graph import Graph
import emd.emd_calculation as emd
import graph.remove_edges as remove_edges
import networkx as nx
import probability.utils as utils


# Busca una particion de mejor perdida, eliminando grupos de aristas hasta encontrar
# la que su valor de perdida sea menor.
# @ param network: grafo con las aristas 0 eliminadas
# @ return: grafo con la mejor particion
# Mientras hay grafos para evaluar, el proceso continua
# Estos se ordenan por perdida y aristas eliminadas
# Una vez evaluados, revisa su hay solucion y almacena en best_solutions
# Una vez evaluados todos los grafos
# @ return: grafo de mejor solucion
def find_best_partition(network: Graph, proccess_data, original_prob):
    network.loss_value = 0
    val_cup = calcule_cut_value(network, len(proccess_data['current']))
    proccess_data['channels'] = proccess_data['current']
    proccess_data['val_cup'] = val_cup
    best_solutions = []
    graphs_evaluated = [network]
    graph_solition = None
    best_value = float('inf')
    finish = False
    graph = None

    while len(graphs_evaluated) > 0 and not finish:
        # Ordena los grafos de menor a mayor perdida y con mayor numero de aristas eliminadas 
        graphs_sort = sorted(graphs_evaluated, key=lambda graph: (
            graph.loss_value, len(graph.removed_edges)))
        graph = graphs_sort[0]
        edges_graph = graph.edges(data=True)
        sort_edges = sorted(edges_graph, key=lambda x: x[2]['weight'])
        new_graphs_deletes_edge = create_graphs_delete_edge(
            graph, best_solutions, sort_edges, proccess_data, original_prob)
        
        graphs_evaluated.remove(graph)

        graphs_evaluated.extend(new_graphs_deletes_edge)

        if len(best_solutions) > 0:
            for graph in best_solutions:
                emd_value = graph.loss_value
                if emd_value < best_value:
                    best_value = emd_value
                    graph_solition = graph

    return graph_solition


# Para cada grafo a evaluar, crea un nuevo grafo con la arista eliminada que este dentro de la cota
# Calcula su nueva probabilidad y valor de perdida, si el grafo deja de ser conexto
# es un grafo solucion y lo almacena en best_solutions, ademas cambia la cota de corte
# de ser conexo agrega el grafo a grafos por evaluar si su perdida es menor a la cota actual
# @return: Lista de grafos por evaluar
def create_graphs_delete_edge(father_network: Graph, best_solutions: list, edges, proccess_data, original_prob):
    new_graphs = []
    emd_graph = father_network.loss_value

    for edge in edges:
        edge_evaluated = (edge[0], edge[1], edge[2]['weight'])
        posible_emd, _ = calcule_posible_emd(
            father_network.removed_edges, edge_evaluated, emd_graph)
        if posible_emd <= proccess_data['val_cup']:
            new_graph = remove_edges.create_new_graph(father_network, edge)
            new_graph.removed_edges.extend(father_network.removed_edges)

            new_tables_prob = remove_edges.calcule_table_probabilities(
                father_network, proccess_data, edge)
            found_emd = emd.calcule_emd(
                new_tables_prob, proccess_data['state'], original_prob)
            new_graph.loss_value = found_emd
            new_graph.table_probability = new_tables_prob
            # print(f'Edge: {edge_evaluated} - Value: {posible_emd}')
            # print(f'Evalate Node: {posible_emd} <= cut value: {proccess_data['val_cup']}')
            # print(f'found EMD: {found_emd}')
            # print(f'Edges: {new_graph.edges(data=True)}')
            # print(f'EdgesRemoved: {new_graph.removed_edges}')
            # print(f'Is connected: {nx.is_connected(new_graph)}')

            if not nx.is_connected(new_graph):
                if found_emd < proccess_data['val_cup']:
                    best_solutions.append(new_graph)
                    proccess_data['val_cup'] = found_emd
                    #print(f'\n New best value: {found_emd}\n')
            else:
                #print(f'Condicional New graph:{found_emd} < {proccess_data['val_cup']}\n')
                if found_emd < proccess_data['val_cup']:
                    new_graphs.append(new_graph)
            
            #print('------------------------\n')

    return new_graphs


# Agrupamos los nodos eliminados y agregamos el nuevo nodo a eliminar a un mismo nodo destino, 
# el mayor peros de estos es el rango de inicio La suma de estos es el rango superior. 
# Sumamos los grupos de nodos eliminados de esa manera de obtener un posible rango de perdida.
def calcule_posible_emd(removed_edges, new_edge, emd_graph):
    group_nodes = {}
    no_zero_edges = [edge for edge in removed_edges if edge[2] != 0]
    no_zero_edges.append(new_edge)

    if len(no_zero_edges) == 0:
        return emd_graph + new_edge[2]

    for edge in no_zero_edges:
        node1, node2, weight = edge
        destino, origen = utils.get_type_nodes(node1, node2)

        if destino not in group_nodes:
            group_nodes[destino] = {}

        group_nodes[destino]["base"] = weight if weight > group_nodes[destino].get(
            "base", 0) else group_nodes[destino].get("base", 0)
        group_nodes[destino]["sum"] = group_nodes[destino].get(
            "sum", 0) + weight

    base_values = round(sum(value['base']
                        for value in group_nodes.values()), 3)
    sum_values = sum(value['sum'] for value in group_nodes.values())

    return base_values, sum_values


# La cota inicial se calcula con el valor de perdida promedio de todas las aristas
# por la maxima cantidad de aristas que puede cortar antes de encontrar una particion,
# este valor equivale al numero de canales actuales. Todas las ariasta encima de este valor
# no se expanderan para encontrar una particion.
def calcule_cut_value(network: Graph, num_channels_current):
    sum_val_emd = 0

    for edge in network.edges(data=True):
        _, _, details_edge = edge
        weight = details_edge['weight']

        sum_val_emd += weight

    value_cup = (sum_val_emd / len(network.edges())) * num_channels_current

    return round(value_cup, 3)
