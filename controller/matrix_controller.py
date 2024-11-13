from helpers.auxiliares import repr_current_to_array, repr_next_to_array
from helpers.condicionar import *
from helpers.generate_states import generate_states

"""
 * Elementos
 # 1. Matriz de representacion o TPM: Representa las probabilidades
      de pasa de un estado actual a un estado futuro o proximo.

#  2. El sistema completo comprende todos los elementos en el presente y futuro.

# 3. Sistema candidato: Es un subconjunto de elementos del sistema completo.
    3.1 Los elementos fuera del sistema candidato se consideran como condiciones de background.
    3.2 De igual forma, se puede trabajar sobre un subconjunto del sistema candidato,
     para lo cual se realizan procesos de marginalización

4. La TPM del sistema candidato se deriva de la TPM del sistema completo.
    b) Se obtiene condicionando la TPM completa sobre los estados en t de los elementos de background.
    c) Luego se marginaliza sobre los elementos de background en t+1.

# Objetivos
    * 1) Se hará el cálculo de la  distribución de probabilidades del sistema a tratar en el estado inicial dado.
            Lo que producto lo que llamamos : Distribución Original,
            Luego se divide el sistema original en dos partes.  Se trabajará sobre una de las partes en que se dividió el sistema

    * Para un conjunto de elementos en t,
     calcular la distribución de probabilidad sobre los estados de los elementos en t+1
"""


import numpy as np
from typing import Dict, Tuple, List, Any


def generate_tpm(
    extended_matrix: np.ndarray,
    states: np.ndarray,
    variables: List[str],
    initial_state: dict,
    candidate: List[str],
    fila_excluir: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera la matriz de transición (TPM) procesando las matrices de estados y extendida.
    """
    # Filtramos las columnas y obtenemos los estados finales
    columns, values = find_columns_to_filter(variables, initial_state, candidate)
    filtered_states, deleted_positions = filter_states(states, columns, values)
    final_extended_matrix = eliminate_rows(extended_matrix, deleted_positions)

    # Identificar pares de columnas repetidas y sumar las columnas correspondientes
    resultado_suma = identificar_y_sumar_pares(states.T, final_extended_matrix, fila_excluir)

    return resultado_suma



def main():
    extended_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    states = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )

    matriz_transpuesta = states.T
    variables = ["A", "B", "C", "D"]
    initial_state = {"A": 1, "B": 0, "C": 0, "D": 0}
    candidate = ["A", "B", "C"]

    # Filtramos las columnas y obtenemos los estados finales
    columns, values = find_columns_to_filter(variables, initial_state, candidate)
    filtered_states, deleted_positions = filter_states(states, columns, values)
    final_extended_matrix = eliminate_rows(extended_matrix, deleted_positions)

    print("Matriz final de estados:")
    print(filtered_states)
    print("\nPosiciones eliminadas:")
    print(deleted_positions)
    print("\nMatriz extendida final:")
    print(final_extended_matrix)

    # Especificar la fila a excluir (la fila correspondiente a D)
    fila_excluir = 3  # La fila correspondiente a D

    # Identificar pares de columnas repetidas y sumar las columnas correspondientes
    resultado_suma = identificar_y_sumar_pares(matriz_transpuesta, final_extended_matrix, fila_excluir)

    # Imprimir el resultado de la suma
    print("Resultado de la suma de columnas correspondientes a los pares repetidos:")
    print(resultado_suma)


def get_indices_marginalizar(states: np.ndarray, state: List[np.int64]) -> Tuple[dict, int]:
    """
    Obtiene los índices de los estados que coinciden con el conjunto de estados dado y su valor entero correspondiente.

    :param states: Matriz de estados.
    :param state: Estado actual representado como lista de valores.
    :return: Tupla que contiene un diccionario de índices y el valor entero correspondiente.
    """
    availableIndices = []
    indices = {}
    csValue = ""
    for i in range(len(state)):
        if state[i] != None:
            availableIndices.append(i)
            csValue = str(state[i]) + csValue

    for i in range(len(states)):
        key = tuple(states[i][j] for j in availableIndices)

        indices[key] = indices.get(key) + [i] if indices.get(key) else [i]

    if csValue == "":
        return indices, 0

    return indices, int(csValue, 2)

def imprimir_resultados(indices: Dict[Tuple[np.int64], List[int]], cs_value: int):
    """
    Imprime los índices y su correspondiente valor de manera legible.

    :param indices: Diccionario con los índices.
    :param cs_value: Valor convertido de binario a entero.
    """
    print("Índices encontrados:")
    if not indices:
        print("  No se encontraron índices.")
    else:
        for key, value in indices.items():
            # Convertir la clave a una representación legible
            clave_legible = f"({', '.join(map(str, key))})"
            # Imprimir la clave y sus índices
            print(f"  Clave: {clave_legible} -> Indices: {value}")

    print(f"\nValor de conjunto (cs_value) como entero: {cs_value}")


def margenalice_next_state(ns_indices, probabilities ):
    """
    La función `margenalice_next_state` calcula las probabilidades de transición para cada estado en una
    cadena de Markov basada en los índices y probabilidades dados.

    :param ns_indices: Diccionario con índices de los estados.
    :param probabilities: Tabla de probabilidades para las transiciones.
    :return: Tabla de transiciones de estados.
    """
    ns_transition_table = [[None] * len(ns_indices) for _ in range(len(probabilities))]
    current_column = 0

    for indices in ns_indices.values():
        for i in range(len(ns_transition_table)):
            probability = 0
            for j in range(len(indices)):
                probability += probabilities[i][indices[j]]
            ns_transition_table[i][current_column] = probability
        current_column += 1

    return ns_transition_table

def margenalice_current_state(cs_indices, ns_transition_table):
    cs_transition_table = [
        [None] * len(ns_transition_table[0]) for _ in range(len(cs_indices))
    ]

    current_row = 0
    for indices in cs_indices.values():
        for i in range(len(cs_transition_table[0])):
            probability = 0
            for j in range(len(indices)):
                probability += ns_transition_table[indices[j]][i]

            cs_transition_table[current_row][i] = probability / len(indices)

        current_row += 1

    return cs_transition_table


def obtener_tabla_probabilidades(currentState, nextState, probabilities, states):
    result = []
    csTransitionTable = []
    csIndices, csValueIndex = get_indices_marginalizar(states, currentState)
    missingCs = any(state is None for state in currentState)
    if missingCs:
        for i, state in enumerate(nextState):
            if state is not None:
                newNs = [None] * len(nextState)
                newNs[i] = nextState[i]

                nsIndices, _ = get_indices_marginalizar(states, newNs)
                nsTransitionTable = margenalice_next_state(nsIndices, probabilities)
                csTransitionTable = margenalice_current_state(
                    csIndices, nsTransitionTable
                )
                csValue = csTransitionTable[csValueIndex]

                if len(result) > 0:
                    result = np.kron(result, csValue)
                else:
                    result = csValue
    else:
        nsIndices, _ = get_indices_marginalizar(states, nextState)
        nsTransitionTable = margenalice_next_state(nsIndices, probabilities)

        csTransitionTable = margenalice_current_state(csIndices, nsTransitionTable)
        result = csTransitionTable[csValueIndex]

    return result


if __name__ == "__main__":
    extended_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    states = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    variables = ["A", "B", "C", "D"]
    initial_state = {"A": 1, "B": 0, "C": 0, "D": 0}
    candidate = ["A", "B", "C"]

    # TODO : debo validar el como saber que excluir
    matriz = generate_tpm(extended_matrix, states, variables, initial_state, candidate, 3)
    states = generate_states(len(candidate))
    filtered_state = {key: initial_state[key] for key in candidate}
    # Crear un arreglo de valores a partir del estado filtrado
    values = [int(value) for value in filtered_state.values()]
    result_indices, result_cs_value = get_indices_marginalizar(states,values)
    #imprimir_resultados(result_indices, result_cs_value)
    estado_futuro = "ABC"
    estado_presente = "AC"
    probabilidades = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ]
    estados = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    e_p = repr_current_to_array(estado_presente, [1,0]),
    e_f = repr_next_to_array(estado_futuro)
    tb = obtener_tabla_probabilidades(e_f,e_p,probabilidades,estados)
    print(tb)