import numpy as np


def get_state_indices(states, state_repr, var_names, values=None):
    """
    Mapea la representación del estado a grupos de índices relevantes.

    Parámetros:
    - states: Lista de estados posibles (listas con valores 0 o 1).
    - state_repr: String que representa las variables relevantes (por ejemplo, 'BC').
    - var_names: Lista con los nombres de las variables (por ejemplo, ['A', 'B', 'C']).
    - values: Lista opcional con los valores de las variables en el estado actual.

    Devuelve:
    - Un diccionario donde las claves son las combinaciones de valores relevantes
      y los valores son listas con los índices correspondientes.
    - El valor binario del estado actual, si se proporciona `values`.
    """
    # Crear una máscara que indica cuáles variables se usan en `state_repr`.
    var_mask = [c in state_repr for c in var_names]

    # Identificar las posiciones de las variables que son relevantes.
    positions = [i for i, mask in enumerate(var_mask) if mask]

    # Agrupar los estados por las combinaciones de valores relevantes.
    state_groups = {}
    for i, state in enumerate(states):
        key = tuple(state[p] for p in positions)  # Crear la clave para el grupo.
        state_groups.setdefault(key, []).append(i)  # Agregar índice al grupo.

    # Si se proporcionan valores, calcular el valor binario del estado actual.
    state_value = 0
    if values:
        # Crear un mapa de las variables relevantes con sus valores correspondientes.
        value_map = {var_names[i]: v for i, v in zip(positions, values)}
        print("Value map:")
        print(value_map)
        # Construir un número binario como string a partir de los valores en `state_repr`.
        binary = "".join(str(value_map[c]) for c in state_repr)

        # Convertir el string binario a un entero.
        state_value = int(binary, 2)

    return state_groups, state_value


def compute_transition_matrix(groups, probabilities):
    """
    Calcula la matriz de transición sumando probabilidades por grupo.

    Parámetros:
    - groups: Diccionario con grupos de índices (de `get_state_indices`).
    - probabilities: Matriz con las probabilidades para cada estado.

    Devuelve:
    - Una matriz de transición con las probabilidades sumadas por grupo.
    """
    matrix = np.zeros((len(probabilities), len(groups)))  # Inicializar matriz de ceros.

    # Para cada grupo, sumar las probabilidades por estado y llenar la columna correspondiente.
    for col, indices in enumerate(groups.values()):
        matrix[:, col] = np.sum(probabilities[:, indices], axis=1)

    return matrix


def compute_state_matrix(cs_groups, transition_matrix):
    """
    Calcula la matriz de probabilidades para los estados relevantes.

    Parámetros:
    - cs_groups: Diccionario con los grupos del estado actual.
    - transition_matrix: Matriz de transición calculada previamente.

    Devuelve:
    - Una matriz de probabilidades ajustada por la cantidad de estados en cada grupo.
    """
    matrix = np.zeros(
        (len(cs_groups), transition_matrix.shape[1])
    )  # Inicializar matriz.

    # Para cada grupo, calcular el promedio de las probabilidades.
    for row, indices in enumerate(cs_groups.values()):
        matrix[row, :] = np.sum(transition_matrix[indices, :], axis=0) / len(indices)

    return matrix


def calculate_probabilities(
    states, probabilities, var_names, current_state, next_state, current_values
):
    """
    Calcula la tabla de probabilidades entre el estado actual y el siguiente estado.

    Parámetros:
    - states: Lista de estados posibles.
    - probabilities: Matriz con las probabilidades de transición.
    - var_names: Lista con los nombres de las variables.
    - current_state: String con las variables del estado actual.
    - next_state: String con las variables del siguiente estado.
    - current_values: Lista con los valores del estado actual.

    Devuelve:
    - Una matriz con las probabilidades calculadas para los estados especificados.
    """
    # Obtener los grupos de estados para el estado actual.
    cs_groups, cs_index = get_state_indices(
        states, current_state, var_names, current_values
    )

    if len(next_state) > 1:
        # Si hay múltiples variables en el siguiente estado, procesarlas por separado.
        result = None  # Inicializar resultado.

        for var in next_state:
            # Obtener los grupos para la variable actual del siguiente estado.
            ns_groups, _ = get_state_indices(states, var, var_names)

            # Calcular la matriz de transición y la matriz de estados.
            trans_matrix = compute_transition_matrix(ns_groups, probabilities)
            state_matrix = compute_state_matrix(cs_groups, trans_matrix)

            # Obtener las probabilidades para el estado actual.
            probs = state_matrix[cs_index]

            # Usar producto tensorial para combinar probabilidades.
            result = np.kron(result, probs) if result is not None else probs

        return result
    else:
        # Si solo hay una variable en el siguiente estado.
        ns_groups, _ = get_state_indices(states, next_state, var_names)
        trans_matrix = compute_transition_matrix(ns_groups, probabilities)
        state_matrix = compute_state_matrix(cs_groups, trans_matrix)
        return state_matrix[cs_index]


# Ejemplo de uso
if __name__ == "__main__":
    # Definir los estados posibles.
    states = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    # Definir la matriz de probabilidades.
    probabilities = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    # Nombres de las variables. pueden ser N variables
    var_names = ["A", "B", "C"]
    # current_values se refiere a los valores actuales de las variables es decir para N variables en el estado actual
    # tenemos n estados 0 o 1
    # Calcular probabilidades para el estado "C" hacia "BC".
    # result = calculate_probabilities(
    #     states,
    #     probabilities,
    #     var_names,
    #     current_state=" ",
    #     next_state="B",
    #     current_values=[],
    # )

    # result = calculate_probabilities(
    #     states,
    #     probabilities,
    #     var_names,
    #     current_state="A",
    #     next_state=" ",
    #     current_values=[1],
    # )

    print("Calculated probabilities:")
    # print(result)
