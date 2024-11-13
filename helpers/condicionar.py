import numpy as np
from typing import List, Tuple



def filter_states(
    states: np.ndarray, columns: List[int], values: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Filtra filas de 'states' según columnas y valores proporcionados.
    """
    # Validaciones mínimas
    if states.ndim < 2 or len(columns) != len(values):
        raise ValueError(
            "La matriz debe tener al menos dos dimensiones y listas coherentes."
        )

    # Encuentra las filas donde todas las condiciones se cumplen
    mask = np.all(states[:, columns] == values, axis=1)
    deleted_positions = np.where(mask)[0]

    # Elimina las filas que cumplen con el filtro
    filtered_states = np.delete(states, deleted_positions, axis=0)

    return filtered_states, deleted_positions.tolist()


def find_columns_to_filter(
    variables: List[str], initial_state: dict, candidate: List[str]
) -> Tuple[List[int], List[int]]:
    """
    Encuentra las columnas y valores para filtrar según las variables no candidatas.
    """
    columns = [i for i, var in enumerate(variables) if var not in candidate]
    values = [1 - initial_state[var] for var in variables if var not in candidate]

    return columns, values


def eliminate_rows(matrix: np.ndarray, positions: List[int]) -> np.ndarray:
    """
    Elimina filas de la matriz según las posiciones dadas.
    """
    return np.delete(matrix, positions, axis=0)


def identificar_y_sumar_pares(matriz, matriz_suma, fila_excluir):
    """Identifica columnas con estados repetidos y suma las columnas especificadas."""
    # Eliminar la fila que queremos excluir
    matriz_modificada = np.delete(matriz, fila_excluir, axis=0)

    # Identificar pares de columnas repetidas
    estados = {}
    pares = []
    for col in range(matriz_modificada.shape[1]):
        estado = tuple(matriz_modificada[:, col])
        if estado in estados:
            pares.append((estados[estado], col))
        else:
            estados[estado] = col

    # Sumar las columnas especificadas en los pares
    suma_resultante = np.array([matriz_suma[:, col1] + matriz_suma[:, col2] for col1, col2 in pares]).T

    return suma_resultante



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


if __name__ == "__main__":
    main()
