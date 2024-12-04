from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Tuple


@dataclass
class BackgroundSystem:
    full_system: Dict[str, List[str]]
    candidate_system: Dict[str, List[str]]
    initial_state: List[int]
    states: List[List[int]]
    transition_matrix: List[List[int]]

    def eliminar_variables(
        self,
    ) -> Tuple[np.ndarray, List[int], List[Dict[str, Any]], np.ndarray]:
        # Inicializamos las variables
        full_system = self.full_system
        candidate_system = self.candidate_system
        initial_state = self.initial_state
        states = np.array(self.states)
        transition_matrix = np.array(self.transition_matrix)

        # Extraemos las variables de cada sistema
        full_variables = full_system["current"].copy()
        candidate_variables = candidate_system["current"]

        # Variables a eliminar (las que no están en el candidato) en orden inverso
        variables_to_eliminate = [
            var for var in full_variables if var not in candidate_variables
        ]
        variables_to_eliminate.reverse()  # Invertimos el orden

        # Mantenemos un registro de los índices originales
        original_indices = np.arange(len(states))
        suprimidos_por_paso = []

        # Para cada variable a eliminar, aplicamos el proceso
        for var in variables_to_eliminate:
            # Obtener el índice de la variable
            var_index = full_variables.index(var)

            # Determinar el valor inicial de la variable
            var_value = initial_state[var_index]

            # Identificar las filas a eliminar donde la variable tiene el valor CONTRARIO al inicial
            mask = states[:, var_index] != var_value

            # Guardamos los índices originales de las filas que se eliminarán
            indices_eliminados = original_indices[mask].tolist()

            # Actualizamos el registro de suprimidos
            suprimidos_por_paso.append(
                {
                    "variable": var,
                    "valor_inicial": var_value,
                    "indices_originales_suprimidos": indices_eliminados,
                }
            )

            # Actualizamos los estados, los índices originales y la matriz de transición
            states = states[~mask]
            original_indices = original_indices[~mask]
            transition_matrix = transition_matrix[~mask]

        # Obtenemos los índices de las columnas que queremos mantener
        indices_columnas = [full_variables.index(var) for var in candidate_variables]

        # Seleccionamos solo las columnas de las variables candidatas
        states = states[:, indices_columnas]

        # Creamos el estado inicial actualizado solo con las variables candidatas
        updated_initial_state = [
            initial_state[full_variables.index(var)] for var in candidate_variables
        ]

        return states, updated_initial_state, suprimidos_por_paso, transition_matrix

    def encontrar_repetidos_transpuesta(
        self, states: List[List[int]], matriz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Convertimos la lista de estados a una matriz numpy
        matriz_numpy = np.array(states)

        # Obtenemos la transpuesta de la matriz
        transpuesta = matriz_numpy.T
        print("Matriz transpuesta inicial:")
        print(transpuesta)

        # Extraemos las variables de cada sistema (en el futuro puedes también considerar las variables 'next' si es necesario)
        full_system = self.full_system
        candidate_system = self.candidate_system

        full_variables = full_system[
            "next"
        ].copy()  # Las variables de la columna "next" de full_system
        candidate_variables = candidate_system["next"]

        # Variables a eliminar (las que no están en el candidato)
        variables_to_eliminate = [
            var for var in full_variables if var not in candidate_variables
        ]
        print(
            f"Variables a eliminar (no están en el sistema candidato): {variables_to_eliminate}"
        )

        # Diccionario para almacenar las posiciones de las columnas con estados iguales
        columnas_iguales = {}

        # Eliminamos las columnas de la transpuesta en orden
        for var in variables_to_eliminate:
            # Obtener el índice de la variable en full_variables
            var_index = full_variables.index(var)

            # Verificar si el índice está dentro de los límites de la transpuesta
            if var_index < transpuesta.shape[0]:
                # Eliminamos la columna correspondiente a esta variable en la transpuesta
                transpuesta = np.delete(transpuesta, var_index, axis=0)
                print(
                    f"Después de eliminar la variable {var}, la transpuesta actualizada es:"
                )
                print(transpuesta)

                # Comparar las columnas de la transpuesta para encontrar estados iguales
                for i in range(transpuesta.shape[1]):
                    for j in range(i + 1, transpuesta.shape[1]):
                        if np.array_equal(transpuesta[:, i], transpuesta[:, j]):
                            if (i, j) not in columnas_iguales:
                                columnas_iguales[(i, j)] = transpuesta[:, i]
                print(f"Columnas iguales encontradas: {columnas_iguales}")

                # Actualizar la matriz original sumando las columnas en las posiciones donde los estados son iguales
                for i, j in columnas_iguales.keys():
                    if j < matriz.shape[1]:
                        matriz[:, i] += matriz[:, j]

                print("Matriz original después de actualizar sumando columnas iguales:")
                print(matriz)

                # Eliminar columnas duplicadas en la transpuesta y en la matriz original
                columnas_a_eliminar = set()
                for i, j in columnas_iguales.keys():
                    columnas_a_eliminar.add(j)

                # Convertir el set a una lista y ordenar en orden descendente
                columnas_a_eliminar = sorted(list(columnas_a_eliminar), reverse=True)
                for col in columnas_a_eliminar:
                    if col < transpuesta.shape[1]:
                        transpuesta = np.delete(transpuesta, col, axis=1)
                    if col < matriz.shape[1]:
                        matriz = np.delete(matriz, col, axis=1)

                # Limpiar el diccionario de columnas iguales para el siguiente paso
                columnas_iguales.clear()

            else:
                print(
                    f"Índice {var_index} fuera de los límites para la transpuesta con tamaño {transpuesta.shape[0]}"
                )

            # Actualizar full_variables eliminando la variable eliminada
            full_variables.remove(var)

        print("Transpuesta después de eliminar columnas duplicadas:")
        print(transpuesta)
        return transpuesta, matriz


# Ejemplo de uso
if __name__ == "__main__":
    system_data = {
        "full_system": {
            "current": ["At", "Bt", "Ct", "Dt"],
            "next": ["At+1", "Bt+1", "Ct+1", "Dt+1"],
        },
        "candidate_system": {
            "current": ["At", "Bt", "Ct"],
            "next": ["At+1", "Bt+1", "Ct+1"],
        },
        "initial_state": [1, 0, 0, 0],
        "states": [
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
        ],
        "transition_matrix": [
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
        ],
    }

    bg_system = BackgroundSystem(**system_data)

    # Ejecutamos el proceso
    updated_states, updated_initial_state, suprimidos, updated_transition_matrix = (
        bg_system.eliminar_variables()
    )
    print("\nEstados actualizados para las filas:")
    print(updated_transition_matrix)
    _, m = bg_system.encontrar_repetidos_transpuesta(
        system_data["states"], updated_transition_matrix
    )
    print("*" * 150)
    print(m)
# [[1 0 0 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0]
#  [0 0 0 0 0 1 0 0]
#  [0 1 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 1]
#  [0 0 0 0 0 1 0 0]
#  [0 0 0 1 0 0 0 0]]
