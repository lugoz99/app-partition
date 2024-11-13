import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pyemd import emd


@dataclass
class StateSystem:
    states: List[List[int]]
    var_names: List[str]
    probabilities: np.ndarray

    def get_variable_positions(self, variables: str) -> List[int]:
        """Obtiene las posiciones de las variables en var_names."""
        return [i for i, var in enumerate(self.var_names) if var in variables]

    def get_state_value(self, variables: str, current_values: List[int]) -> int:
        """Obtiene el valor binario del estado actual."""
        positions = self.get_variable_positions(variables)
        if len(positions) != len(current_values):
            raise ValueError(
                "El número de valores no coincide con las variables especificadas"
            )

        # Mapea las variables con sus valores
        value_map = {
            variables[i]: current_values[i] for i in range(len(current_values))
        }
        # Construye el número binario
        binary = "".join(str(value_map[var]) for var in variables)
        return int(binary, 2)

    def group_states(self, variables: str) -> Dict[Tuple[int, ...], List[int]]:
        """Agrupa los estados según las variables especificadas."""
        positions = self.get_variable_positions(variables)
        groups = {}
        for i, state in enumerate(self.states):
            key = tuple(state[p] for p in positions)
            groups.setdefault(key, []).append(i)
        return groups

    def calculate_table_distribution_probabilities(
        self, current_state: str, next_state: str, current_values: List[int]
    ) -> np.ndarray:
        """
        Calcula las probabilidades de transición entre estados.

        Argumentos:
            current_state: Variables del estado actual (ej: "C")
            next_state: Variables del siguiente estado (ej: "BC")
            current_values: Valores actuales de las variables
        """
        # Obtener el grupo actual y su índice
        current_groups = self.group_states(current_state)
        current_index = self.get_state_value(current_state, current_values)

        if len(next_state) == 1:
            # Caso simple: una sola variable en el siguiente estado
            next_groups = self.group_states(next_state)
            return self._calculate_single_transition(
                current_groups, next_groups, current_index
            )
        else:
            # Caso múltiple: varias variables en el siguiente estado
            return self._calculate_multiple_transition(
                current_groups, next_state, current_index
            )

    def _calculate_single_transition(
        self,
        current_groups: Dict[Tuple[int, ...], List[int]],
        next_groups: Dict[Tuple[int, ...], List[int]],
        current_index: int,
    ) -> np.ndarray:
        """Calcula probabilidades para una sola variable en el siguiente estado."""
        # Crear matriz de transición
        trans_matrix = np.zeros((len(self.states), len(next_groups)))
        for col, indices in enumerate(next_groups.values()):
            trans_matrix[:, col] = np.sum(self.probabilities[:, indices], axis=1)

        # Calcular matriz de estados
        state_matrix = np.zeros((len(current_groups), trans_matrix.shape[1]))
        for row, indices in enumerate(current_groups.values()):
            state_matrix[row, :] = np.sum(trans_matrix[indices, :], axis=0) / len(
                indices
            )

        return state_matrix[current_index]

    def _calculate_multiple_transition(
        self,
        current_groups: Dict[Tuple[int, ...], List[int]],
        next_state: str,
        current_index: int,
    ) -> np.ndarray:
        """Calcula probabilidades para múltiples variables en el siguiente estado."""
        result = None
        # Calcular probabilidades para cada variable y combinarlas
        for var in next_state:
            next_groups = self.group_states(var)
            probs = self._calculate_single_transition(
                current_groups, next_groups, current_index
            )
            result = np.kron(result, probs) if result is not None else probs
        return result

    def hamming_distance(self, a: int, b: int) -> int:
        """Calcula la distancia de Hamming entre dos enteros."""
        return (a ^ b).bit_count()

    def emd_with_hamming(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calcula la EMD entre dos distribuciones usando distancia de Hamming como métrica."""
        if len(u) != len(v):
            raise ValueError("Las distribuciones deben tener la misma longitud.")

        n = len(u)
        # Construcción de la matriz de costos basada en distancia de Hamming
        cost_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                cost_matrix[i, j] = self.hamming_distance(i, j)

        # Cálculo del Earth Mover's Distance (EMD)
        return emd(u, v, cost_matrix)

    def g(self, W: set, V: List) -> float:
        """Calcula la función g usando EMD."""
        P = np.array([1.0 if v in W else 0.0 for v in V])
        Q = np.array([1.0 if v not in W else 0.0 for v in V])
        return self.emd_with_hamming(P, Q)

    def find_candidate(self, W: set, candidates: set, V: List) -> int:
        """Encuentra el mejor candidato para añadir a W."""
        best_value = float("inf")  # Inicializa el mejor valor a infinito
        best_candidate = None  # Inicializa el mejor candidato como None

        # Itera sobre cada candidato
        for u in candidates:
            new_W = W.union(
                {u}
            )  # Crea un nuevo conjunto W añadiendo el candidato actual
            # Calcula el valor como la diferencia de g para el nuevo W y el candidato
            value = self.g(new_W, V) - self.g({u}, V)

            # Si el valor calculado es mejor que el mejor encontrado, actualiza
            if value < best_value:
                best_value = value  # Actualiza el mejor valor
                best_candidate = u  # Actualiza el mejor candidato

        return best_candidate

    def partition_algorithm_recursive(
        self, V, W=None, state1="", state2="", probabilities=[]
    ):
        """Implementa el algoritmo de particionado de manera recursiva."""

        if W is None:
            W = set()  # Inicializa W como un conjunto vacío si no se proporciona

        # Caso base: Si no hay más elementos para particionar, se devuelve
        if len(V) <= 1:
            return [tuple(W)], float(
                "inf"
            )  # Devuelve las particiones y un valor de EMD infinito

        v1 = V[0]  # Toma el primer elemento de V
        W.add(v1)  # Añade este elemento a W

        # Encontrar candidatos y construir W de manera recursiva
        candidates = set(V) - W  # Calcula los candidatos como el conjunto de V menos W
        while candidates:  # Mientras haya candidatos disponibles
            v2 = self.find_candidate(W, candidates, V)  # Encuentra el mejor candidato
            if v2 is not None:
                W.add(v2)  # Añade el candidato encontrado a W
                candidates.remove(v2)  # Elimina el candidato de la lista de candidatos

        # Generar la partición final
        U = set(V) - W  # Calcula U como los elementos que no están en W
        partitions = [(tuple(W), tuple(U))]  # Guarda la partición actual

        # Fusionar los elementos de W en un nuevo elemento
        new_element = tuple(W)  # Crea un nuevo elemento a partir de W
        new_V = list(U) + [new_element]  # Actualiza V para la próxima iteración

        # Llamada recursiva con la nueva V
        sub_partitions, min_emd = self.partition_algorithm_recursive(
            new_V, W, state1, state2, probabilities
        )
        partitions.extend(
            sub_partitions
        )  # Añade las particiones obtenidas recursivamente

        # Calcular EMD de la partición actual
        emd_value = self.g(
            W, self.calculate_transition_probabilities(state1, state2, probabilities)
        )

        return partitions, min(min_emd, emd_value)


# Ejemplo de uso
def main():
    # Definir sistema
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

    system = StateSystem(
        states=states, var_names=["A", "B", "C", "D"], probabilities=probabilities
    )

    # Calcular probabilidades
    current_state = "ABC"  # t
    nex_state = "ABC"  # t+1
    candidate = current_state + "/" + nex_state

    # create a matrix with de current state and the next state, the matrix is called conjunto this consist of de elements strings
    conjunto = (
        [["A", "B", "C", "D"], ["At", "Bt", "Ct", "Dt"]],  # 0 is current state
    )  # 1 is next state]

    # result = system.calculate_table_distribution_probabilities(
    #     current_state="ABC", next_state="ABC", current_values=[1, 0, 0]
    # )

    # print("Probabilidades calculadas:")
    # print(result)
    # V = ["A", "B", "C", "D"]  # Esta lista debe ser adaptada a tus datos reales
    # partitions = system.partition_algorithm(V)
    # partitions = system.partition_algorithm(V)
    # print(partitions)


if __name__ == "__main__":
    main()
