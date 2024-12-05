import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# Enumeración para opciones de inicialización del sistema


@dataclass
class StateProbabilityCalculator:
    """Clase para calcular probabilidades entre estados."""

    states: np.ndarray
    probabilities: np.ndarray
    var_names: List[str]
    
    
    def get_state_indices(
        self, state_repr: str, values: List[int] = None
    ) -> Tuple[Dict[Tuple[int, ...], List[int]], int]:
        var_mask = [c in state_repr for c in self.var_names]
        positions = [i for i, mask in enumerate(var_mask) if mask]
        state_groups = {}
        for i, state in enumerate(self.states):
            key = tuple(state[p] for p in positions)
            state_groups.setdefault(key, []).append(i)
        state_value = 0
        if values:
            value_map = {self.var_names[i]: v for i, v in zip(positions, values)}
            binary = "".join(str(value_map[c]) for c in state_repr)
            state_value = int(binary, 2)
        return state_groups, state_value

    def compute_transition_matrix(
        self, groups: Dict[Tuple[int, ...], List[int]]
    ) -> np.ndarray:
        """Calcula la matriz de transición para los grupos dados."""
        return np.array(
            [
                np.sum(self.probabilities[:, indices], axis=1)
                for indices in groups.values()
            ]
        ).T

    def compute_state_matrix(
        self, cs_groups: Dict[Tuple[int, ...], List[int]], transition_matrix: np.ndarray
    ) -> np.ndarray:
        """Calcula la matriz de estados basada en la matriz de transición."""
        matrix = np.zeros((len(cs_groups), transition_matrix.shape[1]))
        for row, indices in enumerate(cs_groups.values()):
            matrix[row, :] = np.sum(transition_matrix[indices, :], axis=0) / len(
                indices
            )
        return matrix

    def calculate_probabilities(
        self, current_state: str, next_state: str, current_values: List[int]
    ) -> np.ndarray:
        """Calcula las probabilidades de transición entre estados."""
        cs_groups, cs_index = self.get_state_indices(current_state, current_values)
        if len(next_state) > 1:
            result = None
            for var in next_state:
                ns_groups, _ = self.get_state_indices(var)
                trans_matrix = self.compute_transition_matrix(ns_groups)
                state_matrix = self.compute_state_matrix(cs_groups, trans_matrix)
                probs = state_matrix[cs_index]
                result = np.kron(result, probs) if result is not None else probs
            return result
        else:
            ns_groups, _ = self.get_state_indices(next_state)
            trans_matrix = self.compute_transition_matrix(ns_groups)
            state_matrix = self.compute_state_matrix(cs_groups, trans_matrix)
            return state_matrix[cs_index]


if __name__ == "__main__":
    # Definir los estados posibles. , entrada del usuario, puede ser mas grande
    states = np.array([
                [0, 0, 0, 0 ,0],
                [1, 0, 0, 0 ,0],
                [0, 1, 0, 0 ,0],
                [1, 1, 0, 0 ,0],
                [0, 0, 1, 0 ,0],
                [1, 0, 1, 0 ,0],
                [0, 1, 1, 0 ,0],
                [1, 1, 1, 0 ,0],
                [0, 0, 0, 1 ,0],
                [1, 0, 0, 1 ,0],
                [0, 1, 0, 1 ,0],
                [1, 1, 0, 1 ,0],
                [0, 0, 1, 1 ,0],
                [1, 0, 1, 1 ,0],
                [0, 1, 1, 1 ,0],
                [1, 1, 1, 1 ,0],
                [0, 0, 0, 0 ,1],
                [1, 0, 0, 0 ,1],
                [0, 1, 0, 0 ,1],
                [1, 1, 0, 0 ,1],
                [0, 0, 1, 0 ,1],
                [1, 0, 1, 0 ,1],
                [0, 1, 1, 0 ,1],
                [1, 1, 1, 0 ,1],
                [0, 0, 0, 1 ,1],
                [1, 0, 0, 1 ,1],
                [0, 1, 0, 1 ,1],
                [1, 1, 0, 1 ,1],
                [0, 0, 1, 1 ,1],
                [1, 0, 1, 1 ,1],
                [0, 1, 1, 1 ,1],
                [1, 1, 1, 1 ,1],])
      # Definir la matriz de probabilidades.Puede ser mas grande
    probabilities = TPM = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

], dtype=float)

    # Nombres de las variables.
    var_names = [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]  # entrada del usuario , puede ser mas grande

    # Crear una instancia del calculador de probabilidades.
    calculator = StateProbabilityCalculator(states, probabilities, var_names)

    # Calcular probabilidades para el estado "C" hacia "BC". # para diferentes variables
    current_state = "A"
    next_state = "ABCDE"
    current_values = [1]
    # result = calculator.calculate_probabilities(
    #     current_state, next_state, current_values
    # )

    print("Calculated probabilities para la distribucion :")
    print(calculator.calculate_probabilities(current_state, next_state, current_values))
