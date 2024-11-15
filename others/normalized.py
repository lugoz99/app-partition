import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class SystemState:
    current: List[str]
    next: List[str]


class SystemProcessor:
    def __init__(self, transition_matrix: np.ndarray, initial_state: List[int]):
        self.transition_matrix = transition_matrix
        self.initial_state = np.array(initial_state)

    @staticmethod
    def create_state_indices(states: List[str]) -> Dict[str, int]:
        """Crea un mapeo de nombres de estados a sus índices"""
        return {state: idx for idx, state in enumerate(states)}

    def get_binary_representation(self, state: int, num_bits: int) -> List[int]:
        """Convierte un estado decimal a su representación binaria"""
        return [int(x) for x in format(state, f"0{num_bits}b")]

    def binary_to_decimal(self, binary: List[int]) -> int:
        """Convierte una representación binaria de estado de nuevo a decimal"""
        return int("".join(map(str, binary)), 2)

    def marginalize_transition_matrix(
        self, full_system: SystemState, candidate_system: SystemState
    ) -> np.ndarray:
        """
        Crea una matriz de probabilidad de transición marginalizada de tamaño 2x8.
        """
        # Crear mapeo de índices para el sistema completo
        full_indices = self.create_state_indices(full_system.current)
        # Identificar el índice del estado candidato "At"
        candidate_vars = set(candidate_system.current)
        keep_indices = [
            idx for state, idx in full_indices.items() if state in candidate_vars
        ]
        # Calcular las dimensiones requeridas
        n_full = len(full_system.current)  # 3 estados en el sistema completo
        n_candidate = len(candidate_system.current)  # 1 estado en el sistema candidato
        num_cols = 2**n_full  # 8 columnas para el sistema completo
        num_rows = 2**n_candidate  # 2 filas para el sistema candidato (solo "At")

        # Inicializar la matriz marginalizada de tamaño 2x8
        marginalized = np.zeros((num_rows, num_cols))

        # Realizar la marginalización sin reducir columnas
        for current_state in range(2**n_full):
            current_binary = self.get_binary_representation(current_state, n_full)
            # Convertir la representación binaria del estado actual del candidato
            new_current = self.binary_to_decimal(
                [current_binary[i] for i in keep_indices]
            )

            # Llenar la fila correspondiente en la matriz de transición marginalizada
            print("current_state")
            print(current_state)
            marginalized[new_current, :] += self.transition_matrix[current_state, :]

        return marginalized

    def process_candidate_system(
        self, full_system: SystemState, candidate_system: SystemState
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Función principal de procesamiento que maneja la división del sistema
        """
        # Obtener la matriz de TPM marginalizada
        marginalized_tpm = self.marginalize_transition_matrix(
            full_system, candidate_system
        )

        # Procesar el estado inicial para el sistema candidato
        full_indices = self.create_state_indices(full_system.current)
        candidate_initial = [
            self.initial_state[full_indices[var]] for var in candidate_system.current
        ]

        return marginalized_tpm, candidate_initial


def main():
    # Definición de ejemplo del sistema
    system_data = {
        "full_system": SystemState(
            current=["At", "Bt", "Ct"], next=["At+1", "Bt+1", "Ct+1"]
        ),
        "candidate_system": SystemState(current=["At"], next=["At+1", "Bt+1", "Ct+1"]),
        "initial_state": [1],
        # "transition_matrix": [
        #     [0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005],
        #     [0.1, 0.7, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02],
        #     [0.05, 0.05, 0.7, 0.1, 0.03, 0.03, 0.02, 0.02],
        #     [0.02, 0.05, 0.1, 0.7, 0.05, 0.03, 0.03, 0.02],
        #     [0.01, 0.03, 0.03, 0.05, 0.7, 0.1, 0.05, 0.03],
        #     [0.01, 0.03, 0.03, 0.03, 0.1, 0.7, 0.05, 0.05],
        #     [0.005, 0.02, 0.02, 0.03, 0.05, 0.05, 0.7, 0.1],
        #     [0.005, 0.02, 0.02, 0.02, 0.03, 0.05, 0.1, 0.7],
        # ],
        "transition_matrix": [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
    }

    # Inicializar el procesador
    processor = SystemProcessor(
        np.array(system_data["transition_matrix"]), system_data["initial_state"]
    )

    try:
        # Procesar el sistema
        marginalized_tpm, candidate_initial = processor.process_candidate_system(
            system_data["full_system"], system_data["candidate_system"]
        )

        # Imprimir resultados
        print("\nOriginal Transition Matrix:")
        for row in system_data["transition_matrix"]:
            print(row)
        print("\nMarginalized Transition Matrix (Sin normalizar):")
        print(marginalized_tpm)
        print("\nCandidate Initial State:", candidate_initial)

    except Exception as e:
        print(f"Error durante el procesamiento: {e}")


if __name__ == "__main__":
    main()
