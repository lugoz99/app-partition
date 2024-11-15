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
        """Creates a mapping of state names to their indices"""
        return {state: idx for idx, state in enumerate(states)}

    def get_binary_representation(self, state: int, num_bits: int) -> List[int]:
        """Converts a decimal state to its binary representation"""
        return [int(x) for x in format(state, f"0{num_bits}b")]

    def binary_to_decimal(self, binary: List[int]) -> int:
        """Converts a binary state representation back to decimal"""
        return int("".join(map(str, binary)), 2)

    def marginalize_transition_matrix(
        self, full_system: SystemState, candidate_system: SystemState
    ) -> np.ndarray:
        """
        Creates a marginalized transition probability matrix for the candidate system
        """
        # Create mappings
        full_indices = self.create_state_indices(full_system.current)
        candidate_vars = set(candidate_system.current)

        # Get indices to keep
        keep_indices = [
            idx for state, idx in full_indices.items() if state in candidate_vars
        ]

        # Calculate dimensions
        n_full = len(full_system.current)
        n_candidate = len(candidate_system.current)
        new_dim = 2**n_candidate

        # Initialize marginalized matrix
        marginalized = np.zeros((new_dim, new_dim))

        # Perform marginalization
        for current_state in range(2**n_full):
            current_binary = self.get_binary_representation(current_state, n_full)
            new_current = self.binary_to_decimal(
                [current_binary[i] for i in keep_indices]
            )

            for next_state in range(2**n_full):
                next_binary = self.get_binary_representation(next_state, n_full)
                new_next = self.binary_to_decimal(
                    [next_binary[i] for i in keep_indices]
                )

                marginalized[new_current][new_next] += self.transition_matrix[
                    current_state
                ][next_state]

        # Normalize rows
        row_sums = marginalized.sum(axis=1, keepdims=True)
        normalized = np.divide(marginalized, row_sums, where=row_sums != 0)

        return normalized

    def process_candidate_system(
        self, full_system: SystemState, candidate_system: SystemState
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Main processing function that handles the system division
        """
        # Get marginalized TPM
        marginalized_tpm = self.marginalize_transition_matrix(
            full_system, candidate_system
        )

        # Process initial state for candidate system
        full_indices = self.create_state_indices(full_system.current)
        candidate_initial = [
            self.initial_state[full_indices[var]] for var in candidate_system.current
        ]

        return marginalized_tpm, candidate_initial


def main():
    # Example system definition
    system_data = {
        "full_system": SystemState(
            current=["At", "Bt", "Ct"], next=["At+1", "Bt+1", "Ct+1"]
        ),
        "candidate_system": SystemState(current=["At"], next=["At+1", "Bt+1", "Ct+1"]),
        "initial_state": [1, 0, 0],
        "states": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
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

    # Initialize processor
    processor = SystemProcessor(
        np.array(system_data["transition_matrix"]), system_data["initial_state"]
    )

    try:
        # Process the system
        marginalized_tpm, candidate_initial = processor.process_candidate_system(
            system_data["full_system"], system_data["candidate_system"]
        )

        # Print results
        print("\nOriginal Transition Matrix:")
        print(system_data["transition_matrix"])
        print("\nMarginalized Transition Matrix:")
        print(marginalized_tpm)
        print("\nCandidate Initial State:", candidate_initial)

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()
