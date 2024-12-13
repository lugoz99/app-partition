import numpy as np
import re


class StateChannel:
    """
    Encapsulates functionality for state channel probability calculations.
    """

    @staticmethod
    def get_matrix_state_channel_f(patterns: dict, data_frame: np.ndarray) -> dict:
        """
        Determines the probability of the future channel of state 1 after a previous state.

        Args:
            patterns (dict): Dictionary of patterns and their indexes.
            data_frame (np.ndarray): DataFrame containing channel data.

        Returns:
            dict: A dictionary mapping patterns to their probability matrices.
        """
        matrix_state = {}
        for pattern, indexs_pattern in patterns.items():
            next_states = StateChannel.calculate_possible_next_states(
                indexs_pattern, data_frame
            )

            end_index = data_frame.index.max()
            denominator = len(indexs_pattern) if indexs_pattern else 0
            if denominator != 0 and end_index in indexs_pattern:
                denominator -= 1

            tuple_clear = StateChannel.clear_str_pattern(pattern)
            matrix_state[tuple_clear] = (
                next_states / denominator if denominator != 0 else 0
            )

        return matrix_state

    @staticmethod
    def calculate_possible_next_states(
        index_pattern: list, data_frame: np.ndarray
    ) -> np.ndarray:
        """
        Counts the number of times the future channel is 1 after a previous state.

        Args:
            index_pattern (list): List of indices matching the pattern.
            data_frame (np.ndarray): DataFrame containing channel data.

        Returns:
            np.ndarray: Sum of states for possible future channels.
        """
        state_row_sum = np.zeros(len(data_frame.columns), dtype=np.int64)
        if index_pattern is None:
            return state_row_sum

        for index in index_pattern:
            if index + 1 < len(data_frame):
                state_row = data_frame.loc[index + 1]
                state_row_sum += state_row

        return state_row_sum

    @staticmethod
    def clear_str_pattern(tupla: tuple) -> str:
        """
        Cleans and converts a tuple pattern into a string without separators.

        Args:
            tupla (tuple): The tuple to clean.

        Returns:
            str: Cleaned string representation of the tuple.
        """
        str_tuple = str(tupla)
        tuple_clear = re.sub(r"[,\s\(\)]", "", str_tuple)

        return tuple_clear
