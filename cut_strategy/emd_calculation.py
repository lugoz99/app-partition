from itertools import product
from scipy.spatial.distance import cdist
from pyemd import emd
import numpy as np
import pandas as pd
from cut_strategy.probability import ProbabilityCalculator


class EMDEvaluator:
    """
    Encapsulates functionality for calculating Earth Mover's Distance (EMD)
    and related probability metrics.
    """

    @staticmethod
    def calculate_emd(
        probabilities: dict, state: str, original_probability: pd.DataFrame
    ) -> float:
        """
        Calculates the Earth Mover's Distance (EMD) between modified and original probabilities.

        Args:
            probabilities (dict): Dictionary of modified probabilities.
            state (str): State to evaluate.
            original_probability (pd.DataFrame): Original probability table.

        Returns:
            float: Calculated EMD value.
        """
        modified_prob = EMDEvaluator.get_probability_in_state(probabilities, state)
        hamming_matrix = EMDEvaluator.hamming_distance_matrix(
            modified_prob["state"].values
        )

        modified_values = np.ascontiguousarray(
            modified_prob["probability"].values, dtype=np.double
        )
        original_values = np.ascontiguousarray(
            original_probability.loc[state].to_numpy(), dtype=np.double
        )

        return round(emd(modified_values, original_values, hamming_matrix), 3)

    @staticmethod
    def get_probability_in_state(probabilities: dict, state: str) -> pd.DataFrame:
        """
        Extracts probabilities for a specific state from the given probabilities.

        Args:
            probabilities (dict): Dictionary of probability tables.
            state (str): State to extract probabilities for.

        Returns:
            pd.DataFrame: Joint probability table for the given state.
        """
        prob_in_state = {}
        for future, table in probabilities.items():
            if isinstance(table, np.ndarray):
                prob_in_state[future] = table
            elif state in table.index:
                prob_in_state[future] = table.loc[state].values

        return ProbabilityCalculator.calculate_joint_probability(prob_in_state)

    @staticmethod
    def hamming_distance_matrix(states: np.ndarray) -> np.ndarray:
        """
        Calculates the Hamming distance matrix for the given states.

        Args:
            states (np.ndarray): Array of states.

        Returns:
            np.ndarray: Hamming distance matrix.
        """
        states_as_list = list(map(lambda x: list(map(int, x)), states))
        return cdist(states_as_list, states_as_list, "hamming") * len(states_as_list[0])

    @staticmethod
    def sort_dataframe_by_order(dataframe: pd.DataFrame, order: list) -> pd.DataFrame:
        """
        Sorts a DataFrame by a specific order of states.

        Args:
            dataframe (pd.DataFrame): DataFrame to sort.
            order (list): List specifying the desired order of states.

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
        dataframe["order"] = pd.Categorical(
            dataframe["state"], categories=order, ordered=True
        )
        return dataframe.sort_values("order").drop(columns="order")
