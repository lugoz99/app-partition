import numpy as np
import itertools
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from cut_strategy.marginalize import Marginalizer as get_marginalize_channel


class ProbabilityCalculator:
    """
    Encapsulates functionality for probability table calculations and tensor operations.
    """

    @staticmethod
    def calculate_joint_probability(probability_tables: dict) -> pd.DataFrame:
        """
        Computes the joint probability distribution from individual probability tables.

        Args:
            probability_tables (dict): Dictionary of probability tables by channel.

        Returns:
            pd.DataFrame: Joint probability distribution as a DataFrame.
        """
        prob_array = [probability_tables[key] for key in probability_tables]
        result = prob_array[0]
        combinations = ProbabilityCalculator.create_index_table(len(prob_array))

        for arr in prob_array[1:]:
            result = np.tensordot(result, arr, axes=0)

        final_table_prob = dict(zip(combinations, result.ravel()))
        df_final_tb = pd.DataFrame.from_dict(final_table_prob, orient="index")
        df_final_tb = df_final_tb.reset_index()
        df_final_tb.columns = ["state", "probability"]

        return df_final_tb

    @staticmethod
    def create_index_table(num_elements: int) -> list:
        """
        Creates a list of binary state combinations for the given number of elements.

        Args:
            num_elements (int): Number of elements.

        Returns:
            list: List of binary state combinations as strings.
        """
        combinations = list(itertools.product([0, 1], repeat=num_elements))
        combinations_string = ["".join(map(str, comb)) for comb in combinations]

        return combinations_string

    @staticmethod
    def get_probability_tables(process_data: dict, probs_table: dict) -> dict:
        """
        Calculates marginal probabilities for given channels and state.

        Args:
            process_data (dict): Process data including channels and states.
            probs_table (dict): Probability tables.

        Returns:
            dict: Marginal probabilities for each future channel.
        """
        future_channels = process_data["future"]
        current_channels = process_data["current"]
        state_current_channels = process_data["state"]
        all_channels = process_data["channels"]
        probability_tables = {}

        if future_channels == "":
            full_matrix = ProbabilityCalculator.get_original_probability(
                probs_table, current_channels, future_channels, all_channels
            )
            marginalize_table = get_marginalize_channel(
                full_matrix, current_channels, all_channels
            )

            row_sum = marginalize_table.loc[state_current_channels].sum()
            probability_tables[""] = np.array([row_sum, 1 - row_sum])

        for f_channel in future_channels:
            if current_channels == "":
                probability_tables[f_channel] = (
                    ProbabilityCalculator.get_prob_empty_current(probs_table[f_channel])
                )
                continue

            table_prob = get_marginalize_channel(
                probs_table[f_channel], current_channels, all_channels
            )

            row_probability = table_prob.loc[state_current_channels]
            probability_tables[f_channel] = row_probability.values

        return probability_tables

    @staticmethod
    def get_prob_empty_current(table: pd.DataFrame) -> np.ndarray:
        """
        Computes the mean probabilities for a table when no current channels are present.

        Args:
            table (pd.DataFrame): Probability table.

        Returns:
            np.ndarray: Mean probabilities.
        """
        return table.mean(axis=0).values

    @staticmethod
    def get_original_probability(
        probs_table: dict,
        current_channels: str,
        future_channels: str,
        all_channels: str,
    ) -> pd.DataFrame:
        """
        Computes the original joint probability matrix for the given channels.

        Args:
            probs_table (dict): Probability tables.
            current_channels (str): Current channel string.
            future_channels (str): Future channel string.
            all_channels (str): All channel string.

        Returns:
            pd.DataFrame: Joint probability matrix.
        """
        marg_table = {}

        for key, table in probs_table.items():
            if key in future_channels:
                new_table = get_marginalize_channel.get_marginalize_channel(
                    table, current_channels, all_channels
                )
                marg_table[key] = new_table

        key_index = next(iter(marg_table))
        index_tables = marg_table[key_index].index
        n_cols = 2 ** len(future_channels)
        full_matrix = pd.DataFrame(columns=[f"{key}" for key in range(n_cols)])

        for index in index_tables:
            prob_state = {}
            for key, table in marg_table.items():
                value = table.loc[index].values
                prob_state[key] = value

            joint_prob = ProbabilityCalculator.calculate_joint_probability(prob_state)
            columns = joint_prob["state"].values
            full_matrix.columns = columns
            full_matrix.loc[index] = joint_prob["probability"].values

        return full_matrix

    @staticmethod
    def tensor_product_partition(
        partition_left: pd.DataFrame, partition_right: pd.DataFrame, parts_exp: tuple
    ) -> pd.DataFrame:
        """
        Computes the tensor product of two partitions.

        Args:
            partition_left (pd.DataFrame): Left partition DataFrame.
            partition_right (pd.DataFrame): Right partition DataFrame.
            parts_exp (tuple): Tuple of partition expressions.

        Returns:
            pd.DataFrame: Tensor product result as a DataFrame.
        """
        part_left, part_right = parts_exp
        positions_channels = ProbabilityCalculator.get_position_elements(
            part_left[0], part_right[0]
        )
        result = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    ProbabilityCalculator.compute_tensor_product,
                    left,
                    right,
                    parts_exp,
                    positions_channels,
                )
                for _, left in partition_left.iterrows()
                for _, right in partition_right.iterrows()
            ]

            for future in futures:
                response = future.result()
                result.append(response)

        df_results = pd.DataFrame(result, columns=["state", "probability"])

        return df_results

    @staticmethod
    def compute_tensor_product(
        part_left: pd.Series,
        part_right: pd.Series,
        parts_exp: tuple,
        positions: tuple,
    ) -> list:
        """
        Computes the tensor product for a single pair of partitions.

        Args:
            part_left (pd.Series): Left partition row.
            part_right (pd.Series): Right partition row.
            parts_exp (tuple): Tuple of partition expressions.
            positions (tuple): Positions for tensor product.

        Returns:
            list: Tensor product result as [state, probability].
        """
        tensor_product = part_left["probability"] * part_right["probability"]
        index_product = ProbabilityCalculator.get_index_product(
            part_left["state"], part_right["state"], positions, parts_exp[0]
        )

        return [index_product, tensor_product]

    @staticmethod
    def get_index_product(
        state_left: str, state_right: str, positions: tuple, parts_exp: str
    ) -> str:
        """
        Computes the resulting index for the tensor product of two states.

        Args:
            state_left (str): State from the left partition.
            state_right (str): State from the right partition.
            positions (tuple): Positions for tensor product.
            parts_exp (str): Expression for the left partition future.

        Returns:
            str: Resulting state index.
        """
        if parts_exp == "":
            return state_right

        result_string = list(state_left)
        for index, char in zip(positions, state_right):
            result_string.insert(index, char)

        return "".join(result_string)

    @staticmethod
    def get_position_elements(channels_left: str, channels_right: str) -> tuple:
        """
        Computes the positions of elements in the tensor product.

        Args:
            channels_left (str): Channels on the left side.
            channels_right (str): Channels on the right side.

        Returns:
            tuple: Positions of elements for the tensor product.
        """
        unit_channels = "".join(set(channels_left + channels_right))
        order_channels = "".join
