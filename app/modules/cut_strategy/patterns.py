from itertools import product


class PatternFinder:
    """
    Encapsulates functionality for finding patterns in a data frame.
    """

    @staticmethod
    def find_patterns(data_frame):
        """
        Finds unique patterns of 0s and 1s in a data frame and their indices.

        Args:
            data_frame (pd.DataFrame): Data frame to analyze.

        Returns:
            dict: Dictionary with patterns as keys and indices as values.
        """
        combinations = PatternFinder.create_combinations(len(data_frame.columns))
        unique_patterns = dict.fromkeys(combinations, None)

        for index, row in data_frame.iterrows():
            row_pattern = tuple(row)
            if unique_patterns[row_pattern] is None:
                unique_patterns[row_pattern] = []
            unique_patterns[row_pattern].append(index)

        return unique_patterns

    @staticmethod
    def create_combinations(num_columns):
        """
        Creates all combinations of 0 and 1 for a given number of columns.

        Args:
            num_columns (int): Number of columns to consider.

        Returns:
            list: List of all binary combinations.
        """
        combinations = list(product([0, 1], repeat=num_columns))
        return [combo[::-1] for combo in combinations]
