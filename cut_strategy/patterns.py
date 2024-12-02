from itertools import product


### Return a dictionary where the keys are the unique patterns containing 0 and 1
### found in the data_frame and the values are the list of indexes where the pattern appears.
def find_patterms(data_frame):
    convinations = create_convination(len(data_frame.columns))
    unique_patterns = dict.fromkeys(convinations, None)

    for index, row in data_frame.iterrows():
        row_pattern = tuple(row)
        if unique_patterns[row_pattern] is None:
            unique_patterns[row_pattern] = []
        unique_patterns[row_pattern].append(index)

    return unique_patterns


def create_convination(num_columns):
    convinations = list(product([0, 1], repeat=num_columns))
    convinations = [combo[::-1] for combo in convinations]

    return convinations