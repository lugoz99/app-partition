import numpy as np
import re

### Determines the probability of the future channel of state 1 after a previous state.
def get_matrix_state_channel_f(patterns, data_frame):
    matrix_state = {}
    for pattern in patterns:
        indexs_pattern = patterns[pattern]
        next_states = calculate_possible_next_states(
            indexs_pattern, data_frame)
        
        end_index = data_frame.index.max()
        denominador = len(indexs_pattern) if indexs_pattern else 0 
        if denominador != 0 and end_index in indexs_pattern:
            denominador = denominador - 1

        tuple_clear = clear_str_pattern(pattern)
        matrix_state[tuple_clear] = next_states / \
            denominador if denominador != 0 else 0

    return matrix_state


### Counts the number of times the future channel is 1 after a previous state.
def calculate_possible_next_states(index_pattern, data_frame):
    state_row_sum = np.zeros(len(data_frame.columns), dtype=np.int64)
    if index_pattern is None:
        return state_row_sum
    
    for index in index_pattern:
        if index + 1 < len(data_frame):
            state_row = data_frame.loc[index + 1]
            state_row_sum = state_row_sum + state_row

    return state_row_sum


def clear_str_pattern(tupla):
    str_tuple = str(tupla)
    tuple_clear = re.sub(r'[,\s\(\)]', '', str_tuple)
    
    return tuple_clear
