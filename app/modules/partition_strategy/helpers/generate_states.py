import numpy as np


def generate_states(n):
    """
    Genera todos los estados posibles para n variables binarias.

    :param n: Número de variables binarias
    :return: Array de NumPy con todos los estados posibles en little-endian
    """
    # Lista para almacenar todos los estados
    states = []

    for i in range(2**n):
        # Iterar sobre todos los posibles valores (0 a 2^n - 1)
        # Convertir el número a su representación binaria
        # format(i, f'0{n}b') asegura que siempre tengamos n dígitos, rellenando con ceros a la izquierda si es necesario
        binary = format(i, f"0{n}b")

        # Convertir la cadena binaria a una lista de enteros
        # reversed(binary) invierte el orden para obtener la representación little-endian
        little_endian_state = [int(bit) for bit in reversed(binary)]

        # Agregar el estado a la lista de estados
        states.append(little_endian_state)

    # Convertir la lista de estados a un array de NumPy para mejor manipulación y visualización
    return np.array(states)


print(generate_states(4))


# @staticmethod
# def generate_states(n: int) -> np.ndarray:
#     """
#     Genera todos los estados posibles para n variables binarias.

#     :param n: Número de variables binarias
#     :return: Array de NumPy con todos los estados posibles en little-endian
#     """
#     # Lista para almacenar todos los estados
#     states = []

#     for i in range(2**n):
#         # Iterar sobre todos los posibles valores (0 a 2^n - 1)
#         # Convertir el número a su representación binaria
#         # format(i, f'0{n}b') asegura que siempre tengamos n dígitos, rellenando con ceros a la izquierda si es necesario
#         binary = format(i, f"0{n}b")

#         # Convertir la cadena binaria a una lista de enteros
#         # reversed(binary) invierte el orden para obtener la representación little-endian
#         little_endian_state = [int(bit) for bit in reversed(binary)]

#         # Agregar el estado a la lista de estados
#         states.append(little_endian_state)

#     # Convertir la lista de estados a un array de NumPy para mejor manipulación y visualización
#     return np.array(states)
