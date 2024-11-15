import numpy as np


def eliminar_variables(system_data):
    # Inicializamos las variables
    full_system = system_data["full_system"]
    candidate_system = system_data["candidate_system"]
    initial_state = system_data["initial_state"]
    states = np.array(system_data["states"])

    # Extraemos las variables de cada sistema
    full_variables = full_system["current"].copy()
    candidate_variables = candidate_system["current"]

    # Variables a eliminar (las que no están en el candidato) en orden inverso
    variables_to_eliminate = [
        var for var in full_variables if var not in candidate_variables
    ]
    variables_to_eliminate.reverse()  # Invertimos el orden

    # Mantenemos un registro de los índices originales
    original_indices = np.arange(len(states))
    suprimidos_por_paso = []

    # Para cada variable a eliminar, aplicamos el proceso
    for var in variables_to_eliminate:
        # Obtener el índice de la variable
        var_index = full_variables.index(var)

        # Determinar el valor inicial de la variable
        var_value = initial_state[var_index]

        # Identificar las filas a eliminar donde la variable tiene el valor CONTRARIO al inicial
        mask = states[:, var_index] != var_value

        # Guardamos los índices originales de las filas que se eliminarán
        indices_eliminados = original_indices[mask].tolist()

        # Actualizamos el registro de suprimidos
        suprimidos_por_paso.append(
            {
                "variable": var,
                "valor_inicial": var_value,
                "indices_originales_suprimidos": indices_eliminados,
            }
        )

        # Actualizamos los estados y los índices originales
        states = states[~mask]
        original_indices = original_indices[~mask]

    # Obtenemos los índices de las columnas que queremos mantener
    indices_columnas = [full_variables.index(var) for var in candidate_variables]

    # Seleccionamos solo las columnas de las variables candidatas
    states = states[:, indices_columnas]

    # Creamos el estado inicial actualizado solo con las variables candidatas
    updated_initial_state = [
        initial_state[full_variables.index(var)] for var in candidate_variables
    ]

    return states, updated_initial_state, suprimidos_por_paso


def mostrar_proceso_detallado(system_data, states, initial_state, suprimidos):
    print("\nProceso de eliminación detallado:")
    print("\nEstado Inicial:")
    print(f"Variables completas: {system_data['full_system']['current']}")
    print(f"Variables candidatas: {system_data['candidate_system']['current']}")
    print(f"Estado inicial: {system_data['initial_state']}")

    print("\nTabla inicial:")
    headers = system_data["full_system"]["current"]
    for i, estado in enumerate(system_data["states"]):
        print(f"Índice {i}: {estado}")

    for paso in suprimidos:
        print(f"\nEliminación de {paso['variable']}:")
        print(f"Valor inicial: {paso['valor_inicial']}")
        print(f"Índices eliminados: {paso['indices_originales_suprimidos']}")

    print("\nEstado final:")
    print("Estados resultantes:")
    print(states)
    print(f"Estado inicial actualizado: {initial_state}")


# Ejemplo de uso
if __name__ == "__main__":
    system_data = {
        "full_system": {
            "current": ["At", "Bt", "Ct", "Dt"],
            "next": ["At+1", "Bt+1", "Ct+1", "Dt+1"],
        },
        "candidate_system": {
            "current": ["At", "Bt", "Ct"],
            "next": ["At+1", "Bt+1", "Ct+1", "Dt+1"],
        },
        "initial_state": [1, 0, 0, 0],
        "states": [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        "transition_matrix": [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
    }

    # Ejecutamos el proceso
    updated_states, updated_initial_state, suprimidos = eliminar_variables(system_data)

    # Mostramos los resultados detallados
    mostrar_proceso_detallado(
        system_data, updated_states, updated_initial_state, suprimidos
    )
