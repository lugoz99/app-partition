import re
from typing import Dict, List, Tuple
import numpy as np
from ordered_set import OrderedSet

from partition_strategy.data.supermatriz import get_matriz_10


def get_evaluate_system(
    estado_inicial: str,
    sistema_completo: str,
    subsistemas_candidatos: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[str], OrderedSet, Dict[str, int], Dict[str, List[int]], np.ndarray]:
    """
    Evalúa el sistema y permite al usuario seleccionar un sistema y subsistema.

    Args:
        estado_inicial (str): Estado inicial del sistema en binario.
        sistema_completo (str): Variables del sistema completo.
        subsistemas_candidatos (dict): Diccionario de sistemas candidatos y sus subsistemas.

    Returns:
        Tuple: var_names, v, current_values, z, matriz de transición.
    """
    # Mostrar sistemas candidatos disponibles
    print("\nSistemas Candidatos Disponibles:")
    for sistema, subsistemas in subsistemas_candidatos.items():
        print(f"- {sistema}")
        for idx, subsistema in enumerate(subsistemas, 1):
            print(f"  {idx}. {subsistema['Subsistema']}")

    # Selección del sistema
    sistema_seleccionado = (
        input("\nSeleccione el sistema candidato (ej: ABCDE): ").strip().upper()
    )
    if sistema_seleccionado not in subsistemas_candidatos:
        print("Sistema no encontrado. Intente nuevamente.")
        return get_evaluate_system(
            estado_inicial, sistema_completo, subsistemas_candidatos
        )

    # Selección del subsistema
    print("\nSubsistemas disponibles:")
    subsistemas = subsistemas_candidatos[sistema_seleccionado]
    for idx, subsistema in enumerate(subsistemas, 1):
        print(f"{idx}. {subsistema['Subsistema']}")

    try:
        seleccion = int(input("\nSeleccione el número del subsistema: "))
        if 1 <= seleccion <= len(subsistemas):
            subsistema_seleccionado = subsistemas[seleccion - 1]["Subsistema"]
        else:
            print("Selección inválida. Intente nuevamente.")
            return get_evaluate_system(
                estado_inicial, sistema_completo, subsistemas_candidatos
            )
    except ValueError:
        print("Entrada inválida. Intente nuevamente.")
        return get_evaluate_system(
            estado_inicial, sistema_completo, subsistemas_candidatos
        )

    # Extraer variables y estados para el tiempo actual (T)
    parte_t = subsistema_seleccionado.split("|")[-1]
    var_names = list(re.findall(r"[A-Z]", parte_t))

    # Crear current_values como un diccionario basado en el estado presente
    current_values = {
        var: int(estado_inicial[sistema_completo.index(var)]) for var in var_names
    }

    # Construir `v` y `z`
    v = OrderedSet(
        [f"{var.lower()}t" for var in var_names]
        + [f"{var.lower()}t+1" for var in var_names]
    )
    z = {f"{var.lower()}t": [idx + 1] for idx, var in enumerate(var_names)}
    z.update({f"{var.lower()}t+1": [idx + 1] for idx, var in enumerate(var_names)})

    # Obtener la matriz de transición (ejemplo con get_matriz_10)
    matriz = get_matriz_10()

    return var_names, v, current_values, z, matriz
