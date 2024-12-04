import sys
import os
from typing import Dict, List
import numpy as np
from ordered_set import OrderedSet
import ordered_set

from app.modules.partition_strategy.logic.recursive_partitioning import RecursiveCandidateSelection



sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    states = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    # Definir la matriz de probabilidades.Puede ser mas grande
    probabilities = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    # Nombres de las variables.
    var_names = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ]  # entrada del usuario , puede ser mas grande

    # Crear una instancia del calculador de probabilidades.

    V = OrderedSet(["at", "bt", "at+1", "bt+1"])
    # V = ["at", "bt", "at+1", "bt+1"]
    current_values = [1, 0, 0]  # Estado inicial

    v: Dict[str, List[int]] = {}

    # Llenar el diccionario con listas de enteros
    for i, key in enumerate(ordered_set.keys(), start=1):
        v[key] = [i]

    # Ahora v es {"at": [1], "bt": [2], "at+1": [3], "bt+1": [4]}
    print(v)
    logic = RecursiveCandidateSelection(
        v,
        current_values=current_values,
        states=states,
        probabilities=probabilities,
        var_names=var_names,
    )
    particiones_optimas = logic.encontrar_particiones_optimas(V)
    for ciclo, particion1, particion2, emd_valor in particiones_optimas:
        print(
            f"{ciclo} - Partición 1: {particion1}, Partición 2: {particion2}, EMD: {emd_valor}"
        )
