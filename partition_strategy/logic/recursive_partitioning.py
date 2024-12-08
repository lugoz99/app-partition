import os
import sys
import numpy as np
from scipy.stats import wasserstein_distance as EMD
from ordered_set import OrderedSet
from typing import List, Dict, Tuple 
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append("d:/AnalisisYDiseñoAlgoritmos/Project-2024-02/project-ada")

from partition_strategy.services.background import BackgroundSystem
from partition_strategy.client.system_10 import get_evaluate_system
from partition_strategy.model.partition import PartitionModel
from partition_strategy.services.calculator_marginaze import (
    StateProbabilityCalculator,
)
from partition_strategy.services.inicializar_system import (
    OpcionSistema,
    StateHandler,
)
from partition_strategy.helpers.generate_states import generate_states
from partition_strategy.helpers.auxiliares import emd_with_hamming


@dataclass
class RecursiveCandidateSelection:
    v: OrderedSet
    current_values: List[int]
    states: List[List[int]] = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    var_names: List[str] = field(default_factory=list)
    particiones: List[Dict[str, OrderedSet]] = field(default_factory=list)

    def __post_init__(self):
        print(f"Inicializando con elementos: {self.v}")
        self.candidate = StateHandler(self.v, current_values=self.current_values)
        self.complete = StateProbabilityCalculator(
            self.states, self.probabilities, self.var_names
        )

    def get_original_distribution(self) -> np.ndarray:
        respuesta = self.candidate.generate_system(opcion=OpcionSistema.V)
        return self.complete.calculate_probabilities(
            current_state=respuesta.current_state,
            next_state=respuesta.next_state,
            current_values=respuesta.current_values,
        )

    def producto_tensorial(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a is None or b is None or a.size == 0 or b.size == 0:
            raise ValueError("Los arrays para el producto tensorial no son válidos.")
        return np.kron(a, b).flatten()

    def calculate_g(self, subconjunto: OrderedSet) -> float:
        try:
            pv = self.get_original_distribution()
            px = self.candidate.generate_system(opcion=OpcionSistema.VM, VM=subconjunto)
            px_complement = self.candidate.generate_system(
                opcion=OpcionSistema.VM_COMPLEMENTO, VM=subconjunto
            )

            px_p = self.complete.calculate_probabilities(
                current_state=px.current_state,
                next_state=px.next_state,
                current_values=px.current_values,
            )
            px_p_c = self.complete.calculate_probabilities(
                current_state=px_complement.current_state,
                next_state=px_complement.next_state,
                current_values=px_complement.current_values,
            )

            producto_tensor = self.producto_tensorial(px_p, px_p_c)
            return EMD(producto_tensor, pv)

        except Exception as e:
            print(f"Error en calculate_g para {subconjunto}: {e}")
            return float("inf")
        
    def calculate_g_partition(self, subconjunto_1: OrderedSet, subconjunto_2: OrderedSet) -> float:
        """
        Calcula g (EMD) para dos subconjuntos de una partición.
        """
        try:
            # Distribución original
            pv = self.get_original_distribution()

            # Generar sistemas para ambos subconjuntos
            px_1 = self.candidate.generate_system(opcion=OpcionSistema.VM, VM=subconjunto_1)
            px_2 = self.candidate.generate_system(opcion=OpcionSistema.VM, VM=subconjunto_2)

            # Calcular probabilidades para ambos subconjuntos
            px_p_1 = self.complete.calculate_probabilities(
                current_state=px_1.current_state,
                next_state=px_1.next_state,
                current_values=px_1.current_values,
            )
            px_p_2 = self.complete.calculate_probabilities(
                current_state=px_2.current_state,
                next_state=px_2.next_state,
                current_values=px_2.current_values,
            )

            # Producto tensorial entre las probabilidades de los subconjuntos
            producto_tensor = self.producto_tensorial(px_p_1, px_p_2)

            # Calcular la métrica EMD entre el producto tensorial y la distribución original
            return EMD(producto_tensor, pv)

        except Exception as e:
            print(f"Error en calculate_g_partition para {subconjunto_1}, {subconjunto_2}: {e}")
            return float('inf')  # Retornar un valor alto en caso de error


    def built_sequence_W(self, inicial: str) -> OrderedSet:
        """
        Construye la secuencia W basada en la métrica G.
        """
        W = OrderedSet([inicial])
        elementos_restantes = self.v - W
        print(f"Construyendo W desde el inicial: {inicial}")

        while len(W) < len(self.v):
            mejor_elemento = min(
                elementos_restantes,
                key=lambda e: self.calculate_g(W | OrderedSet([e]))
                - self.calculate_g(OrderedSet([e])),
                default=None,
            )
            if mejor_elemento:
                W.add(mejor_elemento)
                elementos_restantes.remove(mejor_elemento)
                print(f"Añadido {mejor_elemento} a W: {W}")
            else:
                print("No se encontró un mejor elemento. Finalizando construcción de W.")
                break

        return W

    def particionar_recursivo(self, v: OrderedSet = None):
        """
        Realiza particionado recursivo basado en W.
        """
        if v is None:
            v = self.v

        print(f"Iniciando partición recursiva con: {v}")

        # Condición base: detener si quedan 2 o menos elementos
        if len(v) <= 2:
            self._guardar_particion_final(v)
            return

        try:
            v1 = next(iter(v))  # Seleccionar un elemento inicial
            W = self.built_sequence_W(v1)
            v_n = W[-1]  # Último elemento de W

            print(f"Secuencia W construida: {W}")
            print(f"Último elemento de W: {v_n}")

            # Guardar partición
            self.particiones.append(
                {
                    "Partición 1": OrderedSet([v_n]),
                    "Partición 2": v - OrderedSet([v_n]),
                }
            )

            # Reducir el conjunto y continuar recursión
            v = v.remove(v_n)

            self.particionar_recursivo(v)

        except Exception as e:
            print(f"Error durante la recursión: {e}")
            raise

    def _guardar_particion_final(self, v: OrderedSet):
        if len(v) == 2:
            elementos = list(v)
            self.particiones.append(
                {
                    "Partición 1": OrderedSet([elementos[0]]),
                    "Partición 2": OrderedSet([elementos[1]]),
                }
            )


    def mostrar_particion_minima(self):
        min_g_valor = float('inf')
        min_particion = None
        for idx, particion in enumerate(self.particiones, start=1):
            # Extraer los subconjuntos de la partición
            subconjunto_1 = particion["Partición 1"]
            subconjunto_2 = particion["Partición 2"]
            # Calcular g para ambos subconjuntos
            g_valor = self.calculate_g_partition(subconjunto_1, subconjunto_2)
            if g_valor < min_g_valor:
                min_g_valor = g_valor
                min_particion = PartitionModel(subconjunto_1, subconjunto_2, g_valor)
        if min_particion:
            print(f"\nPartición con el valor mínimo de g: {min_particion}")
        else:
            print("\nNo se encontraron particiones.")

        return min_particion
# # Definir los estados posibles
# states = [
#     [0, 0, 0],
#     [1, 0, 0],
#     [0, 1, 0],
#     [1, 1, 0],
#     [0, 0, 1],
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1],
# ]

# # Definir la matriz de probabilidades
# probabilities = np.array(
#     [
#         [1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0],
#     ]
# )

# # Definir los nombres de las variables
# var_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

# # Definir el conjunto de variables y sus valores actuales
# v = OrderedSet(["at", "bt", "at+1", "bt+1"])
# current_values = [1, 0, 0]
# z: Dict[str, List[int]] = {}
# # Llenar el diccionario con listas de enteros
# ordered_set_d = OrderedDict.fromkeys(["at", "bt", "at+1", "bt+1"])

# for i, key in enumerate(ordered_set_d.keys(), start=1):
#     z[key] = [i]
# # Ahora v es {"at": [1], "bt": [2], "at+1": [3], "bt+1": [4]}
# print(z)
# # Crear una instancia de RecursiveCandidateSelection
# selector = RecursiveCandidateSelection(
#     z,
#     current_values=current_values,
#     states=states,
#     probabilities=probabilities,
#     var_names=var_names,
# )

# # Encontrar las particiones óptimas
# particiones_optimas = selector.encontrar_particiones_optimas(v)

# minimo_particion = selector.encontrar_particion_con_menor_emd(particiones_optimas)
# print(minimo_particion)


# # print(selector.calculate_g(OrderedSet(["at"]), v))
def transform_system(input_string):
    # Convertir la cadena en una lista de caracteres
    characters = list(input_string)
    
    # Crear las listas current y next
    current = [f"{char}t" for char in characters]
    next_state = [f"{char}t+1" for char in characters]
    
    # Construir el diccionario directamente
    full_system = {
        "current": current,
        "next": next_state
    }
    
    return full_system

# Ejemplo de uso


# Imprimir el resultado como diccionario
def transform_orderedset_to_system(ordered_set):
    # Separar los elementos en current y next
    current = []
    next_state = []
    
    for item in ordered_set:
        # Si termina en 't+1', va en next
        if item.endswith('t+1'):
            next_state.append(item.capitalize())
        # Si termina en 't', va en current
        elif item.endswith('t'):
            current.append(item.capitalize())
    
    # Crear el diccionario del sistema
    candidate_system = {
        "current": current,
        "next": next_state
    }
    
    return candidate_system

def main():
    # Configuración inicial del sistema
    estado_inicial = "1000000000"  # Estado inicial en binario
    sistema_completo = "ABCDEFGHIJ"  # Variables del sistema completo
    subsistemas_candidatos = {
        "ABCDEFGHIJ": [
            {"Subsistema": "ABCDEFGHIJt+1|ABCDEFGHIJt", "estado": None},
            {"Subsistema": "ABCDEt+1|ABCDEIJt", "estado": None},
            {"Subsistema": "ABCDEFGHIJt+1|ABCDEFGJt", "estado": None},
        ],
        "ABCDEFG": [
            {"Subsistema": "ABt+1|ABCt", "estado": None},
            {"Subsistema": "ACt+1|ABCt", "estado": None},
            {"Subsistema": "ABCt+1|ACt", "estado": None},
            {"Subsistema": "ABCt+1|ABCt", "estado": None},
        ],
        "ABCDE": [
            {"Subsistema": "ABCDt+1|ABCDt", "estado": None},
            {"Subsistema": "ABCDt+1|ABCDEt", "estado": None},
            {"Subsistema": "ABCDEt+1|ABCDt", "estado": None},
            {"Subsistema": "ABCt+1|ABCDEt", "estado": None},
        ],
    }

    # Llama a la función y obtiene las configuraciones seleccionadas
    var_names, v, current_values, z, matriz = get_evaluate_system(
        estado_inicial,
        sistema_completo,
        subsistemas_candidatos
    )
    c = np.array([int(digito) for digito in estado_inicial])
    system_data = {
        "full_system": transform_system(sistema_completo),
        "candidate_system":transform_orderedset_to_system(v),
        "initial_state": c,
        "states": generate_states(len(c)),
        "transition_matrix": matriz
    }

    bg_system = BackgroundSystem(**system_data)
    updated_states, updated_initial_state, suprimidos, updated_transition_matrix = (
        bg_system.eliminar_variables()
    )
    print(updated_transition_matrix)
    m = bg_system.encontrar_repetidos_transpuesta(
        system_data["states"], updated_transition_matrix
    )
    print(m)
    selector = RecursiveCandidateSelection(
        v,
        current_values=current_values,
        states=generate_states(len(current_values)),
        probabilities=m,
        var_names=var_names,
    )

    # Encontrar las particiones óptimas
    selector.particionar_recursivo()
    selector.mostrar_particion_minima()
    # Encontrar la partición con menor EMD
    # minimo_particion = selector.encontrar_particion_con_menor_emd()
    # print("\nPartición con menor EMD:")
    # print(minimo_particion)

if __name__ == "__main__":
    main()
