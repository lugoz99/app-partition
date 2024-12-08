import os
import sys
import numpy as np
from scipy.stats import wasserstein_distance as EMD
from ordered_set import OrderedSet
from typing import List, Dict 
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
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


@dataclass
class RecursiveCandidateSelection:
    v: Dict[str, List[int]]
    current_values: List[int]
    states: List[List[int]] = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    var_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.candidate = StateHandler(self.v, current_values=self.current_values)
        self.complete = StateProbabilityCalculator(
            self.states, self.probabilities, self.var_names
        )
        
    def get_original_distribution(self) -> np.ndarray:
        respuesta = self.candidate.generate_system(opcion=OpcionSistema.V)
        return self.complete.calculate_probabilities(
            current_state= respuesta.current_state,
            next_state=respuesta.next_state,
            current_values=respuesta.current_values,
        )

    def producto_tensorial(self, a: np.ndarray, b: np.ndarray):
        return np.kron(a, b).flatten()

    def calcular_probabilidad_fusionada(self, fusion_node):
        if isinstance(fusion_node, tuple):
            resultados_internos = []
            for sub_node in fusion_node:
                sub_probs = self.calcular_probabilidad_fusionada(sub_node)
                resultados_internos.append(sub_probs)

            probabilidad_fusionada = resultados_internos[0]
            for siguiente in resultados_internos[1:]:
                probabilidad_fusionada = self.producto_tensorial(
                    probabilidad_fusionada, siguiente
                )

            probabilidad_fusionada = probabilidad_fusionada / np.sum(
                probabilidad_fusionada
            )
            return probabilidad_fusionada
        else:
            v = self.candidate.inicializar_sistema(
                opcion=OpcionSistema.VM, VM=[fusion_node]
            )
            simple_probs = self.complete.calculate_probabilities(
                v["Current State"], v["Next State"], v["Current Values"]
            )
            return simple_probs / np.sum(simple_probs)

    def calculate_g(self, subset: OrderedSet, total_set: OrderedSet) -> float:
        p_v_set = None
        p_v_set_complement = None
        v = self.candidate.inicializar_sistema(opcion=OpcionSistema.V)
        p_v = self.complete.calculate_probabilities(
            v["Current State"], v["Next State"], v["Current Values"]
        )
        if any(isinstance(node, tuple) for node in subset):
            p_v_set = self.calcular_probabilidad_fusionada(tuple(subset))
        else:
            v_w_u = self.candidate.inicializar_sistema(
                opcion=OpcionSistema.VM, VM=list(subset)
            )
            v_w_u_complement = self.candidate.inicializar_sistema(
                opcion=OpcionSistema.VM_COMPLEMENTO, VM=list(subset)
            )
            p_v_set = self.complete.calculate_probabilities(
                v_w_u["Current State"], v_w_u["Next State"], v_w_u["Current Values"]
            )
            p_v_set_complement = self.complete.calculate_probabilities(
                v_w_u_complement["Current State"],
                v_w_u_complement["Next State"],
                v_w_u_complement["Current Values"],
            )
            if np.sum(p_v_set) != 1:
                p_v_set = p_v_set / np.sum(p_v_set)
            if np.sum(p_v_set_complement) != 1:
                p_v_set_complement = p_v_set_complement / np.sum(p_v_set_complement)

        tensor_product_vm = self.producto_tensorial(p_v_set, p_v)
        tensor_product_vm_complemento = self.producto_tensorial(p_v_set_complement, p_v)

        emd_value = EMD(tensor_product_vm, tensor_product_vm_complemento)
        return emd_value

    def encontrar_particiones_optimas(self, V: OrderedSet, particiones=[]):
        """
        Encuentra las particiones óptimas del conjunto V utilizando el algoritmo recursivo.
        """
        if len(V) < 2:
            raise ValueError("El conjunto V debe tener al menos dos elementos.")

        n = len(V)
        W = [OrderedSet() for _ in range(n + 1)]
        W[1] = OrderedSet([list(V)[0]])

        for i in range(2, n + 1):
            vi_min = None
            min_valor = float("inf")
            for vi in V:
                if vi not in W[i - 1]:
                    g_union = self.calculate_g(tuple(W[i - 1] | OrderedSet([vi])), tuple(V))
                    g_single = self.calculate_g((vi,), tuple(V))
                    valor = g_union - g_single
                    if valor < min_valor:
                        min_valor = valor
                        vi_min = vi

            W[i] = W[i - 1] | OrderedSet([vi_min])

        par_candidato = (list(W[n - 1])[-1], list(W[n])[-1])
        particion1 = OrderedSet([par_candidato[1]])
        particion2 = OrderedSet(V) - particion1
        particiones.append((particion1, particion2))

        if len(V) > 2:
            u = tuple(par_candidato)
            nuevo_V = OrderedSet([x for x in V if x not in par_candidato] + [u])
            particiones.append((OrderedSet([u]), OrderedSet(nuevo_V) - OrderedSet([u])))

            self.encontrar_particiones_optimas(nuevo_V, particiones)

        # Recalcular g para todas las particiones al final
        particiones_recalculadas = []
        for particion in particiones:
            print("Particion->",particion)
            g_value = self.calculate_g(tuple(particion[0]), tuple(V))
            particiones_recalculadas.append((particion[0], particion[1], g_value))

        particion_models = [
            PartitionModel(p[0], p[1], p[2]) for p in particiones_recalculadas
        ]
        return particion_models

    @staticmethod
    def encontrar_particion_con_menor_emd(particiones):
        particiones_dict = [p.to_dict() for p in particiones]
        particion_minima = min(particiones_dict, key=lambda x: x["emd_valor"])
        return {
            "particion1": list(particion_minima["particion1"]),
            "particion2": list(particion_minima["particion2"]),
            "emd_valor": particion_minima["emd_valor"],
        }


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
        z,
        current_values=current_values,
        states=generate_states(len(current_values)),
        probabilities=m,
        var_names=var_names,
    )

    # Encontrar las particiones óptimas
    particiones_optimas = selector.encontrar_particiones_optimas(v)

    # Encontrar la partición con menor EMD
    minimo_particion = selector.encontrar_particion_con_menor_emd(particiones_optimas)
    print("\nPartición con menor EMD:")
    print(minimo_particion)

if __name__ == "__main__":
    main()
