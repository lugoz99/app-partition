import os
import sys





sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append("d:/AnalisisYDiseñoAlgoritmos/Project-2024-02/project-ada")

from app.partition_strategy.services.background import BackgroundSystem
from app.partition_strategy.client.system_10 import get_evaluate_system
from app.partition_strategy.model.partition import PartitionModel
from app.partition_strategy.services.calculator_marginaze import (
    StateProbabilityCalculator,
)

from app.partition_strategy.services.calculator_marginaze import (
    StateProbabilityCalculator,
)
from app.partition_strategy.services.inicializar_system import (
    OpcionSistema,
    ProbabilidadM,
)

from app.partition_strategy.helpers.generate_states import generate_states
from enum import Enum
from typing import List, Dict, Tuple, OrderedDict
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import wasserstein_distance as EMD
from ordered_set import OrderedSet


@dataclass
class RecursiveCandidateSelection:
    v: Dict[str, List[int]]
    current_values: List[int]
    states: List[List[int]] = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    var_names: List[str] = field(default_factory=list)
    candidate: ProbabilidadM = field(init=False)
    complete: StateProbabilityCalculator = field(init=False)

    def __post_init__(self):
        V = list(self.v) if isinstance(self.v, OrderedSet) else list(self.v.keys())
        self.candidate = ProbabilidadM(
        V=V, current_values=self.current_values
        )
        self.complete = StateProbabilityCalculator(
            self.states, self.probabilities, self.var_names
        )

    def calcular_probabilidad_fusionada(self, fusion_node):
        """
        Procesa nodos fusionados o simples, comenzando desde los más internos.
        """
        if isinstance(fusion_node, tuple):  # Nodo fusionado
            print(f"Procesando tupla: {fusion_node}")
            
            # Procesar el nodo más interno primero
            resultados_internos = []
            for sub_node in fusion_node:
                # Inicializar y calcular probabilidades recursivamente
                sub_probs = self.calcular_probabilidad_fusionada(sub_node)
                resultados_internos.append(sub_probs)
            
            # Calcular el producto tensorial entre los resultados
            probabilidad_fusionada = resultados_internos[0]
            for siguiente in resultados_internos[1:]:
                probabilidad_fusionada = np.kron(probabilidad_fusionada, siguiente)
            
            print(f"Probabilidad combinada para {fusion_node}: {probabilidad_fusionada}")
            return probabilidad_fusionada

        else:  # Nodo simple
            print(f"Inicializando nodo simple: {fusion_node}")
            v = self.candidate.inicializar_sistema(
                opcion=OpcionSistema.VM, VM=[fusion_node]
            )
            simple_probs = self.complete.calculate_probabilities(
                v["Current State"], v["Next State"], v["Current Values"]
            )
            print(f"Probabilidad para nodo {fusion_node}: {simple_probs}")
            return simple_probs


    def calculate_g(self, subset: OrderedSet, total_set: OrderedSet) -> float:
        """
        Calcula la función g basada en las probabilidades de los subconjuntos y sus complementos.
        """
        print(f"Procesando subset: {subset}")
        
        # 1. Probabilidad del conjunto completo V
        v = self.candidate.inicializar_sistema(opcion=OpcionSistema.V)
        p_v = self.complete.calculate_probabilities(
            v["Current State"], v["Next State"], v["Current Values"]
        )
        print(f"Probabilidad del conjunto total (V): {p_v}")
        
        # 2. Probabilidad del subconjunto actual (subset)
        if any(isinstance(node, tuple) for node in subset):
            # Si hay tuplas, procesarlas recursivamente
            p_v_set = self.calcular_probabilidad_fusionada(tuple(subset))
        else:
            # Procesar el subset como un conjunto de nodos simples
            v_w_u = self.candidate.inicializar_sistema(opcion=OpcionSistema.VM, VM=list(subset))
            p_v_set = self.complete.calculate_probabilities(
                v_w_u["Current State"], v_w_u["Next State"], v_w_u["Current Values"]
            )
        print(f"Probabilidad del subconjunto (subset): {p_v_set}")
        
        # 3. Probabilidad del complemento del subconjunto (total_set \ subset)
        v_w_u_complement = self.candidate.inicializar_sistema(
            opcion=OpcionSistema.VM_COMPLEMENTO, VM=list(subset)
        )
        p_v_set_complement = self.complete.calculate_probabilities(
            v_w_u_complement["Current State"],
            v_w_u_complement["Next State"],
            v_w_u_complement["Current Values"],
        )
        print(f"Probabilidad del complemento del subconjunto: {p_v_set_complement}")
        
        # 4. Producto tensorial entre las probabilidades
        tensor_product_vm = np.kron(p_v_set, p_v)
        tensor_product_vm_complemento = np.kron(p_v_set_complement, p_v)
        
        # 5. Cálculo de EMD (Earth Mover's Distance)
        emd_value = EMD(tensor_product_vm, tensor_product_vm_complemento)
        print(f"[calculate_g] EMD (g): {emd_value}")
        
        return emd_value

    def encontrar_particiones_optimas(self, V: OrderedSet, particiones=[]):
        if len(V) < 2:
            raise ValueError("El conjunto V debe tener al menos dos elementos.")

        n = len(V)
        W = [OrderedSet() for _ in range(n + 1)]
        W[1] = OrderedSet([list(V)[0]])  # Explicitar la selección del primer elemento

        for i in range(2, n + 1):
            vi_min = None
            min_valor = float("inf")
            for vi in V:
                if vi not in W[i - 1]:
                    valor = self.calculate_g(
                        tuple(W[i - 1] | OrderedSet([vi])), tuple(V)
                    ) - self.calculate_g((vi,), tuple(V))
                    if valor < min_valor:
                        min_valor = valor
                        vi_min = vi
            W[i] = W[i - 1] | OrderedSet([vi_min])

        par_candidato = (list(W[n - 1])[-1], list(W[n])[-1])
        particion1 = OrderedSet([par_candidato[1]])
        particion2 = OrderedSet(V) - particion1
        particiones.append(
            (particion1, particion2, self.calculate_g(tuple(particion1), tuple(V)))
        )

        if len(V) > 2:
            u = tuple(par_candidato)
            nuevo_V = OrderedSet([x for x in V if x not in par_candidato] + [u])
            print("LLEGUE A LA RECURSION")
            particiones.append(
                (
                    OrderedSet([u]),
                    OrderedSet(nuevo_V) - OrderedSet([u]),
                    self.calculate_g((u,), tuple(V)),
                )
            )
            self.encontrar_particiones_optimas(nuevo_V, particiones)

        particion: List[PartitionModel] = [
            PartitionModel(p[0], p[1], p[2]) for p in particiones
        ]

        return particion

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
