import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from enum import Enum
from typing import List, Dict, Tuple, OrderedDict
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import wasserstein_distance as EMD
from ordered_set import OrderedSet
from model.partition import PartitionModel

# Agregar el directorio raíz del proyecto al PYTHONPATH


# Enumeración para opciones de inicialización del sistema
class OpcionSistema(Enum):
    V = "V"
    VM = "VM"
    VM_COMPLEMENTO = "VM complemento"


@dataclass
class ProbabilidadM:
    """Clase para manejar las probabilidades de los sistemas."""

    V: List[str]
    current_values: List[int]
    V_t: List[str] = field(init=False)
    V_t1: List[str] = field(init=False)

    def __post_init__(self):
        self.V = [v.upper() for v in self.V]
        self.current_values = {
            v.upper(): val
            for v, val in zip(self.V, self.current_values)
            if "+1" not in v
        }
        self.V_t = [e for e in self.V if "T" in e and "+1" not in e]
        self.V_t1 = [e for e in self.V if "T+1" in e]

    def inicializar_sistema(self, opcion: OpcionSistema, VM=None) -> Dict:
        """Inicializa el sistema con la configuración dada."""
        if VM is None:
            VM = []

        # Procesar elementos de VM en función de su tipo
        VM = [v[0].upper() if isinstance(v, tuple) else v.upper() for v in VM]

        # Lógica para cada opción
        if opcion == OpcionSistema.V:
            return self._crear_estado(self.V_t, self.V_t1, self.current_values.values())

        elif opcion == OpcionSistema.VM:
            return self._crear_estado(
                [e for e in VM if "T" in e and "+1" not in e],
                [e for e in VM if "T+1" in e],
                [
                    self.current_values.get(e, 0)
                    for e in VM
                    if "T" in e and "+1" not in e
                ],
            )

        elif opcion == OpcionSistema.VM_COMPLEMENTO:
            complemento_t, complemento_t1 = self._calcular_complemento(VM)
            return self._crear_estado(
                complemento_t,
                complemento_t1,
                [self.current_values.get(e, 0) for e in complemento_t],
            )

        else:
            raise ValueError("Opción no válida. Use OpcionSistema.")

    def _crear_estado(self, current_t, next_t, current_values) -> Dict:
        """Crea el estado formateado con la información proporcionada."""
        return {
            "Current State": self._formatear_estado(current_t, "T"),
            "Next State": self._formatear_estado(next_t, "T+1"),
            "Current Values": list(current_values),
        }

    def _calcular_complemento(self, VM: List[str]) -> Tuple[List[str], List[str]]:
        """Calcula el complemento de los conjuntos actuales."""
        VM_t = [e for e in VM if "T" in e and "+1" not in e]
        VM_t1 = [e for e in VM if "T+1" in e]
        complemento_t = [e for e in self.V_t if e not in VM_t]
        complemento_t1 = [e for e in self.V_t1 if e not in VM_t1]
        return complemento_t, complemento_t1

    def _formatear_estado(self, estado_dict: List[str], sufijo: str) -> str:
        """Formatea el estado en una cadena ordenada."""
        return "".join(
            elem.replace(sufijo, "")
            for elem in sorted(estado_dict, key=self.ordenar_elementos)
            if sufijo in elem
        )

    def ordenar_elementos(self, elem: str) -> float:
        """Define el orden de los elementos basado en su sufijo."""
        base_elem = elem.replace("T", "").replace("T+1", "")
        if base_elem.isalpha():
            return ord(base_elem) - ord("A")
        return float("inf")


@dataclass
class StateProbabilityCalculator:
    """Clase para calcular probabilidades entre estados."""

    states: List[List[int]]
    probabilities: np.ndarray
    var_names: List[str]

    def get_state_indices(
        self, state_repr: str, values: List[int] = None
    ) -> Tuple[Dict[Tuple[int, ...], List[int]], int]:
        """Obtiene los índices de estados compatibles con la representación dada."""
        var_mask = [c in state_repr for c in self.var_names]
        positions = [i for i, mask in enumerate(var_mask) if mask]
        state_groups = {}
        for i, state in enumerate(self.states):
            key = tuple(state[p] for p in positions)
            state_groups.setdefault(key, []).append(i)
        state_value = 0
        if values:
            value_map = {self.var_names[i]: v for i, v in zip(positions, values)}
            binary = "".join(str(value_map[c]) for c in state_repr)
            state_value = int(binary, 2)
        return state_groups, state_value

    def compute_transition_matrix(
        self, groups: Dict[Tuple[int, ...], List[int]]
    ) -> np.ndarray:
        """Calcula la matriz de transición para los grupos dados."""
        return np.array(
            [
                np.sum(self.probabilities[:, indices], axis=1)
                for indices in groups.values()
            ]
        ).T

    def compute_state_matrix(
        self, cs_groups: Dict[Tuple[int, ...], List[int]], transition_matrix: np.ndarray
    ) -> np.ndarray:
        """Calcula la matriz de estados basada en la matriz de transición."""
        matrix = np.zeros((len(cs_groups), transition_matrix.shape[1]))
        for row, indices in enumerate(cs_groups.values()):
            matrix[row, :] = np.sum(transition_matrix[indices, :], axis=0) / len(
                indices
            )
        return matrix

    def calculate_probabilities(
        self, current_state: str, next_state: str, current_values: List[int]
    ) -> np.ndarray:
        """Calcula las probabilidades de transición entre estados."""
        cs_groups, cs_index = self.get_state_indices(current_state, current_values)
        if len(next_state) > 1:
            result = None
            for var in next_state:
                ns_groups, _ = self.get_state_indices(var)
                trans_matrix = self.compute_transition_matrix(ns_groups)
                state_matrix = self.compute_state_matrix(cs_groups, trans_matrix)
                probs = state_matrix[cs_index]
                result = np.kron(result, probs) if result is not None else probs
            return result
        else:
            ns_groups, _ = self.get_state_indices(next_state)
            trans_matrix = self.compute_transition_matrix(ns_groups)
            state_matrix = self.compute_state_matrix(cs_groups, trans_matrix)
            return state_matrix[cs_index]


# Continuación en el siguiente bloque...


# Clase RecursiveCandidateSelection con encontrar_particiones_optimas integrado
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
        self.candidate = ProbabilidadM(
            V=list(self.v.keys()), current_values=self.current_values
        )
        self.complete = StateProbabilityCalculator(
            self.states, self.probabilities, self.var_names
        )

    def calcular_probabilidad_fusionada(self, fusion_node):
        """
        Calcula la probabilidad para un nodo fusionado o simple.
        """
        if isinstance(fusion_node, tuple):  # Nodo fusionado
            print(f"[Fusionado] Procesando nodo compuesto: {fusion_node}")
            probs = None
            for sub_node in fusion_node:
                sub_probs = self.calcular_probabilidad_fusionada(sub_node)
                probs = np.kron(probs, sub_probs) if probs is not None else sub_probs
            return probs
        else:  # Nodo simple
            # Usar la enumeración OpcionSistema.VM en lugar de la cadena literal
            v = self.candidate.inicializar_sistema(
                opcion=OpcionSistema.VM, VM=[fusion_node]
            )
            simple_probs = self.complete.calculate_probabilities(
                v["Current State"], v["Next State"], v["Current Values"]
            )
            return simple_probs

    def calculate_g(self, subset: OrderedSet, total_set: OrderedSet) -> float:
        """Calcula la función g basada en la distancia EMD entre probabilidades."""
        print("==== Inicio de cálculo de g ====")

        # Probabilidad del conjunto completo V
        v = self.candidate.inicializar_sistema(opcion=OpcionSistema.V)
        p_v = self.complete.calculate_probabilities(
            v["Current State"], v["Next State"], v["Current Values"]
        )

        # Probabilidad del subconjunto actual
        if any(isinstance(node, tuple) for node in subset):
            p_v_set = self.calcular_probabilidad_fusionada(tuple(subset))
        else:
            v_w_u = self.candidate.inicializar_sistema(
                opcion=OpcionSistema.VM, VM=list(subset)
            )
            p_v_set = self.complete.calculate_probabilities(
                v_w_u["Current State"], v_w_u["Next State"], v_w_u["Current Values"]
            )

        # Probabilidad del complemento del subconjunto
        v_w_u_complement = self.candidate.inicializar_sistema(
            opcion=OpcionSistema.VM_COMPLEMENTO, VM=list(subset)
        )

        p_v_set_complement = self.complete.calculate_probabilities(
            v_w_u_complement["Current State"],
            v_w_u_complement["Next State"],
            v_w_u_complement["Current Values"],
        )

        # Producto tensorial y cálculo de EMD
        tensor_product_vm = np.kron(p_v_set, p_v)
        tensor_product_vm_complemento = np.kron(p_v_set_complement, p_v)

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
        return min(particiones_dict, key=lambda x: x["emd_valor"])


# Definir los estados posibles
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

# Definir la matriz de probabilidades
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

# Definir los nombres de las variables
var_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Definir el conjunto de variables y sus valores actuales
v = OrderedSet(["at", "bt", "at+1", "bt+1"])
current_values = [1, 0, 0]
z: Dict[str, List[int]] = {}
# Llenar el diccionario con listas de enteros
ordered_set_d = OrderedDict.fromkeys(["at", "bt", "at+1", "bt+1"])

for i, key in enumerate(ordered_set_d.keys(), start=1):
    z[key] = [i]
# Ahora v es {"at": [1], "bt": [2], "at+1": [3], "bt+1": [4]}
print(z)
# Crear una instancia de RecursiveCandidateSelection
selector = RecursiveCandidateSelection(
    z,
    current_values=current_values,
    states=states,
    probabilities=probabilities,
    var_names=var_names,
)

# Encontrar las particiones óptimas
particiones_optimas = selector.encontrar_particiones_optimas(v)

minimo_particion = selector.encontrar_particion_con_menor_emd(particiones_optimas)
print(minimo_particion)
