import numpy as np
from scipy.stats import wasserstein_distance as EMD
from ordered_set import OrderedSet
from dataclasses import dataclass, field
from typing import List, Dict, OrderedDict, Tuple


# Clase ProbabilidadM
@dataclass
class ProbabilidadM:
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

    def inicializar_sistema(self, opcion="V", VM=None):
        def formatear_estado(estado_dict, sufijo):
            return "".join(
                [
                    elem.replace(sufijo, "")
                    for elem in sorted(
                        estado_dict, key=lambda x: self.ordenar_elementos(x)
                    )
                    if sufijo in elem
                ]
            )

        if opcion == "V":
            current_state = formatear_estado(self.V_t, "T")
            next_state = formatear_estado(self.V_t1, "T+1")
            current_values = list(self.current_values.values())

        elif opcion == "VM":
            if VM is None:
                VM = []
            VM = [v[0].upper() if isinstance(v, tuple) else v.upper() for v in VM]
            current_state = formatear_estado(
                [e for e in VM if "T" in e and "+1" not in e], "T"
            )
            next_state = formatear_estado([e for e in VM if "T+1" in e], "T+1")
            current_values = [
                self.current_values.get(e, 0) for e in VM if "T" in e and "+1" not in e
            ]

        elif opcion == "VM complemento":
            if VM is None:
                VM = []
            VM = [v[0].upper() if isinstance(v, tuple) else v.upper() for v in VM]
            VM_t = [e for e in VM if "T" in e and "+1" not in e]
            VM_t1 = [e for e in VM if "T+1" in e]
            complemento_t = [e for e in self.V_t if e not in VM_t]
            complemento_t1 = [e for e in self.V_t1 if e not in VM_t1]
            current_state = formatear_estado(complemento_t, "T")
            next_state = formatear_estado(complemento_t1, "T+1")
            current_values = [self.current_values.get(e, 0) for e in complemento_t]
        else:
            raise ValueError("Opción no válida. Use 'V', 'VM', o 'VM complemento'.")

        return {
            "Current State": current_state,
            "Next State": next_state,
            "Current Values": current_values,
        }

    def ordenar_elementos(self, elem):
        base_elem = elem.replace("T", "").replace("T+1", "")
        if base_elem.isalpha():
            return ord(base_elem) - ord("A")
        return float("inf")


# Clase StateProbabilityCalculator
@dataclass
class StateProbabilityCalculator:
    states: List[List[int]]
    probabilities: np.ndarray
    var_names: List[str]

    def get_state_indices(
        self, state_repr: str, values: List[int] = None
    ) -> Tuple[Dict[Tuple[int, ...], List[int]], int]:
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
        matrix = np.zeros((len(self.probabilities), len(groups)))
        for col, indices in enumerate(groups.values()):
            matrix[:, col] = np.sum(self.probabilities[:, indices], axis=1)
        return matrix

    def compute_state_matrix(
        self, cs_groups: Dict[Tuple[int, ...], List[int]], transition_matrix: np.ndarray
    ) -> np.ndarray:
        matrix = np.zeros((len(cs_groups), transition_matrix.shape[1]))
        for row, indices in enumerate(cs_groups.values()):
            matrix[row, :] = np.sum(transition_matrix[indices, :], axis=0) / len(
                indices
            )
        return matrix

    def calculate_probabilities(
        self, current_state: str, next_state: str, current_values: List[int]
    ) -> np.ndarray:
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

    def calculate_g(self, subset: OrderedSet, total_set: OrderedSet):
        W_with_u = subset
        print(W_with_u, "|", " Complete->", total_set)
        v = self.candidate.inicializar_sistema(opcion="V")
        p_v = self.complete.calculate_probabilities(
            v["Current State"], v["Next State"], v["Current Values"]
        )
        v_w_u = self.candidate.inicializar_sistema(opcion="VM", VM=W_with_u)
        p_v_set = self.complete.calculate_probabilities(
            v_w_u["Current State"], v_w_u["Next State"], v_w_u["Current Values"]
        )
        v_w_u_complement = self.candidate.inicializar_sistema(
            opcion="VM complemento", VM=W_with_u
        )
        p_v_set_complement = self.complete.calculate_probabilities(
            v_w_u_complement["Current State"],
            v_w_u_complement["Next State"],
            v_w_u_complement["Current Values"],
        )
        tensor_product_vm = np.kron(p_v_set, p_v)
        tensor_product_vm_complemento = np.kron(p_v_set_complement, p_v)

        # print(v, "\n", v_w_u, "\n", v_w_u_complement)
        # print("#" * 100)
        return EMD(tensor_product_vm, tensor_product_vm_complemento)

    def encontrar_particiones_optimas(self, V: OrderedSet, ciclo=1, particiones=[]):
        n = len(V)
        W = [OrderedSet() for _ in range(n + 1)]
        W[1] = OrderedSet([V[0]])

        for i in range(2, n + 1):
            vi_min = None
            min_valor = float("inf")
            for vi in V:
                if vi not in W[i - 1]:
                    valor = self.calculate_g(
                        W[i - 1] | OrderedSet([vi]), V
                    ) - self.calculate_g(OrderedSet([vi]), V)
                    if valor < min_valor:
                        min_valor = valor
                        vi_min = vi
            W[i] = W[i - 1] | OrderedSet([vi_min])

        par_candidato = (list(W[n - 1])[-1], list(W[n])[-1])
        particion1 = OrderedSet([par_candidato[1]])
        particion2 = OrderedSet(V) - particion1
        particiones.append(
            (f"Ciclo {ciclo}", particion1, particion2, self.calculate_g(particion1, V))
        )

        if len(V) > 2:
            u = tuple(par_candidato)
            nuevo_V = OrderedSet([x for x in V if x not in par_candidato] + [u])
            print(nuevo_V)
            particiones.append(
                (
                    f"Ciclo {ciclo}",
                    OrderedSet([u]),
                    OrderedSet(nuevo_V) - OrderedSet([u]),
                    self.calculate_g(OrderedSet([u]), V),
                )
            )
            self.encontrar_particiones_optimas(nuevo_V, ciclo + 1, particiones)

        return particiones


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

# Imprimir los resultados
for ciclo, particion1, particion2, emd_valor in particiones_optimas:
    print(
        f"{ciclo} - Partición 1: {particion1}, Partición 2: {particion2}, EMD: {emd_valor}"
    )
