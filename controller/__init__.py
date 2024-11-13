from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ProbabilidadM:
    V: List[str]
    current_values: List[int]
    V_t: List[str] = field(init=False)
    V_t1: List[str] = field(init=False)

    def __post_init__(self):
        # Convertir todos los elementos de V a mayúsculas y asignar valores iniciales solo para elementos en t
        self.V = [v.upper() for v in self.V]
        self.current_values = {
            v.upper(): val
            for v, val in zip(self.V, self.current_values)
            if "+1" not in v
        }

        # Conjunto de elementos en t (sin '+1') y en t+1 (con '+1')
        self.V_t = [e for e in self.V if "T" in e and "+1" not in e]
        self.V_t1 = [e for e in self.V if "T+1" in e]

    def inicializar_sistema(self, opcion="V", VM=None):
        """
        Inicializa el sistema según la opción seleccionada: 'V', 'M', 'M complemento', 'VM', o 'VM complemento',
        con Current State y Next State concatenados de forma limpia.
        """

        # Función para limpiar y concatenar los elementos
        def formatear_estado(estado_dict, sufijo):
            """
            Esta función limpia los sufijos 'T' o 'T+1' de los elementos y ordena
            de acuerdo con la secuencia en V.
            """
            # Eliminar los sufijos 'T' y 'T+1' y ordenar los elementos en función de V
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
            # Formatear el estado en base a la opción 'V'
            current_state = formatear_estado(self.V_t, "T")
            next_state = formatear_estado(self.V_t1, "T+1")
            current_values = list(
                self.current_values.values()
            )  # Devolver solo los valores en lista
        elif opcion == "VM":
            if VM is None:
                VM = []
            VM = [v.upper() for v in VM]
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
            VM = [v.upper() for v in VM]
            VM_t = [e for e in VM if "T" in e and "+1" not in e]
            VM_t1 = [e for e in VM if "T+1" in e]

            complemento_t = [e for e in self.V_t if e not in VM_t]
            complemento_t1 = [e for e in self.V_t1 if e not in VM_t1]

            current_state = formatear_estado(complemento_t, "T")
            next_state = formatear_estado(complemento_t1, "T+1")
            current_values = [self.current_values.get(e, 0) for e in complemento_t]
        else:
            raise ValueError("Opción no válida. Use 'V', 'VM', o 'VM complemento'.")

        # Inicializar el sistema en formato limpio y ordenado
        sistema = {
            "Current State": current_state,
            "Next State": next_state,
            "Current Values": current_values,  # Devolver como lista
        }

        return sistema

    def ordenar_elementos(self, elem):
        base_elem = elem.replace("T", "").replace("T+1", "")

        # Asegurar que el elemento es una letra y obtener su valor ASCII
        if base_elem.isalpha():
            return ord(base_elem) - ord("A")
        return float("inf")  # Si no está en V, lo coloca al final


# Definir el universo V y valores iniciales
V = ["at", "bt", "ct", "dt", "et", "at+1", "bt+1", "ct+1", "dt+1", "et+1"]
current_values = [
    1,
    0,
    0,
    1,
    0,
]  # Estado inicial para elementos en t: at=1, bt=0, ct=0, dt=1, et=0

# Crear la instancia de ProbabilidadM
probabilidad = ProbabilidadM(V, current_values)

# Inicializar el sistema en base a V
sistema_V = probabilidad.inicializar_sistema(opcion="V")
print("Sistema basado en V:", sistema_V)

# Inicializar el sistema en base a VM
VM = ["bt", "at+1"]
sistema_VM = probabilidad.inicializar_sistema(opcion="VM", VM=VM)
print("Sistema basado en VM:", sistema_VM)

# Inicializar el sistema en base a VM complemento
sistema_VM_complemento = probabilidad.inicializar_sistema(
    opcion="VM complemento", VM=VM
)
print("Sistema basado en VM complemento:", sistema_VM_complemento)

# Otros casos de inicialización en base a VM y sus complementos
VM = ["at", "bt"]
sistema_VM = probabilidad.inicializar_sistema(opcion="VM", VM=VM)
print("Sistema basado en VM:", sistema_VM)
sistema_VM_complemento = probabilidad.inicializar_sistema(
    opcion="VM complemento", VM=VM
)
print("Sistema basado en VM complemento:", sistema_VM_complemento)

VM = ["at", "bt+1"]
sistema_VM = probabilidad.inicializar_sistema(opcion="VM", VM=VM)
print("Sistema basado en VM:", sistema_VM)
sistema_VM_complemento = probabilidad.inicializar_sistema(
    opcion="VM complemento", VM=VM
)
print("Sistema basado en VM complemento:", sistema_VM_complemento)

VM = ["at+1", "bt+1"]
sistema_VM = probabilidad.inicializar_sistema(opcion="VM", VM=VM)
print("Sistema basado en VM:", sistema_VM)
sistema_VM_complemento = probabilidad.inicializar_sistema(
    opcion="VM complemento", VM=VM
)
print("Sistema basado en VM complemento:", sistema_VM_complemento)

VM = []
sistema_VM = probabilidad.inicializar_sistema(opcion="VM", VM=VM)
print("Sistema basado en VM:", sistema_VM)
sistema_VM_complemento = probabilidad.inicializar_sistema(
    opcion="VM complemento", VM=VM
)
print("Sistema basado en VM complemento:", sistema_VM_complemento)
