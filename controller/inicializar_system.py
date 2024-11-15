# Importar las librerías necesarias para trabajar con dataclasses y tipos de datos específicos
from dataclasses import dataclass, field
from typing import List


# Definir la clase ProbabilidadM con decorador @dataclass para facilitar la inicialización
@dataclass
class ProbabilidadM:
    # Definir los atributos iniciales de la clase
    V: List[str]  # Universo de elementos (t y t+1) en formato de cadena
    current_values: List[int]  # Valores actuales de los elementos en t
    V_t: List[str] = field(
        init=False
    )  # Elementos en t (se inicializan en __post_init__)
    V_t1: List[str] = field(
        init=False
    )  # Elementos en t+1 (se inicializan en __post_init__)

    def __post_init__(self):
        # Convertir todos los elementos en V a mayúsculas para estandarizar la notación
        self.V = [v.upper() for v in self.V]

        # Asignar los valores actuales solo a los elementos en t, ignorando los elementos en t+1
        self.current_values = {
            v.upper(): val
            for v, val in zip(self.V, self.current_values)
            if "+1" not in v
        }

        # Crear subconjuntos de V: uno para t (sin '+1') y otro para t+1 (con '+1')
        self.V_t = [e for e in self.V if "T" in e and "+1" not in e]
        self.V_t1 = [e for e in self.V if "T+1" in e]

    def inicializar_sistema(self, opcion="V", VM=None):
        """
        Inicializa el sistema basado en la opción seleccionada:
        'V' - para el universo completo,
        'VM' - para un subconjunto M específico,
        'VM complemento' - para el complemento de M.
        Se formatean los estados actuales y futuros en cadenas ordenadas y limpias.
        """

        # Función auxiliar para formatear los estados
        def formatear_estado(estado_dict, sufijo):
            """
            Limpia los sufijos 'T' y 'T+1' de los elementos, los ordena según el orden en V.
            """
            # Remover sufijos y ordenar elementos de acuerdo al universo V
            return "".join(
                [
                    elem.replace(sufijo, "")
                    for elem in sorted(
                        estado_dict, key=lambda x: self.ordenar_elementos(x)
                    )
                    if sufijo in elem
                ]
            )

        # Manejo de cada opción de inicialización
        if opcion == "V":
            # Configuración para el universo completo
            current_state = formatear_estado(self.V_t, "T")  # Estado actual
            next_state = formatear_estado(self.V_t1, "T+1")  # Estado futuro
            current_values = list(
                self.current_values.values()
            )  # Valores actuales en una lista

        elif opcion == "VM":
            # Configuración para un subconjunto específico, M
            if VM is None:
                VM = []
            VM = [v.upper() for v in VM]  # Convertir VM a mayúsculas
            current_state = formatear_estado(
                [e for e in VM if "T" in e and "+1" not in e], "T"
            )
            next_state = formatear_estado([e for e in VM if "T+1" in e], "T+1")
            current_values = [
                self.current_values.get(e, 0) for e in VM if "T" in e and "+1" not in e
            ]

        elif opcion == "VM complemento":
            # Configuración para el complemento del subconjunto M
            if VM is None:
                VM = []
            VM = [v.upper() for v in VM]
            VM_t = [e for e in VM if "T" in e and "+1" not in e]
            VM_t1 = [e for e in VM if "T+1" in e]

            # Complemento: elementos en V_t y V_t1 que no están en VM_t y VM_t1
            complemento_t = [e for e in self.V_t if e not in VM_t]
            complemento_t1 = [e for e in self.V_t1 if e not in VM_t1]

            current_state = formatear_estado(complemento_t, "T")
            next_state = formatear_estado(complemento_t1, "T+1")
            current_values = [self.current_values.get(e, 0) for e in complemento_t]
        else:
            raise ValueError("Opción no válida. Use 'V', 'VM', o 'VM complemento'.")

        # Devolver el sistema inicializado en formato limpio y ordenado
        sistema = {
            "Current State": current_state,
            "Next State": next_state,
            "Current Values": current_values,  # Valores como lista para simplicidad
        }

        return sistema

    def ordenar_elementos(self, elem):
        # Eliminar sufijos 'T' y 'T+1' para ordenar solo por el elemento base
        base_elem = elem.replace("T", "").replace("T+1", "")

        # Si es una letra, devuelve su valor ASCII menos 'A', sino lo coloca al final
        if base_elem.isalpha():
            return ord(base_elem) - ord("A")
        return float("inf")


# Definir el universo V y valores iniciales
V = ["at", "bt", "at+1", "bt+1"]
current_values = [1, 0, 0]  # Estado inicial de los elementos en t: at=1, bt=0

# Crear una instancia de la clase ProbabilidadM
probabilidad = ProbabilidadM(V, current_values)

# Inicializar el sistema en base a V (universo completo)
sistema_V = probabilidad.inicializar_sistema(opcion="V")
print("Sistema basado en V:", sistema_V)

# Inicializar el sistema en base a un subconjunto VM
VM = ["at"]
sistema_VM = probabilidad.inicializar_sistema(opcion="VM", VM=VM)
print("Sistema basado en VM:", sistema_VM)

# Inicializar el sistema en base al complemento de VM
sistema_VM_complemento = probabilidad.inicializar_sistema(
    opcion="VM complemento", VM=VM
)
print("Sistema basado en VM complemento:", sistema_VM_complemento)
