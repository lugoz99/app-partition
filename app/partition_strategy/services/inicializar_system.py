from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
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
        processed_vm = []
        print("Tuples")
        print(VM)
        for v in VM:
            if isinstance(v, tuple):
                processed_vm.append(v[0].upper())  # Extraer y convertir la primera letra
            elif isinstance(v, str):
                processed_vm.append(v.upper())  # Convertir cadenas a mayúsculas
            else:
                raise ValueError("VM contiene un elemento no válido.")

        # Lógica para cada opción
        if opcion == OpcionSistema.V:
            return self._crear_estado(self.V_t, self.V_t1, self.current_values.values())

        elif opcion == OpcionSistema.VM:
            return self._crear_estado(
                [e for e in processed_vm if "T" in e and "+1" not in e],
                [e for e in processed_vm if "T+1" in e],
                [
                    self.current_values.get(e, 0)
                    for e in processed_vm
                    if "T" in e and "+1" not in e
                ],
            )

        elif opcion == OpcionSistema.VM_COMPLEMENTO:
            complemento_t, complemento_t1 = self._calcular_complemento(processed_vm)
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
