import re
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict


# Definir las opciones del sistema
class OpcionSistema(Enum):
    V = "V"
    VM = "VM"
    VM_COMPLEMENTO = "VM complemento"


@dataclass
class EstadoRespuesta:
    """Clase que representa la respuesta de la estrategia para un estado."""

    current_state: str = ""  # Cadena vacía por defecto
    next_state: str = ""  # Cadena vacía por defecto
    current_values: List[int] = field(default_factory=list)  # Lista vacía por defecto


class Strategy:
    """Interfaz base para todas las estrategias."""

    def execute(
        self, handler: "StateHandler", VM: Optional[List[str]] = None
    ) -> EstadoRespuesta:
        pass


class VStrategy(Strategy):
    """Estrategia para manejar la opción 'V'."""

    def execute(
        self, handler: "StateHandler", VM: Optional[List[str]] = None
    ) -> EstadoRespuesta:
        return EstadoRespuesta(
            current_state=handler._formatear_estado(handler.V_t, "T"),
            next_state=handler._formatear_estado(handler.V_t1, "T+1"),
            current_values=[handler.current_values.get(e, 0) for e in handler.V_t],
        )


class VMStrategy(Strategy):
    """Estrategia para manejar la opción 'VM'."""

    def execute(
        self, handler: "StateHandler", VM: Optional[List[str]] = None
    ) -> EstadoRespuesta:
        VM = [v.upper() for v in (VM or [])]
        VM_t = [e for e in VM if "T" in e and "+1" not in e]
        VM_t1 = [e for e in VM if "T+1" in e]
        return EstadoRespuesta(
            current_state=handler._formatear_estado(VM_t, "T"),
            next_state=handler._formatear_estado(VM_t1, "T+1"),
            current_values=[handler.current_values.get(e, 0) for e in VM_t],
        )


class VMComplementoStrategy(Strategy):
    """Estrategia para manejar la opción 'VM complemento'."""

    def execute(
        self, handler: "StateHandler", VM: Optional[List[str]] = None
    ) -> EstadoRespuesta:
        VM = [v.upper() for v in (VM or [])]
        complemento_t = [e for e in handler.V_t if e not in VM]
        complemento_t1 = [e for e in handler.V_t1 if e not in VM]
        return EstadoRespuesta(
            current_state=handler._formatear_estado(complemento_t, "T"),
            next_state=handler._formatear_estado(complemento_t1, "T+1"),
            current_values=[handler.current_values.get(e, 0) for e in complemento_t],
        )


@dataclass
class StateHandler:
    """Clase principal para manejar estados y estrategias."""

    V: List[str] = field(
        default_factory=lambda: [
            "AT",
            "BT",
            "CT",
            "DT",
            "ET",
            "AT+1",
            "BT+1",
            "CT+1",
            "DT+1",
            "ET+1",
        ]
    )
    current_values: Dict[str, int] = field(
        default_factory=lambda: {"AT": 1, "BT": 0, "CT": 0, "DT": 1, "ET": 0}
    )

    def __post_init__(self):
        self.V = [v.upper() for v in self.V]
        self.V_t = [e for e in self.V if "T" in e and "+1" not in e]
        self.V_t1 = [e for e in self.V if "T+1" in e]

        if len(self.current_values) != len(self.V_t):
            raise ValueError(
                f"La longitud de current_values ({len(self.current_values)}) "
                f"no coincide con el número de elementos en V_t ({len(self.V_t)})."
            )

    def generate_system(
        self, opcion: OpcionSistema, VM: Optional[List[str]] = None
    ) -> EstadoRespuesta:
        """Genera el sistema basado en la opción dada utilizando el patrón Strategy."""
        strategies = {
            OpcionSistema.V: VStrategy(),
            OpcionSistema.VM: VMStrategy(),
            OpcionSistema.VM_COMPLEMENTO: VMComplementoStrategy(),
        }
        strategy = strategies.get(opcion)
        if not strategy:
            raise ValueError("Opción no válida. Use 'V', 'VM', o 'VM complemento'.")
        return strategy.execute(self, VM)

    def _formatear_estado(self, estado: List[str], sufijo: str) -> str:
        """Formatea el estado en una cadena ordenada eliminando sufijos."""
        return "".join(
            elem.replace(sufijo, "")
            for elem in sorted(estado, key=self._ordenar_elementos)
            if sufijo in elem
        )

    @staticmethod
    def _ordenar_elementos(elem: str) -> int:
        """Ordena los elementos en base al prefijo y la presencia de '+1'."""
        base, *rest = elem.split("+")
        # Si tiene '+1', asignamos un mayor peso para que venga después de 'T'
        return (ord(base[0]) * 10) + (1 if "+1" in elem else 0)


# # Ejemplo de uso con valores personalizados
# V_custom = ["XT", "YT", "ZT", "XT+1", "YT+1", "ZT+1"]
# current_values_custom = {"XT": 1, "YT": 0, "ZT": 1}

# # Inicializar el StateHandler con los valores personalizados
# handler_custom = StateHandler(V=V_custom, current_values=current_values_custom)

# # Generar el sistema para la opción 'V'
# resultado_v = handler_custom.generate_system(opcion=OpcionSistema.V)
# print("Resultado para opción 'V':")
# print(f"Current State: {resultado_v.current_state}")
# print(f"Next State: {resultado_v.next_state}")
# print(f"Current Values: {resultado_v.current_values}")

# # Generar el sistema para la opción 'VM' con un subconjunto
# VM = [""]
# resultado_vm = handler_custom.generate_system(opcion=OpcionSistema.VM, VM=VM)
# print("\nResultado para opción 'VM':")
# print(f"Current State: {resultado_vm.current_state}")
# print(f"Next State: {resultado_vm.next_state}")
# print(f"Current Values: {resultado_vm.current_values}")

# resultado_vm_complemento = handler_custom.generate_system(opcion=OpcionSistema.VM_COMPLEMENTO, VM=VM)
# print("\nResultado para opción 'VM complemento':")
# print(f"Current State: {resultado_vm_complemento.current_state}")
# print(f"Next State: {resultado_vm_complemento.next_state}")
# print(f"Current Values: {resultado_vm_complemento.current_values}")
