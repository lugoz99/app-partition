from calendar import c
from dataclasses import dataclass

from fastapi.background import P


@dataclass
class ProbabilidadM:
    V: list
    current_values: list

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

    def obtener_PM(self, conjunto_t):
        """
        Dado un conjunto de elementos en t, devuelve la estructura de P(M),
        basada en los valores iniciales asignados.
        """
        derecha = [e.upper() for e in conjunto_t]  # Elementos t proporcionados
        izquierda = [
            e for e in self.V if e not in derecha
        ]  # Elementos que no están en derecha

        # Asignar valores iniciales de acuerdo al conjunto t
        self.PM_derecha = {e: self.current_values.get(e, 0) for e in derecha}

        # Retorna la estructura de P(M)
        return {
            "P(M)": f"P({izquierda} | {self.PM_derecha})",
            "t_elements": self.PM_derecha,
            "t1_elements": izquierda,
        }

    def obtener_complemento_de_PM(self):
        """
        Calcula P(M complemento) a partir de P(M) ya obtenida, usando el complemento
        de los valores iniciales de t.
        """
        # Primero, calculamos P(M) usando los elementos en t que nos dan
        pm = self.obtener_PM(self.V_t)
        print("pm", pm)
        pm_izquierda = pm["t1_elements"]
        pm_derecha = pm["t_elements"]  # Elementos en P(M) a la derecha (en t)
        print("pm_izquierda", pm_izquierda)
        complemento_izquierda = [izq for izq in self.V_t1 if izq not in pm_izquierda]
        complemento_derecha = [der for der in self.V_t if der not in pm_derecha.keys()]

        PM_complemento_derecha = {
            e: self.current_values.get(e, 0) for e in complemento_derecha
        }
        PM_complemento_izquierda = {
            e: self.current_values.get(e, 0) for e in complemento_izquierda
        }

        return {
            "P(M complemento)": f"P({complemento_izquierda} | {complemento_derecha})",
            "t_elements": PM_complemento_derecha,
            "t1_elements": PM_complemento_izquierda,
        }

    def inicializar_sistema(self, opcion="V"):
        """
        Inicializa el sistema según la opción seleccionada: 'V', 'M', o 'M complemento',
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
        elif opcion == "M":
            pm = self.obtener_PM(self.V_t)
            # Formatear el estado en base a P(M)
            current_state = formatear_estado(pm["t_elements"], "T")
            next_state = formatear_estado(pm["t1_elements"], "T+1")
            current_values = list(
                pm["t_elements"].values()
            )  # Devolver solo los valores en lista
        elif opcion == "M complemento":
            pm_complemento = self.obtener_complemento_de_PM()
            # Formatear el estado en base a P(M complemento)
            current_state = formatear_estado(pm_complemento["t_elements"], "T")
            next_state = formatear_estado(pm_complemento["t1_elements"], "T+1")
            current_values = list(
                pm_complemento["t_elements"].values()
            )  # Devolver solo los valores en lista
        else:
            raise ValueError("Opción no válida. Use 'V', 'M', o 'M complemento'.")

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
V = ["at", "bt", "ct", "at+1", "bt+1"]
current_values = [1, 0, 0]  # Estado inicial para elementos en t: at=1, bt=0

# Crear la instancia de ProbabilidadM
probabilidad = ProbabilidadM(V, current_values)

# Inicializar el sistema en base a V
sistema_V = probabilidad.inicializar_sistema(opcion="V")
print("Sistema basado en V:", sistema_V)

# Inicializar el sistema en base a P(M)
sistema_M = probabilidad.inicializar_sistema(opcion="M")
print("Sistema basado en P(M):", sistema_M)

# Inicializar el sistema en base a P(M complemento)
sistema_M_complemento = probabilidad.inicializar_sistema(opcion="M complemento")
print("Sistema basado en P(M complemento):", sistema_M_complemento)
