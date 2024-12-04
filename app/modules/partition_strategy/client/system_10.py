import re
from collections import OrderedDict
from ordered_set import OrderedSet

# Estado del sistema
estado_sistema = {
    "Estado Inicial": "100000000000000",
    "Sistema": "ABCDEFGHIJKLMNO",
    "Sistemas Candidatos": {
        "ABCDEFGHIJKLMNO": [
            {"Subsistema": "ABCDEFGHIJKLMNOt+1|ABCDEFGHIJKLMNOt", "estado": None},
            {"Subsistema": "ABCDEFGHIJKLMNOt+1|ABCDEFGHIJOt", "estado": None},
            {"Subsistema": "ABCGHIJKLMNOt+1|ABCDEFGHIJKLMNOt", "estado": None},
        ],
        "ABCDEFG": [
            {"Subsistema": "ABt+1|ABCt", "estado": None},
            {"Subsistema": "ACt+1|ABCt", "estado": None},
            {"Subsistema": "ABCt+1|ACt", "estado": None},
            {"Subsistema": "ABCt+1|ABCt", "estado": None},
            {"Subsistema": "ABCDEt+1|ABCDEt", "estado": None},
        ],
        "ABCDE": [
            {"Subsistema": "ABCDt+1|ABCDt", "estado": None},
            {"Subsistema": "ABCDt+1|ABCDEt", "estado": None},
            {"Subsistema": "ABCDEt+1|ABCDt", "estado": None},
            {"Subsistema": "ABCt+1|ABCDEt", "estado": None},
        ],
    },
}

def calcular_estado_subsistema(subsistema, sistema_completo, estado_inicial):
    """Calcula el estado del subsistema derivado del `Estado Inicial`."""
    parte_t = subsistema.split("|")[-1]  # Tomar todo lo que está después de "|"
    variables = list(re.findall(r"[A-Z]", parte_t))
    estado_binario = "".join(estado_inicial[sistema_completo.index(var)] for var in variables)
    return estado_binario

def actualizar_estados(estado_sistema):
    """Actualiza el estado de cada subsistema basado en `Estado Inicial`."""
    sistema_completo = estado_sistema["Sistema"]
    estado_inicial = estado_sistema["Estado Inicial"]
    for sistema_candidato, subsistemas in estado_sistema["Sistemas Candidatos"].items():
        for subsistema in subsistemas:
            subsistema["estado"] = calcular_estado_subsistema(
                subsistema["Subsistema"], sistema_completo, estado_inicial
            )
    return estado_sistema

def mostrar_sistemas(estado_sistema):
    """Muestra los sistemas candidatos y sus subsistemas actualizados."""
    print("\nSistemas Candidatos Disponibles:")
    for sistema, subsistemas in estado_sistema["Sistemas Candidatos"].items():
        print(f"- {sistema}")
        for idx, subsistema in enumerate(subsistemas, 1):
            print(f"  {idx}. {subsistema['Subsistema']} (Estado: {subsistema['estado']})")

def seleccionar_sistema(estado_sistema):
    """
    Permite al usuario seleccionar un sistema y un subsistema.

    Args:
        estado_sistema (dict): Diccionario principal del sistema.

    Returns:
        tuple: Sistema seleccionado y subsistema seleccionado.
    """
    print("\nSistemas Candidatos Disponibles:")
    for sistema, subsistemas in estado_sistema["Sistemas Candidatos"].items():
        print(f"- {sistema}")
        for idx, subsistema in enumerate(subsistemas, 1):
            print(f"  {idx}. {subsistema['Subsistema']} (Estado: {subsistema['estado']})")

    sistema = input("\nIngrese el sistema candidato (ej: ABCDE): ").strip()
    subsistemas = estado_sistema["Sistemas Candidatos"].get(sistema.upper())

    if not subsistemas:
        print("Sistema no encontrado. Intente nuevamente.")
        return seleccionar_sistema(estado_sistema)

    print("\nSubsistemas disponibles:")
    for idx, subsistema in enumerate(subsistemas, 1):
        print(f"{idx}. {subsistema['Subsistema']} (Estado: {subsistema['estado']})")

    try:
        seleccion = int(input("\nSeleccione el número del subsistema: "))
        if 1 <= seleccion <= len(subsistemas):
            return sistema.upper(), subsistemas[seleccion - 1]["Subsistema"]
        else:
            print("Selección inválida. Intente nuevamente.")
            return seleccionar_sistema(estado_sistema)
    except ValueError:
        print("Entrada inválida. Intente nuevamente.")
        return seleccionar_sistema(estado_sistema)

def inicializar_variables(sistema, subsistema):
    """
    Inicializa las variables necesarias para RecursiveCandidateSelection.

    Args:
        sistema (str): Sistema completo seleccionado.
        subsistema (str): Subsistema seleccionado.

    Returns:
        tuple: var_names, v, current_values, z
    """
    # Extraer las variables de la parte en `t`
    parte_t = subsistema.split("|")[-1]
    var_names = list(re.findall(r"[A-Z]", parte_t))

    # Mapear las variables al sistema completo y derivar valores iniciales
    sistema_completo = list(estado_sistema["Sistema"])
    estado_inicial = estado_sistema["Estado Inicial"]
    current_values = [int(estado_inicial[sistema_completo.index(var)]) for var in var_names]

    # Crear `v` y `z`
    v = OrderedSet([f"{var.lower()}t" for var in var_names])
    z = {f"{var.lower()}t": [idx + 1] for idx, var in enumerate(var_names)}

    return var_names, v, current_values, z

# Actualizar estados y mostrar sistemas
estado_sistema_actualizado = actualizar_estados(estado_sistema)
mostrar_sistemas(estado_sistema_actualizado)

# Seleccionar sistema y subsistema
sistema, subsistema = seleccionar_sistema(estado_sistema_actualizado)
var_names, v, current_values, z = inicializar_variables(sistema, subsistema)

# Mostrar inicialización
print("\nInicialización:")
print(f"var_names: {var_names}")
print(f"v: {v}")
print(f"current_values: {current_values}")
print(f"z: {z}")
