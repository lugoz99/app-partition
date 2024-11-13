class ProbabilidadM:
    def __init__(self, V):
        self.V = V  # Universo de elementos
        # Conjunto de elementos en t (sin '+1')
        self.V_t = {e for e in V if "t" in e and "+1" not in e}
        # Conjunto de elementos en t+1 (con '+1')
        self.V_t1 = {e for e in V if "t+1" in e}

    def obtener_PM(self, conjunto_t):
        """
        Dado un conjunto de elementos en t, devuelve la estructura de P(M).
        """
        # Parte izquierda es vacía (sin elementos en t+1)
        izquierda = set()
        # Parte derecha es el conjunto t proporcionado
        derecha = conjunto_t

        # Guardamos las partes de P(M) en atributos para utilizarlas en el complemento
        self.PM_izquierda = izquierda
        self.PM_derecha = derecha

        # Retorna un diccionario con acceso a elementos en t y t+1
        return {
            "P(M)": f"P({izquierda} | {derecha})",
            "t_elements": derecha,
            "t1_elements": izquierda,
        }

    def obtener_complemento_de_PM(self):
        """
        Calcula P(M complemento) a partir de P(M) ya obtenida.
        """
        # Parte complementaria de la parte izquierda
        complemento_izquierda = self.V_t1 - self.PM_derecha
        # Parte complementaria de la parte derecha
        complemento_derecha = self.V_t - self.PM_derecha

        # Retorna un diccionario con acceso a elementos en t y t+1 del complemento
        return {
            "P(M complemento)": f"P({complemento_izquierda} | {complemento_derecha})",
            "t_elements": complemento_derecha,
            "t1_elements": complemento_izquierda,
        }


# Definir el universo V
V = {"at", "bt", "at+1", "bt+1"}

# Crear la instancia
probabilidad = ProbabilidadM(V)

# Ejemplo de uso
P_M = probabilidad.obtener_PM({"at"})
P_M_complemento = probabilidad.obtener_complemento_de_PM()

print(P_M["P(M)"])  # P(M) = P(set() | {'at'})
print(
    P_M_complemento["P(M complemento)"]
)  # P(M complemento) = P({'at+1', 'bt+1'} | {'bt'})

# Acceso directo a elementos t y t+1 en cada probabilidad
print("Elementos en t para P(M):", P_M["t_elements"])
print("Elementos en t+1 para P(M complemento):", P_M_complemento["t1_elements"])
