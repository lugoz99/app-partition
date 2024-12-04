import numpy as np
from scipy.stats import wasserstein_distance as EMD
from ordered_set import OrderedSet


# Supongamos que tenemos una función para calcular la EMD
def calcular_emd(distribucion1, distribucion2):
    if not distribucion1.size or not distribucion2.size:
        raise ValueError("Las distribuciones no pueden estar vacías.")
    return EMD(distribucion1, distribucion2)


# Definir la función g que usaremos
def g(X, V):
    # Simular distribuciones de probabilidad para los ejemplos
    P_X = np.random.rand(len(X)) if len(X) > 0 else np.array([0])
    P_X_complemento = (
        np.random.rand(len(V) - len(X)) if len(V) - len(X) > 0 else np.array([0])
    )
    P_V = np.random.rand(len(V))

    # Calcular EMD
    return calcular_emd(np.outer(P_X, P_X_complemento).flatten(), P_V)


# Algoritmo principal
def encontrar_particiones_optimas(V, ciclo=1, particiones=[]):
    n = len(V)
    W = [OrderedSet() for _ in range(n + 1)]
    W[1] = OrderedSet([V[0]])

    for i in range(2, n + 1):
        vi_min = None
        min_valor = float("inf")
        for vi in V:
            if vi not in W[i - 1]:
                valor = g(W[i - 1] | OrderedSet([vi]), V) - g(OrderedSet([vi]), V)
                if valor < min_valor:
                    min_valor = valor
                    vi_min = vi
        W[i] = W[i - 1] | OrderedSet([vi_min])

    # Obtener el par candidato
    par_candidato = (list(W[n - 1])[-1], list(W[n])[-1])

    # Añadir partición como "Partición 1" y "Partición 2"
    particion1 = OrderedSet([par_candidato[1]])
    particion2 = OrderedSet(V) - particion1
    particiones.append((f"Ciclo {ciclo}", particion1, particion2, g(particion1, V)))

    # Recursión con fusión y expansión del nodo fusionado
    if len(V) > 2:
        u = tuple(par_candidato)
        # Aquí hacemos la fusión de los dos nodos en un nuevo nodo 'u'
        V = [x for x in V if x not in par_candidato] + [u]
        # Calcular el nuevo g teniendo en cuenta el nodo 'u' fusionado
        particiones.append(
            (
                f"Ciclo {ciclo}",
                OrderedSet([u]),
                OrderedSet(V) - OrderedSet([u]),
                g(OrderedSet([u]), V),
            )
        )
        # Llamada recursiva con el conjunto V actualizado
        encontrar_particiones_optimas(V, ciclo + 1, particiones)

    return particiones


# Ejemplo
V = ["at", "bt", "at+1", "bt+1"]
particiones_optimas = encontrar_particiones_optimas(V)
for ciclo, particion1, particion2, emd_valor in particiones_optimas:
    print(
        f"{ciclo} - Partición 1: {particion1}, Partición 2: {particion2}, EMD: {emd_valor}"
    )
