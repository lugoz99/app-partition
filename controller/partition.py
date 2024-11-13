import numpy as np
from pyemd import emd

from numpy.typing import NDArray


def hamming_distance(a: int, b: int) -> int:
    """Calcula la distancia de Hamming entre dos enteros."""
    return (a ^ b).bit_count()


def emd_pyphi(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """
    Calcula la Earth Mover's Distance (EMD) entre dos distribuciones de probabilidad u y v.
    La distancia de Hamming se utiliza como métrica base.
    """
    if not all(isinstance(arr, np.ndarray) for arr in [u, v]):
        raise TypeError("u y v deben ser arreglos de numpy.")

    n: int = len(u)
    costs: NDArray[np.float64] = np.empty((n, n))

    for i in range(n):
        costs[i, :i] = [hamming_distance(i, j) for j in range(i)]
        costs[:i, i] = costs[i, :i]
    np.fill_diagonal(costs, 0)

    cost_matrix: NDArray[np.float64] = np.array(costs, dtype=np.float64)
    return emd(u, v, cost_matrix)


def g(X: NDArray[np.float64], V: NDArray[np.float64]) -> float:
    """Calcula la función EMD entre la distribución de probabilidades resultante de P(X) ⊗ P(X’) y la distribución del sistema sin dividir."""

    print("X:", X)
    print("V:", V)


def strategi_best_partition(V: NDArray[np.float64]) -> list:
    """Encuentra la mejor partición del conjunto V."""
    n = len(V)
    W = []
    W.append([])  # W0 = ∅
    W.append([V[0]])  # W1 = {v1}, donde v1 es un elemento arbitrario de V
    print("W1:", W[1])

    for i in range(2, n + 1):
        best_candidate = None
        best_value = float("inf")
        for v in V:
            if v not in W[i - 1]:
                candidate_value = g(np.array(W[i - 1] + [v]), V) - g(np.array([v]), V)
                if candidate_value < best_value:
                    best_value = candidate_value
                    best_candidate = v
        W.append(W[i - 1] + [best_candidate])
        print(f"W{i}:", W[i])

    # Construcción de pares candidatos
    candidate_pairs = [(W[-2][-1], W[-1][-1])]
    print("Candidate pairs:", candidate_pairs)

    # Recursión
    while len(V) > 2:
        u = candidate_pairs[-1]
        V = [v for v in V if v not in u] + [u]
        W = []
        W.append([])  # W0 = ∅
        W.append([V[0]])  # W1 = {v1}, donde v1 es un elemento arbitrario de V
        for i in range(2, len(V) + 1):
            best_candidate = None
            best_value = float("inf")
            for v in V:
                if v not in W[i - 1]:
                    candidate_value = g(np.array(W[i - 1] + [v]), V) - g(
                        np.array([v]), V
                    )
                    if candidate_value < best_value:
                        best_value = candidate_value
                        best_candidate = v
            W.append(W[i - 1] + [best_candidate])
        candidate_pairs.append((W[-2][-1], W[-1][-1]))
        print("Candidate pairs:", candidate_pairs)

    # Evaluación final
    best_partition = None
    best_value = float("inf")
    for a, b in candidate_pairs:
        partition_value = g(np.array([b]), V) - g(np.array([a]), V)
        if partition_value < best_value:
            best_value = partition_value
            best_partition = (a, b)
    print("Best partition:", best_partition)
    return best_partition


if __name__ == "__main__":
    # Ejemplo con 4 nodos
    V = np.array(["at", "bt", "at+1", "bt+1"])
    best_partition = strategi_best_partition(V)

if __name__ == "__main__":
    # Ejemplo con 4 nodos
    V = np.array(["at", "bt", "at+1", "bt+1"])
    best_partition = strategi_best_partition(V)
