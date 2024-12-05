import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append("d:/AnalisisYDiseñoAlgoritmos/Project-2024-02/project-ada")

def leer_matriz(file_path, delimiter=','):
    # Lee el archivo CSV y conviértelo en una matriz NumPy
    data = np.loadtxt(file_path, delimiter=delimiter)
    
    # Retorna la matriz NumPy resultante
    return data

# Ejemplo de uso
file_path = 'app/partition_strategy/data/red10.csv'  # Cambia esto por la ruta de tu archivo
matriz = leer_matriz(file_path)

print(matriz)  # Imprime la matriz NumPy resultante
print(f"Filas: {matriz.shape[0]}, Columnas: {matriz.shape[1]}")