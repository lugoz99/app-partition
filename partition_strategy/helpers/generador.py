import pandas as pd
import numpy as np

# Datos de entrada
data = {
    "primogenitalTables": {
        "A": [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]],
        "B": [[1, 1, 1, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1]],
        "C": [[1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0]],
    },
    "stateSought": "110",
}


data_matrices = data["primogenitalTables"]
transposed_numpy = {
    key: np.array(value).T for key, value in data["primogenitalTables"].items()
}

# Mostrar las matrices transpuestas

# Mostrar las transpuestas


# Definición de funciones
def productTensor_v3(row, col, data):
    if row == 0:
        return data["A"][0][col] * data["B"][0][col] * data["C"][0][col]
    if row == 1:
        return data["A"][1][col] * data["B"][0][col] * data["C"][0][col]
    if row == 2:
        return data["A"][0][col] * data["B"][1][col] * data["C"][0][col]
    if row == 3:
        return data["A"][1][col] * data["B"][1][col] * data["C"][0][col]
    if row == 4:
        return data["A"][0][col] * data["B"][0][col] * data["C"][1][col]
    if row == 5:
        return data["A"][1][col] * data["B"][0][col] * data["C"][1][col]
    if row == 6:
        return data["A"][0][col] * data["B"][1][col] * data["C"][1][col]
    if row == 7:
        return data["A"][1][col] * data["B"][1][col] * data["C"][1][col]


def getStatus_v3():
    return [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]


def generatorProbabilities(data):
    def createTableGeneral(data):
        num_cols = len(data["primogenitalTables"]["A"][0])
        status = []
        result_matrix = [[-1] * num_cols for _ in range(num_cols)]

        for col in range(num_cols):
            for row in range(num_cols):
                if len(data["primogenitalTables"].keys()) == 3:
                    result_matrix[col][row] = productTensor_v3(
                        row, col, data["primogenitalTables"]
                    )
        valueStatus = searchStatus(data, result_matrix)
        if len(data["primogenitalTables"].keys()) == 3:
            status = getStatus_v3()
        return valueStatus, result_matrix, status

    def searchStatus(data, result_matrix):
        statusPosition = int(str(data["stateSought"])[::-1], 2)
        return result_matrix[statusPosition]

    return createTableGeneral(data)


# Generar resultados
_, result_matrix, status = generatorProbabilities(data)

# Convertir la matriz resultante a DataFrame para mejor visualización
df_result = pd.DataFrame(result_matrix)
print(df_result)

print(transposed_numpy["A"])


"""
import json

from generator.calculateProbabilities_v3 import productTensor_v3, getStatus_v3
from generator.calculateProbabilities_v4 import productTensor_v4, getStatus_v4
from generator.calculateProbabilities_v5 import productTensor_v5, getStatus_v5
from generator.calculateProbabilities_v6 import productTensor_v6, getStatus_v6
from generator.calculateProbabilities_v7 import productTensor_v7, getStatus_v7
from generator.calculateProbabilities_v8 import productTensor_v8, getStatus_v8


def generatorProbabilities(data):

    def createTableGeneral(data):

        num_cols = len(data["primogenitalTables"]["A"][0])
        status = []

        # Inicializar una matriz para almacenar los resultados de la multiplicación
        result_matrix = [[-1] * num_cols for _ in range(num_cols)]

        for col in range(num_cols):
            for row in range(num_cols):

                if(len(data["primogenitalTables"].keys()) == 3):
                    result_matrix[col][row] = productTensor_v3(row,col,data["primogenitalTables"]);
                if(len(data["primogenitalTables"].keys()) == 4):
                    result_matrix[col][row] = productTensor_v4(row,col,data["primogenitalTables"]);
                if(len(data["primogenitalTables"].keys()) == 5):
                    result_matrix[col][row] = productTensor_v5(row,col,data["primogenitalTables"]);
                if(len(data["primogenitalTables"].keys()) == 6):
                    result_matrix[col][row] = productTensor_v6(row,col,data["primogenitalTables"]);
                if(len(data["primogenitalTables"].keys()) == 7):
                    result_matrix[col][row] = productTensor_v7(row,col,data["primogenitalTables"]);
                if(len(data["primogenitalTables"].keys()) == 8):
                    result_matrix[col][row] = productTensor_v8(row,col,data["primogenitalTables"]);
                

        # Imprimir la matriz resultante
        # print("Matriz resultante:")
        #for row in result_matrix:
            #print(row)

        #Buscar el estado especifico
        valueStatus = searchStatus(data,result_matrix)

        if(len(data["primogenitalTables"].keys()) == 3):
            status = getStatus_v3()
        if(len(data["primogenitalTables"].keys()) == 4):
            status = getStatus_v4()
        if(len(data["primogenitalTables"].keys()) == 5):
            status = getStatus_v5()
        if(len(data["primogenitalTables"].keys()) == 6):
            status = getStatus_v6()
        if(len(data["primogenitalTables"].keys()) == 7):
            status = getStatus_v7()
        if(len(data["primogenitalTables"].keys()) == 8):
            status = getStatus_v8()
        

        return valueStatus, result_matrix, status
        
    def searchStatus(data, result_matrix):

        statusPosition = int(str(data["stateSought"])[::-1],2);
        return result_matrix[statusPosition]


    return createTableGeneral(data)

"""
