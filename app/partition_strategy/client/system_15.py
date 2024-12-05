estado_inicial = "100000000000000"  # Estado inicial en binario
sistema_completo = "ABCDEFGHIJKLMNO"  # Variables del sistema completo
subsistemas_candidatos = {
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
}
    
    