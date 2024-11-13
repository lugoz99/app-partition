## **Entrada de Datos**

1. **La TPM del sistema completo:**  
   - Matriz de transición de probabilidades (TPM) que representa el comportamiento del sistema.

2. **El subconjunto de elementos a analizar (sistema candidato):**  
   - **t**: Elementos presentes únicamente en el tiempo inicial \(t\).

3. **El subconjunto del sistema candidato a analizar:**  
   - **t** y **t + 1**: Elementos presentes tanto en el tiempo actial como en el futuro.
     - aqui deben darse los elementos tanto en t como en t+1, ya que no necesariamente 
     se tendran en t+1 los mismos e elementos que en t.

4. **El estado actual de todos los elementos del sistema:**  
   - 1 ó 0


def original_distribution(matrix, initial_state):
    """
    Calcula la distribución de probabilidades original del sistema.

    Args:
        matrix (np.ndarray): Matriz de transición de estados.
        initial_state (np.ndarray): Estado inicial del sistema.

    Returns:
        np.ndarray: Distribución de probabilidades original.
    """
    # Inicializamos la distribución de probabilidades con el estado inicial
    distribution = initial_state

    # Iteramos
    pass


def marginalize_column(matrix, column):
    """
    Marginaliza una columna de una matriz.

    Args:
        matrix (np.ndarray): Matriz a marginalizar.
        column (int): Columna a marginalizar.

    Returns:
        np.ndarray: Matriz marginalizada.
    """
    return

def marginalize_rows(matrix, rows):
    """
    Marginaliza filas de una matriz.

    Args:
        matrix (np.ndarray): Matriz a marginalizar.
        rows (List[int]): Filas a marginalizar.

    Returns:
        np.ndarray: Matriz marginalizada.
    """
    return


def marginalize_special_case():
    pass




def product_tensor(matrix1, matrix2):
    """
    Calcula el producto tensorial entre dos matrices.

    Args:
        matrix1 (np.ndarray): Primera matriz.
        matrix2 (np.ndarray): Segunda matriz.

    Returns:
        np.ndarray: Producto tensorial.
    """
    return




def compare_distributions(distribution1, distribution2):
    """
    Compara dos distribuciones de probabilidades.

    Args:
        distribution1 (np.ndarray): Primera distribución sin dividir.
        distribution2 (np.ndarray): Segunda distribución dividida.

    Returns:
        float: Distancia entre las distribuciones.
    """
    return