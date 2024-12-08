from ordered_set import OrderedSet

# Crear un OrderedSet
v = OrderedSet(["a", "b", "c", "d"])

# Imprimir el conjunto original
print("Conjunto original:", v)

# Eliminar un elemento
v.remove(v[-1])

# Imprimir el conjunto modificado
print("Conjunto después de eliminar 'c':", v)

# Intentar eliminar un elemento que no existe
try:
    v.remove("z")
except KeyError as e:
    print("Error al intentar eliminar un elemento inexistente:", e)
