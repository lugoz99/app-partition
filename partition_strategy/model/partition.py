class PartitionModel:
    def __init__(self, particion1, particion2, emd_valor):
        self.particion1 = particion1
        self.particion2 = particion2
        self.emd_valor = emd_valor

    def to_dict(self):
        return {
            "particion1": self.particion1,
            "particion2": self.particion2,
            "emd_valor": self.emd_valor,
        }

    def __str__(self):
        return (
            f"Partición 1: {self.particion1}, "
            f"Partición 2: {self.particion2}, EMD: {self.emd_valor}"
        )

    def __repr__(self):
        return self.__str__()
