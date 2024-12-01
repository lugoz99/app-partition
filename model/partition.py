from typing import List


class PartitionModel:
    def __init__(self, particion1: List[str], particion2: List[str], emd: float):
        self.particion1 = particion1
        self.particion2 = particion2
        self.emd = emd

    def print(self):
        print("Partición 1:", self.particion1)
        print("Partición 2:", self.particion2)
        print("EMD:", self.emd)
