from typing import *
import numpy as np
from numpy.typing import *


class DistanceMatrixWrapper:
    def __init__(self, array: ArrayLike):
        self.array = array
        self.N = (np.sqrt(1 + 8 * len(array)) - 1) / 2
        if self.N - int(self.N) > 0.00001:
            # print("WARNING: INVALID SIZE FOR FLATTEN DISTANCE MATRIX")
            raise ValueError("INVALID SIZE FOR FLATTEN DISTANCE MATRIX")
        self.N = int(self.N)
        self.shape = (self.N, self.N)

    def get(self, i: int, j: int) -> float:
        if i > j:
            i, j = j, i
        return self.array[i * self.N - self._h(i) + j]

    # number of missing elements of bottom triangle at row i
    def _h(self, i: int) -> int:
        return i * (i + 1) // 2

    def convert_2d_index(self, i: int, j: int) -> int:
        return i*self.N - self._h(i) + j

    def convert_flatten_index(self, k: int) -> Tuple[int, int]:
        discriminant = (2 * self.N + 1)**2 - 8*k
        if discriminant < 0 or k < 0:
            raise ValueError("Invalid 1D index k for the given matrix dimension n.")
        i = int((2 * self.N + 1 - np.sqrt(discriminant)) / 2)
        start_index = i * self.N - (i * (i - 1)) // 2
        offset = k - start_index
        j = i + offset
        return i, j
