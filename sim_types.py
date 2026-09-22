"""Shared numerical and simulation data types."""

from typing import Mapping, Sequence, Union

import numpy as np
from numpy.typing import NDArray

Scalar = Union[float, np.float64]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
VectorLike = Union[FloatArray, Sequence[Scalar]]
MatrixLike = Union[FloatArray, Sequence[VectorLike]]
Point3 = tuple[Scalar, Scalar, Scalar]
GridCell = tuple[int, int]
TerrainGrid = FloatArray
ContactMap = Mapping[GridCell, Sequence[FloatArray]]
BladeFrame = list[FloatArray]
PileState = tuple[Scalar, Scalar, float, float]
ForceSample = dict[str, float]
