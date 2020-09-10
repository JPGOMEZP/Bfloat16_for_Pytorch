#
# This script performs a matrix-matrix addition for real numbers in single
# precision.
#

# imports

import torch as pt
import os
import sys

#Casting x to int32 and bfloat16
x = pt.tensor([[1, 2, 3], [4, 5, 6]], dtype=pt.float32)
pt.tensor(x, dtype=pt.int32)
pt.tensor(x, dtype=pt.bfloat16)

#Casting y to int32 and bfloat16
y = pt.tensor([[7, 8, 9], [10, 11, 12]], dtype=pt.float32)
pt.tensor(x, dtype=pt.int32)
pt.tensor(x, dtype=pt.bfloat16)
