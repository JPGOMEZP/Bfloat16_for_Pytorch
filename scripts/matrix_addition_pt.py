#
# This script performs a matrix-matrix addition for real numbers in single
# precision.
#

# imports
import torch as pt
import os
import sys

# matrix result to compare
r = pt.tensor([[2, 4, 6], [8,10, 12]])

# first matrix in float32
a = pt.tensor([[1, 2, 3], [4, 5, 6]], dtype=pt.float32)

# casting from flot32 to int32 to bfloat16
pt.tensor(a, dtype=pt.int32)
pt.tensor(a, dtype=pt.bfloat16)

# second matrix in float32
b = pt.tensor([[1, 2, 3], [4, 5, 6]], dtype=pt.float32)

# casting from flot32 to int32 to bfloat16
pt.tensor(b, dtype=pt.int32)
pt.tensor(b, dtype=pt.bfloat16)

# matrix operation
c = pt.add(a, b)

# comparing the result
result = pt.eq(c,r)

# validate the exit code
if result.all():
    sys.exit(0)
else:
    sys.exit(1)
