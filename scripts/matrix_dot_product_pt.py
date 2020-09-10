#
# This script performs a matrix-matrix dot product for real numbers in single
# precision.
#

# imports
import torch as pt
import os
import sys

r = pt.tensor([[13, 13], [13, 13]])

# first matrix in float32
a = pt.tensor([[1, 2, 3], [1, 2, 3]]).view(-1, 2)

# casting from flot32 to int32 to bfloat16
pt.tensor(a, dtype=pt.int32)
pt.tensor(a, dtype=pt.bfloat16)

# second matrix in float32
b = pt.tensor([[1, 2, 3], [1, 2, 3]]).view(2, -1)

# casting from flot32 to int32 to bfloat16
pt.tensor(b, dtype=pt.int32)
pt.tensor(b, dtype=pt.bfloat16)

# matrix operation
c = pt.mm(b, a)

# comparing the result
result = pt.eq(c,r)

if result.all():
    sys.exit(0)
else:
    sys.exit(1)
