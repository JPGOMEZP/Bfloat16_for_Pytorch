#
# This script performs a matrix-matrix addition for real numbers in single
# precision.
#

# imports
import torch as pt
import os
import sys


# matrix result to compare
r = pt.tensor([11])

# first constant for the addition in float32
a = pt.tensor(5.0, dtype=pt.float32)

# casting from flot32 to int32 to bfloat16
pt.tensor(a, dtype=pt.int32)
pt.tensor(a, dtype=pt.bfloat16)

# second constant for the addition in float32
b = pt.tensor(6.0, dtype=pt.float32)

# casting from flot32 to int32 to bfloat16
pt.tensor(b, dtype=pt.int32)
pt.tensor(b, dtype=pt.bfloat16)

# addition
c = pt.add(a, b)

# comparing the result
result = pt.eq(c,r)

if result.all():
    sys.exit(0)
else:
    sys.exit(1)
