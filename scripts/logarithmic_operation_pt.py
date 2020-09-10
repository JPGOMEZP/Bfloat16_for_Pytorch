#
# This script performs a logarithmic operations for real numbers in single
# precision.
#

# imports
import torch as pt
import os
import sys
import subprocess

# matrix result to compare
r = pt.tensor(2.6390574, dtype=pt.float32)

# first matrix in float32
a = pt.tensor(14, dtype=pt.float32)

# casting from flot32 to int32 to bfloat16
pt.tensor(a, dtype=pt.int32)
pt.tensor(a, dtype=pt.bfloat16)

# logarithmic operation
c = pt.log(a)

# comparing the result
result = pt.eq(c,r)

# validate exit code
if result.all():
    sys.exit(0)
else:
    sys.exit(1)
