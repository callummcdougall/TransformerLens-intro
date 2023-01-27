# idea for einops - support (batch chan) out

# 3 cases: 

# (1) have brackets in the inputs, want to rearrange

# einsum(X, Y, "a (b c), b c -> a")

# should actually call:

# X_temp = einops.rearrange(X, "a (b c) -> a b c")
# return einsum(X_temp, Y, "a b c, b c -> a")

# (2) have brackets in the outputs, want to rearrange them

# einsum(X, Y, "a b, b c -> (a c)")

# should actually call:

# out_temp = einsum(X, Y, "a b, b c -> a c")
# return einops.rearrange(out_temp, "a c -> (a c)")

# (3) have extra singletons in the output (in the input would make no sense!)

# This is dealt with in the same way as (2)

import einops
from fancy_einsum import einsum
import re
import torch as t

def my_einsum(*args, **kwargs) -> t.Tensor:

    *tensors, equation = args

    assert all([isinstance(tensor, t.Tensor) for tensor in tensors])
    assert isinstance(equation, str)
    assert "()" not in equation, "Please specify singletons in output with '1' rather than '()'"

    inputs, output = equation.split("->")

    inputs = [s.strip() for s in inputs.split(",")]
    output = output.strip()

    assert len(inputs) == len(tensors)

    for i, (tensor, inp) in enumerate(zip(tensors, inputs)):
        if "(" in inp:
            inp_new = inp.replace('(', '').replace(')', '')
            kwargs_rearrange = {k: v for k, v in kwargs.items() if k in inp_new.split(" ")}
            tensors[i] = einops.rearrange(tensor, f"{inp} -> {inp_new}", **kwargs_rearrange)
            inputs[i] = inp_new
    output_new = output.replace('(', '').replace(')', '').replace("1", "")
    while "  " in output_new:
        output_new = output.replace("  ", " ")
    equation_intermediate = " -> ".join([", ".join(inputs), output_new])
    out_intermediate = einops.einsum(*tensors, equation_intermediate)
    if output == output_new:
        return out_intermediate
    else:
        kwargs_output = {k: v for k, v in kwargs.items() if k in output_new.split(" ")}
        return einops.rearrange(out_intermediate, f"{output_new} -> {output}", **kwargs_output)

    

# %%

import torch as t
import einops
einops.rearrange(t.ones(3, 4, 5), "i (j k) l -> i j k l", j=2, z=2)
# %%
