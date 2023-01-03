# %%
from typing import Any, NoReturn, Union, List, Type, Annotated
from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked
from collections import namedtuple
from frozendict import frozendict
from transformer_lens import FactoredMatrix, utils
import torch as t

# %%

class _FactoredMatrixTypeMeta(type(FactoredMatrix)):
    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, cls.base_cls)

class FactoredMatrixTypeMixin(metaclass=_FactoredMatrixTypeMeta):
    def __new__(cls, *args, **kwargs) -> NoReturn:
        raise RuntimeError(f"Class {cls.__name__} cannot be instantiated.")

    @staticmethod
    def _type_error(item: Any) -> NoReturn:
        raise TypeError(f"{item} not a valid type argument.")

    def __class_getitem__(cls, item: Any):
        details = item
        return Annotated[
            cls.base_cls,
            frozendict({
                    "details": details, 
                    "cls_name": cls.__name__
            }),
        ]

class FactoredMatrixType(FactoredMatrix, FactoredMatrixTypeMixin):
    base_cls = FactoredMatrix

# %%

FMT = FactoredMatrixType

def multiply_factored_matrices(
    X: FMT["ldim_X", "mdim_X", "rdim_X"], 
    Y: FMT["rdim_X", "mdim_Y", "rdim_Y"],
    bottleneck_on_right: bool = True
) -> FMT["ldim_X", "rdim_Y"]:

    if bottleneck_on_right:
        return FactoredMatrix(X @ Y.A, Y.B)
    else:
        return FactoredMatrix(X.A, X.B @ Y)

# %%

# assert str(TT.__class_getitem__("a")) == str(TT["a"])

class FactoredMatrixType(FactoredMatrix, TT):
    
    def __init__(self):
        super().__init__()
        self.base_cls = FactoredMatrix

    def __class_getitem__(cls, item: Any):
        type_annotation_for_tensor = super().__class_getitem__(item)
        type_annotation_for_tensor.__args__ = (FactoredMatrix,)
        print(type_annotation_for_tensor.__args__)
        return type_annotation_for_tensor

FMT = FactoredMatrixType

# %%

# patch_typeguard()

# @typechecked
# def add_together(X: FMT["A", "B"], Y: FMT["C": "D", "A", "B"]) -> FMT["C":"E", "A", "B"]:
#     return X + Y

# a = add_together(t.randn(3, 4), t.randn(2, 3, 4))

# %%

s = r"""

Conclusion: there are too many moving parts for this to be practical.

I'd love something which would allow you to do this:
"""

patch_typeguard()

@typechecked
def multiply(
    X: FMT["ldim_X", "rdim_X"],
    Y: FMT["rdim_X", "rdim_Y", "mdim": "mdim_Y"],
    bottleneck_on_right: bool = True
) -> FMT["ldim_X", "rdim_Y"]:
    
    if bottleneck_on_right:
        return FactoredMatrix(X @ Y.A, Y.B)
    else:
        return FactoredMatrix(X.A, X.B @ Y)

a = multiply(
    FactoredMatrix(t.randn(10, 2), t.randn(2, 15)),
    FactoredMatrix(t.randn(15, 3), t.randn(3, 20)),
)

# %%

s = r"""

Another thought - can I hackily do something under the hood like:

FactoredMatrixType[TT["a", "mid"], TT["mid", "b"]]

i.e. each of those refers to the tensors A and B.
"""

# %%

class FactoredMatrixType(TT):
    pass

FMT = FactoredMatrixType