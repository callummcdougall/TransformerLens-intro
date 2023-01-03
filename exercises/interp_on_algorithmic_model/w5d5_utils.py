import torch as t

def remove_hooks(module: t.nn.Module):
    """Remove all hooks from module.
    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()