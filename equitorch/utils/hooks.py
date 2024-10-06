import torch

import inspect

def null_hook(module, args, kwargs):
    return args, kwargs

def get_kwargs_filter_hook(module):
    args_name = inspect.signature(module.forward).parameters.keys()
    if 'kwargs' in args_name:
        return null_hook
    else:
        def kwargs_filter_hook(module, args, kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k in args_name}
            return args, kwargs
        return kwargs_filter_hook