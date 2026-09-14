import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded.add(weight_name)
    # Fail loudly when the checkpoint does not cover every parameter, e.g. when a
    # model builds q_norm/k_norm that the checkpoint has no weights for. Tied
    # parameters (lm_head sharing embed_tokens storage) are excused via data_ptr.
    loaded_ptrs = {
        param.data_ptr() for name, param in model.named_parameters() if name in loaded
    }
    missing = [
        name
        for name, param in model.named_parameters()
        if name not in loaded and param.data_ptr() not in loaded_ptrs
    ]
    if missing:
        raise RuntimeError(
            f"checkpoint at {path} does not cover {len(missing)} parameter(s): {missing[:5]}"
        )
