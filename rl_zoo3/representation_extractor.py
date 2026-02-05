from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn


class RepresentationExtractor:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.hook_layers = []

        if device is None:
            if hasattr(model, "device"):
                self.device = model.device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        print(self.model)

    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        layers = dict(self.model.named_modules())
        return layers.get(layer_name)

    def register_forward_hook(self, layer_name: str) -> bool:
        layer = self._get_layer_by_name(layer_name)
        if layer is None:
            return False

        if layer_name in self.hook_layers:
            return True

        def hook_fn(module, input, output):
            try:
                self.activations[layer_name] = {}
                if isinstance(module, nn.LSTM):
                    self.activations[layer_name]["input"] = input[0].detach()
                    self.activations[layer_name]["output"] = output[0].detach()
                    self.activations[layer_name]["hidden"] = output[1][0].detach()
                    self.activations[layer_name]["cell"] = output[1][1].detach()
                else:
                    self.activations[layer_name]["input"] = input[0].detach()
                    self.activations[layer_name]["output"] = output.detach()
            except Exception as e:
                print(f"Error in forward hook for layer {layer_name}: {e}")
                import pdb

                pdb.set_trace()

        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        self.hook_layers.append(layer_name)
        return True

    def register_forward_hooks_by_layer_names(self, layer_names: List[str]) -> int:
        count = 0
        for name in layer_names:
            if self.register_forward_hook(name):
                count += 1
        return count

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.hook_layers = []

    def clear_activations(self):
        self.activations = {}

    def get_activations(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        if layer_name is not None:
            return self.activations.get(layer_name)
        return {k: deepcopy(v) for k, v in self.activations.items()}

    def register_forward_hooks(
        self,
        include: Optional[Union[List[str], str]] = None,
        exclude: Optional[Union[List[str], str]] = None,
        selector: Optional[Callable[[str, nn.Module], bool]] = None,
    ):
        if selector is not None:
            layers = dict(self.model.named_modules())
            for name, layer in layers.items():
                if selector(name, layer):
                    self.register_forward_hook(name)
        else:
            include_layers = []
            exclude_layers = []

            if include is not None:
                include_layers = [include] if isinstance(include, str) else include
            if exclude is not None:
                exclude_layers = [exclude] if isinstance(exclude, str) else exclude

            layers = dict(self.model.named_modules())
            for name, layer in layers.items():
                if include and len(include) > 0 and name not in include_layers:
                    continue
                if exclude and len(exclude) > 0 and name in exclude_layers:
                    continue
                if isinstance(layer, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU, nn.MultiheadAttention)):
                    self.register_forward_hook(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False
