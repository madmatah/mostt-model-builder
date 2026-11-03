"""Ported from NVIDIA-NeMo/Speech #16052, under its Apache-2.0 license, which CREDITS.md keeps."""

import types
from typing import Any

import torch

from mostt_recipe_parakeet_tdt.domain.export.pointwise_convolutions import (
    CONVOLUTION_MODULES,
    POINTWISE_CONVOLUTIONS,
)

# torch is installed in the export image alone, so mypy on the host knows nothing of its types.
type TorchModule = Any
type TorchTensor = Any

CONVOLUTION_MODULE_TYPE = "ConformerConvolution"
GATED_LINEAR_UNIT_ACTIVATION = "glu_"
LAYER_NORM_TYPE = "layer_norm"
POINTWISE_WEIGHT_DIMENSIONS = 3


class PointwiseForwardError(Exception):
    pass


def apply_pointwise(features: TorchTensor, convolution: TorchModule) -> TorchTensor:
    return torch.nn.functional.linear(features, convolution.weight, convolution.bias)


# Ported from NVIDIA-NeMo/Speech #16052 onto NeMo 3.0.0, which predates it, so that the export
# reads each product's weight as a plain 2-D initializer. NeMo calls it with these keywords.
def forward_with_matrix_products(
    convolution_module: TorchModule,
    features: TorchTensor,
    pad_mask: TorchTensor | None = None,
    cache: TorchTensor | None = None,
) -> TorchTensor:
    features = apply_pointwise(features, convolution_module.pointwise_conv1)
    if convolution_module.pointwise_activation == GATED_LINEAR_UNIT_ACTIVATION:
        features = torch.nn.functional.glu(features, dim=-1)
    else:
        features = convolution_module.pointwise_activation(features)
    features = features.transpose(1, 2)
    if pad_mask is not None:
        features = features.masked_fill(pad_mask.unsqueeze(1), 0.0)
    features = convolution_module.depthwise_conv(features, cache=cache)
    if cache is not None:
        features, cache = features
    if convolution_module.norm_type == LAYER_NORM_TYPE:
        features = features.transpose(1, 2)
        features = convolution_module.batch_norm(features)
    else:
        features = convolution_module.batch_norm(features)
        features = features.transpose(1, 2)
    features = convolution_module.activation(features)
    features = apply_pointwise(features, convolution_module.pointwise_conv2)
    if cache is None:
        return features
    return features, cache


def flatten_pointwise_weight(model: TorchModule, name: str) -> None:
    convolution = model.get_submodule(name)
    weight = convolution.weight.detach()
    if weight.dim() != POINTWISE_WEIGHT_DIMENSIONS or weight.shape[-1] != 1:
        raise PointwiseForwardError(
            f"{name} has a weight of the shape {list(weight.shape)}, not [out, in, 1]"
        )
    convolution.weight = torch.nn.Parameter(weight.squeeze(-1).contiguous(), requires_grad=False)


def run_pointwise_as_matrix_products(model: TorchModule) -> None:
    found_modules = tuple(
        name
        for name, module in model.named_modules()
        if type(module).__name__ == CONVOLUTION_MODULE_TYPE
    )
    if found_modules != CONVOLUTION_MODULES:
        raise PointwiseForwardError(
            f"the model holds {len(found_modules)} convolution modules, not the "
            f"{len(CONVOLUTION_MODULES)} of Parakeet TDT's encoder"
        )
    for pointwise_convolution in POINTWISE_CONVOLUTIONS:
        flatten_pointwise_weight(model, pointwise_convolution)
    for module_name in CONVOLUTION_MODULES:
        convolution_module = model.get_submodule(module_name)
        convolution_module.forward = types.MethodType(
            forward_with_matrix_products, convolution_module
        )
