# Parakeet TDT 0.6B v3's encoder stacks 24 Conformer layers, and the convolution module of each
# runs two convolutions of kernel size 1, which NVIDIA-NeMo/Speech #16052 runs as matrix products.
CONFORMER_LAYER_COUNT = 24
POINTWISE_CONVOLUTION_NAMES = ("pointwise_conv1", "pointwise_conv2")
CONVOLUTION_MODULES = tuple(
    f"encoder.layers.{layer_index}.conv" for layer_index in range(CONFORMER_LAYER_COUNT)
)
POINTWISE_CONVOLUTIONS = tuple(
    f"{convolution_module}.{pointwise_name}"
    for convolution_module in CONVOLUTION_MODULES
    for pointwise_name in POINTWISE_CONVOLUTION_NAMES
)
