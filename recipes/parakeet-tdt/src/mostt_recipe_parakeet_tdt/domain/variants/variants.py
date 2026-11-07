from dataclasses import dataclass

from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.onnx.conversion_request import (
    Conversion,
    ConversionKind,
    EightBitBlocksConversion,
    FloatConversion,
)
from mostt_recipe_parakeet_tdt.domain.variants.candidate_layout import name_encoder_file

FLOAT32_VARIANT = "float32"
FLOAT16_VARIANT = "float16"
BFLOAT16_VARIANT = "bfloat16"
EIGHT_BIT_BLOCKS_VARIANT = "eight-bit-blocks"
# The exported encoder holds 265 matrix products of a constant weight, its 48 pointwise
# convolutions included.
QUANTIZED_PRODUCT_COUNT = 265
# A converted encoder is a single file of 1.24 GB in float16 and bfloat16 and 692 MB in 8-bit
# blocks, under the 2 GiB that a protobuf may hold.
CONVERTED_ENCODER_MAX_SIZE_BYTES = 2 * 2**30


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    devices: tuple[Device, ...]
    conversion: Conversion | None = None


# float16 and bfloat16 serve the GPU, whose kernels compute in them; on the CPU, ONNX Runtime would
# convert them back to float32 around most nodes. The 8-bit blocks serve both devices.
VARIANTS = (
    Variant(FLOAT32_VARIANT, (Device.CPU, Device.GPU)),
    Variant(
        FLOAT16_VARIANT,
        (Device.GPU,),
        FloatConversion(ConversionKind.FLOAT16, name_encoder_file(FLOAT16_VARIANT)),
    ),
    Variant(
        BFLOAT16_VARIANT,
        (Device.GPU,),
        FloatConversion(ConversionKind.BFLOAT16, name_encoder_file(BFLOAT16_VARIANT)),
    ),
    Variant(
        EIGHT_BIT_BLOCKS_VARIANT,
        (Device.GPU, Device.CPU),
        # Blocks of 64 weights, symmetric, multiplied in int8 at accuracy level 4: the parameters
        # whose transcripts were measured within 0.1 point of float32's on both devices.
        EightBitBlocksConversion(
            name_encoder_file(EIGHT_BIT_BLOCKS_VARIANT),
            block_size=64,
            is_symmetric=True,
            accuracy_level=4,
            quantized_product_count=QUANTIZED_PRODUCT_COUNT,
        ),
    ),
)


def name_run(variant: str, device: Device) -> str:
    return f"{variant}-{device.value}"
