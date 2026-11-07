from collections.abc import Mapping
from types import MappingProxyType

from mostt_model_builder.domain.containers.devices import Device

# Each step gives its libraries as many threads as its container has CPUs.
EXPORT_CPU_COUNT = 8
REFERENCE_CPU_COUNT = 8
TRANSFORMERS_CPU_COUNT = 8
TRANSCRIPTION_CPU_COUNTS: Mapping[Device, int] = MappingProxyType({Device.CPU: 2, Device.GPU: 8})
