import re
from collections.abc import Sequence
from dataclasses import dataclass

CPU_INFORMATION_MAX_SIZE_BYTES = 2**20
DRIVER_VERSION_MAX_SIZE_BYTES = 64 * 2**10
GPU_INFORMATION_MAX_SIZE_BYTES = 64 * 2**10
MAX_FIELD_CHARACTERS = 128
CPU_MODEL_FIELD = "model name"
GPU_MODEL_FIELD = "Model"
DRIVER_VERSION_LINE_PREFIX = "NVRM version:"
DRIVER_VERSION_GRAMMAR = re.compile(r"[0-9]{2,4}(\.[0-9]{1,3}){1,2}")
# The directories of /proc/driver/nvidia/gpus are named after each GPU's bus address, its domain,
# bus, device and function.
BUS_ADDRESS_GRAMMAR = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
PRINTABLE_CHARACTERS = frozenset(chr(code) for code in range(0x20, 0x7F)) | {"\t", "\n"}
NO_GPU_REASON = "/proc/driver/nvidia holds no GPU, which the key of a run on the GPU needs"


class HardwareFactsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HardwareFacts:
    cpu_models: tuple[str, ...]
    gpu_models: tuple[str, ...] = ()
    driver_version: str | None = None

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_models) and self.driver_version is not None


def is_bus_address(name: str) -> bool:
    return BUS_ADDRESS_GRAMMAR.fullmatch(name) is not None


def decode_printable(content: bytes, max_size_bytes: int, what: str) -> str:
    if len(content) > max_size_bytes:
        raise HardwareFactsError(f"{what} holds {len(content)} bytes, beyond {max_size_bytes}")
    text = content.decode("ascii", errors="replace")
    if not set(text) <= PRINTABLE_CHARACTERS:
        raise HardwareFactsError(f"{what} holds a character outside printable ASCII")
    return text


def read_field_values(text: str, field: str, what: str) -> list[str]:
    values = []
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if not separator or name.strip() != field:
            continue
        value = value.strip()
        if not value or len(value) > MAX_FIELD_CHARACTERS:
            raise HardwareFactsError(
                f"{what} gives a {field} of {len(value)} characters, outside 1 to "
                f"{MAX_FIELD_CHARACTERS}"
            )
        values.append(value)
    return values


def read_cpu_models(cpu_information: bytes) -> tuple[str, ...]:
    what = "/proc/cpuinfo"
    text = decode_printable(cpu_information, CPU_INFORMATION_MAX_SIZE_BYTES, what)
    models = sorted(set(read_field_values(text, CPU_MODEL_FIELD, what)))
    if not models:
        raise HardwareFactsError(f"{what} names no {CPU_MODEL_FIELD}")
    return tuple(models)


def read_driver_version(driver_version: bytes) -> str:
    what = "/proc/driver/nvidia/version"
    text = decode_printable(driver_version, DRIVER_VERSION_MAX_SIZE_BYTES, what)
    for line in text.splitlines():
        if line.startswith(DRIVER_VERSION_LINE_PREFIX):
            for word in line.removeprefix(DRIVER_VERSION_LINE_PREFIX).split():
                if DRIVER_VERSION_GRAMMAR.fullmatch(word):
                    return word
    raise HardwareFactsError(
        f"{what} gives no driver version under {DRIVER_VERSION_GRAMMAR.pattern} on its "
        f"{DRIVER_VERSION_LINE_PREFIX} line"
    )


def read_gpu_model(gpu_information: bytes) -> str:
    what = "the information of a GPU in /proc/driver/nvidia/gpus"
    text = decode_printable(gpu_information, GPU_INFORMATION_MAX_SIZE_BYTES, what)
    models = read_field_values(text, GPU_MODEL_FIELD, what)
    if len(models) != 1:
        raise HardwareFactsError(f"{what} gives {len(models)} models, not one")
    return models[0]


# The NVIDIA files are missing while its driver is not loaded: the facts then hold no GPU.
def read_hardware_facts(
    cpu_information: bytes, driver_version: bytes | None, gpu_information_contents: Sequence[bytes]
) -> HardwareFacts:
    cpu_models = read_cpu_models(cpu_information)
    if driver_version is None:
        return HardwareFacts(cpu_models)
    return HardwareFacts(
        cpu_models=cpu_models,
        gpu_models=tuple(
            sorted(read_gpu_model(information) for information in gpu_information_contents)
        ),
        driver_version=read_driver_version(driver_version),
    )
