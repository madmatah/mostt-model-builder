import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.cache.hardware_facts import (
    HardwareFacts,
    HardwareFactsError,
    is_bus_address,
    read_hardware_facts,
)

CPU_MODEL = "AMD Ryzen 9 7950X 16-Core Processor"
OTHER_CPU_MODEL = "AMD EPYC 9654 96-Core Processor"


def build_processor(index: int, model: str) -> bytes:
    return (
        f"processor\t: {index}\n"
        "vendor_id\t: AuthenticAMD\n"
        "cpu family\t: 25\n"
        f"model\t\t: 97\n"
        f"model name\t: {model}\n"
        "flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr\n"
        "\n"
    ).encode()


CPU_INFORMATION = build_processor(0, CPU_MODEL) + build_processor(1, CPU_MODEL)
TWO_SOCKETS_INFORMATION = build_processor(0, OTHER_CPU_MODEL) + build_processor(1, CPU_MODEL)
PROPRIETARY_DRIVER_VERSION = (
    b"NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.14  Thu Feb 22 01:44:30 UTC 2024\n"
    b"GCC version:  gcc version 12.2.0 (Debian 12.2.0-14)\n"
)
OPEN_DRIVER_VERSION = (
    b"NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.133.07  Release Build  "
    b"(<builder>)  Fri Mar 14 12:44:21 UTC 2025\n"
    b"GCC version:  gcc version 12.2.0 (Debian 12.2.0-14)\n"
)


def build_gpu_information(model: str) -> bytes:
    return (
        f"Model: \t\t {model}\n"
        "IRQ:   \t\t 143\n"
        "GPU UUID: \t GPU-<uuid>\n"
        "Video BIOS: \t 95.02.18.80.5f\n"
        "Bus Type: \t PCIe\n"
        "DMA Size: \t 47 bits\n"
        "DMA Mask: \t 0x7fffffffffff\n"
        "Bus Location: \t <bus address>\n"
        "Device Minor: \t 0\n"
        "GPU Excluded:\t No\n"
    ).encode()


RTX_4090 = "NVIDIA GeForce RTX 4090"
RTX_3090 = "NVIDIA GeForce RTX 3090"
MACHINES = {
    "one GPU on the proprietary module": (
        CPU_INFORMATION,
        PROPRIETARY_DRIVER_VERSION,
        (build_gpu_information(RTX_4090),),
        HardwareFacts((CPU_MODEL,), (RTX_4090,), "550.54.14"),
    ),
    "one GPU on the open module": (
        CPU_INFORMATION,
        OPEN_DRIVER_VERSION,
        (build_gpu_information(RTX_4090),),
        HardwareFacts((CPU_MODEL,), (RTX_4090,), "570.133.07"),
    ),
    "two GPUs, sorted by their models": (
        CPU_INFORMATION,
        PROPRIETARY_DRIVER_VERSION,
        (build_gpu_information(RTX_4090), build_gpu_information(RTX_3090)),
        HardwareFacts((CPU_MODEL,), (RTX_3090, RTX_4090), "550.54.14"),
    ),
    "the driver loaded and no GPU": (
        CPU_INFORMATION,
        PROPRIETARY_DRIVER_VERSION,
        (),
        HardwareFacts((CPU_MODEL,), (), "550.54.14"),
    ),
    "no driver loaded": (CPU_INFORMATION, None, (), HardwareFacts((CPU_MODEL,))),
    "two models of CPU, each named once and sorted": (
        TWO_SOCKETS_INFORMATION,
        None,
        (),
        HardwareFacts((OTHER_CPU_MODEL, CPU_MODEL)),
    ),
}
REFUSED_FILES = {
    "a model of CPU beyond 128 characters": (
        build_processor(0, "x" * 129),
        None,
        (),
        "/proc/cpuinfo gives a model name of 129 characters, outside 1 to 128",
    ),
    "a byte outside ASCII": (
        build_processor(0, CPU_MODEL).replace(b"AMD", b"AM\xc3\x89"),
        None,
        (),
        "/proc/cpuinfo holds a character outside printable ASCII",
    ),
    "a control character": (
        build_processor(0, CPU_MODEL + "\x1b[2J"),
        None,
        (),
        "/proc/cpuinfo holds a character outside printable ASCII",
    ),
    "no model of CPU": (b"processor\t: 0\n", None, (), "/proc/cpuinfo names no model name"),
    "a driver line without a version": (
        CPU_INFORMATION,
        b"NVRM version: NVIDIA UNIX x86_64 Kernel Module\n",
        (),
        "/proc/driver/nvidia/version gives no driver version",
    ),
    "a GPU without its model": (
        CPU_INFORMATION,
        PROPRIETARY_DRIVER_VERSION,
        (b"IRQ: 143\n",),
        "the information of a GPU in /proc/driver/nvidia/gpus gives 0 models, not one",
    ),
    "an oversized file of the driver": (
        CPU_INFORMATION,
        PROPRIETARY_DRIVER_VERSION + b"x" * 2**16,
        (),
        "/proc/driver/nvidia/version holds 65676 bytes, beyond 65536",
    ),
}


class TestTheHardwareFacts:
    class TestTheReadHardwareFactsFunction:
        @pytest.mark.parametrize(
            ("cpu_information", "driver_version", "gpu_information_contents", "expected_facts"),
            list(MACHINES.values()),
            ids=list(MACHINES.keys()),
        )
        class TestWhenTheMachineIsOneOfThese:
            @pytest.fixture
            def hardware_facts(
                self,
                cpu_information: bytes,
                driver_version: bytes | None,
                gpu_information_contents: tuple[bytes, ...],
            ) -> HardwareFacts:
                return read_hardware_facts(
                    cpu_information, driver_version, gpu_information_contents
                )

            def test_should_read_the_models_and_the_drivers_version_alone(
                self, hardware_facts: HardwareFacts, expected_facts: HardwareFacts
            ) -> None:
                assert hardware_facts == expected_facts

        @pytest.mark.parametrize(
            ("cpu_information", "driver_version", "gpu_information_contents", "reason"),
            list(REFUSED_FILES.values()),
            ids=list(REFUSED_FILES.keys()),
        )
        class TestWhenAFileBreaksItsForm:
            @pytest.fixture
            def read_act(
                self,
                cpu_information: bytes,
                driver_version: bytes | None,
                gpu_information_contents: tuple[bytes, ...],
            ) -> Callable[[], HardwareFacts]:
                return partial(
                    read_hardware_facts, cpu_information, driver_version, gpu_information_contents
                )

            def test_should_refuse_it_as_outside_input(
                self, read_act: Callable[[], HardwareFacts], reason: str
            ) -> None:
                with pytest.raises(HardwareFactsError, match=re.escape(reason)):
                    read_act()

    class TestTheHasGpuProperty:
        @pytest.mark.parametrize(
            ("hardware_facts", "has_gpu"),
            [
                pytest.param(MACHINES["one GPU on the open module"][3], True, id="a GPU"),
                pytest.param(
                    MACHINES["the driver loaded and no GPU"][3], False, id="a driver alone"
                ),
                pytest.param(MACHINES["no driver loaded"][3], False, id="no driver"),
            ],
        )
        class TestWhenTheFactsAreRead:
            def test_should_hold_a_gpu_only_with_its_model_and_its_driver(
                self, hardware_facts: HardwareFacts, has_gpu: bool
            ) -> None:
                assert hardware_facts.has_gpu is has_gpu

    class TestTheIsBusAddressFunction:
        @pytest.mark.parametrize(
            ("name", "is_address"),
            [
                pytest.param("0000:01:00.0", True, id="a bus address"),
                pytest.param("0000:01:00.8", False, id="a function beyond 7"),
                pytest.param("../../etc", False, id="a path"),
                pytest.param("0000:01:00.0\n", False, id="a trailing line feed"),
            ],
        )
        class TestWhenADirectoryOfTheGpusIsListed:
            @pytest.fixture
            def is_taken(self, name: str) -> bool:
                return is_bus_address(name)

            def test_should_take_only_a_bus_address_as_a_gpus_directory(
                self, is_taken: bool, is_address: bool
            ) -> None:
                assert is_taken is is_address
