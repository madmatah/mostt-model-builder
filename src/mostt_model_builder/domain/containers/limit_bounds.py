# Docker reads 0 or -1 as no limit, and refuses a memory limit under 6 MiB; the ceilings only
# refuse a value that no step could mean on the machines a build runs on.
MIN_MEMORY_MIB = 6
MAX_MEMORY_MIB = 57_344
MIN_PROCESS_COUNT = 1
MAX_PROCESS_COUNT = 8192
MIN_CPU_COUNT = 1
MAX_CPU_COUNT = 64
MIN_TEMPORARY_FILESYSTEM_MIB = 1
MAX_TEMPORARY_FILESYSTEM_MIB = 16_384
MIN_OUTPUT_SIZE_BYTES = 1
MAX_OUTPUT_SIZE_BYTES = 8 * 2**30
