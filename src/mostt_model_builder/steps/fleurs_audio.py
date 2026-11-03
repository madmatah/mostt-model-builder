import tarfile
from collections.abc import Collection, Iterator
from pathlib import Path

from mostt_model_builder.domain.bench.fleurs_index import locate_archive_member

# FLEURS's recordings weigh under 2 MiB; the bound only keeps a member that no recording of the
# bench could be from filling the container's memory.
MAX_RECORDING_SIZE_BYTES = 64 * 2**20


class FleursAudioError(Exception):
    pass


def read_recordings(
    archive_path: Path, recording_names: Collection[str]
) -> Iterator[tuple[str, bytes]]:
    wanted_members = {locate_archive_member(name): name for name in recording_names}
    found_names: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            recording_name = wanted_members.get(member.name)
            if recording_name is None:
                continue
            if recording_name in found_names:
                raise FleursAudioError(f"the archive holds {member.name} twice")
            if not member.isfile() or member.size > MAX_RECORDING_SIZE_BYTES:
                raise FleursAudioError(
                    f"the member {member.name} is not a regular file of at most "
                    f"{MAX_RECORDING_SIZE_BYTES} bytes"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise FleursAudioError(f"the member {member.name} cannot be read")
            found_names.add(recording_name)
            yield recording_name, stream.read()
    missing = sorted(set(recording_names) - found_names)
    if missing:
        raise FleursAudioError(
            f"{len(missing)} recordings of the index are missing from {archive_path.name}, "
            f"such as {missing[0]}"
        )
