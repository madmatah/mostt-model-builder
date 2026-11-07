import hashlib

BLOB_HEADER_ENCODING = "ascii"


# git names a file's bytes by the SHA-1 of a header and the bytes, which the Hub lists as a file's
# `oid`. The comparison it serves only advises the maintainer before a publication, whose container
# compares SHA-256 digests, so SHA-1 is declared as serving no security.
def compute_git_blob_id(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode(BLOB_HEADER_ENCODING)
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()
