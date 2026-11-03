import re

SUFFIX_SEPARATOR = "."
# Ported from mostt's model store, whose installer reads every published file under this name,
# so that a name holds no path, no leading dot or dash, and nothing a shell or Docker reads as
# syntax.
FILE_NAME_GRAMMAR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
# Ported from mostt's model store, so that every published file passes it: a file never holds
# code, an archive, or a format that runs code when it loads. `.bin` is among them since
# `pytorch_model.bin` is a pickle.
REFUSED_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
        ".pickle",
        ".nemo",
        ".bin",
        ".joblib",
        ".npy",
        ".npz",
        ".py",
        ".pyc",
        ".pyo",
        ".so",
        ".sh",
        ".exe",
        ".dll",
        ".dylib",
        ".jar",
        ".zip",
        ".whl",
        ".egg",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".zst",
        ".7z",
    }
)


def find_refused_suffixes(name: str, refused_suffixes: frozenset[str]) -> tuple[str, ...]:
    suffixes = name.lower().split(SUFFIX_SEPARATOR)[1:]
    return tuple(
        SUFFIX_SEPARATOR + suffix
        for suffix in suffixes
        if SUFFIX_SEPARATOR + suffix in refused_suffixes
    )


def is_file_name(name: str) -> bool:
    return FILE_NAME_GRAMMAR.fullmatch(name) is not None
