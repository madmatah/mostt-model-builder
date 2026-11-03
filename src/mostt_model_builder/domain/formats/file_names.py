SUFFIX_SEPARATOR = "."
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
