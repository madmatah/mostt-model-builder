import re
import unicodedata

# The normalization of the 2026-11-02 measurements, which compared engines on FLEURS's
# transcription column, with the opening quotation mark of English mapped as well: FLEURS en writes
# it twice where an apostrophe stands.
TYPOGRAPHIC_APOSTROPHES = ("\N{RIGHT SINGLE QUOTATION MARK}", "\N{LEFT SINGLE QUOTATION MARK}")
APOSTROPHE = "'"
WORD_SEPARATOR = " "
NON_WORD_RUN = re.compile(r"[^\w']+")
OUTER_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")


def normalize_words(text: str) -> tuple[str, ...]:
    folded = unicodedata.normalize("NFKC", text).lower()
    for typographic_apostrophe in TYPOGRAPHIC_APOSTROPHES:
        folded = folded.replace(typographic_apostrophe, APOSTROPHE)
    spaced = NON_WORD_RUN.sub(WORD_SEPARATOR, folded)
    return tuple(OUTER_APOSTROPHE.sub(WORD_SEPARATOR, spaced).split())
