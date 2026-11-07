RUN_LINES_FILE_NAME = "lines.jsonl"
# A run writes one line per utterance of FLEURS's test sets, under 1 000 per language, each a few
# hundred bytes; the bound only refuses lines that no run of the bench could write.
RUN_LINES_MAX_SIZE_BYTES = 16 * 2**20
