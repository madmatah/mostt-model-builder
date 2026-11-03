from collections.abc import Sequence

PERCENT = 100


class WordErrorRateError(ValueError):
    pass


def count_word_errors(reference_words: Sequence[str], hypothesis_words: Sequence[str]) -> int:
    previous_row = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            current_row.append(
                min(
                    previous_row[hypothesis_index] + 1,
                    current_row[hypothesis_index - 1] + 1,
                    previous_row[hypothesis_index - 1] + int(reference_word != hypothesis_word),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def compute_word_error_rate_percent(error_count: int, word_count: int) -> float:
    if word_count <= 0:
        raise WordErrorRateError(f"a word error rate over {word_count} words has no meaning")
    return PERCENT * error_count / word_count
