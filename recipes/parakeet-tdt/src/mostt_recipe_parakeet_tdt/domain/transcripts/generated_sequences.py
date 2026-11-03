from collections.abc import Sequence


class GeneratedSequenceError(ValueError):
    pass


def read_generated_token_ids(sequence: Sequence[int], blank_id: int) -> tuple[int, ...]:
    token_ids = []
    for token_id in sequence:
        if token_id > blank_id:
            raise GeneratedSequenceError(
                f"the generated sequence holds the id {token_id}, above the blank {blank_id}"
            )
        if token_id < blank_id:
            token_ids.append(token_id)
    return tuple(token_ids)
