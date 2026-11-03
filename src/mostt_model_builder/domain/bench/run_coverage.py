from collections.abc import Collection, Mapping, Sequence

from mostt_model_builder.domain.bench.hypotheses import Hypothesis


class RunCoverageError(ValueError):
    pass


def judge_run_coverage(
    run_name: str,
    hypotheses: Sequence[Hypothesis],
    languages: Sequence[str],
    utterance_ids_by_language: Mapping[str, Collection[str]],
) -> None:
    where = f"the run {run_name}"
    seen: set[tuple[str, str]] = set()
    for hypothesis in hypotheses:
        key = (hypothesis.language, hypothesis.utterance_id)
        if hypothesis.language not in languages:
            raise RunCoverageError(
                f"{where} holds a hypothesis in {hypothesis.language!a}, not one of its languages"
            )
        if hypothesis.utterance_id not in utterance_ids_by_language[hypothesis.language]:
            raise RunCoverageError(
                f"{where} holds {hypothesis.utterance_id!a} of {hypothesis.language}, which the "
                "index of the bench does not hold"
            )
        if key in seen:
            raise RunCoverageError(
                f"{where} holds {hypothesis.utterance_id} of {hypothesis.language} twice, across "
                "its shards"
            )
        seen.add(key)
    for language in languages:
        missing = [
            utterance_id
            for utterance_id in utterance_ids_by_language[language]
            if (language, utterance_id) not in seen
        ]
        if missing:
            raise RunCoverageError(
                f"{where} lacks {len(missing)} utterances of {language}, such as {missing[0]}"
            )
