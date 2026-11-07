from mostt_model_builder.domain.bench.bench_languages import BENCH_LANGUAGES, LANGUAGE_NAMES


class TestTheBenchLanguages:
    class TestTheLanguageNames:
        class TestWhenReadForEveryLanguageOfTheBench:
            def test_should_check_at_least_one_language(self) -> None:
                assert len(BENCH_LANGUAGES) > 0

            def test_should_name_each_one_so_that_the_card_can_give_its_rates(self) -> None:
                unnamed = [
                    language for language in BENCH_LANGUAGES if language not in LANGUAGE_NAMES
                ]
                assert unnamed == []
