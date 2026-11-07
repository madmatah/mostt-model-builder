import json
import random
import re
from collections.abc import Callable
from functools import partial

import pytest

from mostt_model_builder.domain.containers.devices import Device
from mostt_model_builder.domain.contract.shard_request import (
    MAX_SHARD_COUNT,
    SHARD_REQUEST_MAX_SIZE_BYTES,
    ShardRequest,
    ShardRequestError,
    parse_shard_request,
    render_shard_request,
    select_shard,
)

BOTH_LANGUAGES = ("fr_fr", "en_us")
# The 102 languages of FLEURS, as the names of the directories of its data at the pinned revision.
FLEURS_LANGUAGES = (
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
)
REQUEST = ShardRequest("int8-cpu", "int8", Device.CPU, BOTH_LANGUAGES, 1, 3)
ACCEPTED_REQUESTS = {
    "the only shard of a run": ShardRequest("int8-cpu", "int8", Device.CPU, ("en_us",), 0, 1),
    "the last shard of the most shards": ShardRequest(
        "fp16-gpu", "fp16", Device.GPU, ("fr_fr",), MAX_SHARD_COUNT - 1, MAX_SHARD_COUNT
    ),
    "a run and a variant of 32 characters": ShardRequest(
        "r" * 32, "v" * 32, Device.CPU, BOTH_LANGUAGES, 0, 1
    ),
}
SHARD_REQUEST_FIELDS: dict[str, object] = {
    "run": "int8-cpu",
    "variant": "int8",
    "device": "cpu",
    "languages": ["fr_fr", "en_us"],
    "shard": 1,
    "shard_count": 3,
}


def build_request(**changes: object) -> bytes:
    return json.dumps(SHARD_REQUEST_FIELDS | changes).encode("ascii")


def build_request_without(key: str) -> bytes:
    remaining_fields = {name: field for name, field in SHARD_REQUEST_FIELDS.items() if name != key}
    return json.dumps(remaining_fields).encode("ascii")


REFUSED_REQUESTS = {
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "more than 4 KiB": (
        b" " * (SHARD_REQUEST_MAX_SIZE_BYTES + 1),
        f"the shard's request holds {SHARD_REQUEST_MAX_SIZE_BYTES + 1} bytes, too many",
    ),
    "not json": (b"{", "the shard's request is not ASCII JSON"),
    "a key twice": (
        b'{"run": "a", "run": "b"}',
        "the shard's request is not ASCII JSON: a key appears twice in an object",
    ),
    "bytes beyond ascii": (
        build_request().replace(b"int8-cpu", b"int8-cpu\xc3\xa9"),
        "the shard's request is not ASCII JSON",
    ),
    "a list": (b"[]", "the shard's request is not an object of"),
    "a missing key": (build_request_without("languages"), "is not an object of"),
    "an unknown key": (build_request(command="rm"), "is not an object of"),
    "a run outside the grammar": (
        build_request(run="Int8-CPU"),
        "its run, 'Int8-CPU', does not match [a-z0-9][a-z0-9-]{0,31}",
    ),
    "a run that is a path": (build_request(run="../int8"), "its run, '../int8', does not match"),
    "a run beyond 32 characters": (build_request(run="r" * 33), "its run, 'rrrr"),
    "a run that is not a string": (build_request(run=8), "its run, 8, does not match"),
    "a variant that starts with a dash": (
        build_request(variant="-int8"),
        "its variant, '-int8', does not match [a-z0-9][a-z0-9-]{0,31}",
    ),
    "a device neither cpu nor gpu": (
        build_request(device="tpu"),
        "its device, 'tpu', is neither cpu nor gpu",
    ),
    "a device in capitals": (build_request(device="GPU"), "its device, 'GPU', is neither"),
    "no language": (
        build_request(languages=[]),
        "its languages, [], are not distinct languages named as FLEURS does, "
        "[a-z]{2,3}(_[a-z0-9]{2,4}){1,2}",
    ),
    "a language with a dash and capitals": (
        build_request(languages=["fr_fr", "en-US"]),
        "its languages, ['fr_fr', 'en-US'], are not distinct languages named as FLEURS does",
    ),
    "a language that is a path": (
        build_request(languages=["../fr_fr"]),
        "its languages, ['../fr_fr'], are not distinct languages",
    ),
    "a language without its region": (
        build_request(languages=["fr"]),
        "its languages, ['fr'], are not distinct languages",
    ),
    "a language with one part too many": (
        build_request(languages=["cmn_hans_cn_x1"]),
        "its languages, ['cmn_hans_cn_x1'], are not distinct languages",
    ),
    "a language twice": (
        build_request(languages=["fr_fr", "fr_fr"]),
        "its languages, ['fr_fr', 'fr_fr'], are not distinct",
    ),
    "languages written as one string": (
        build_request(languages="fr_fr"),
        "its languages, 'fr_fr', are not distinct",
    ),
    "a language that is a list": (
        build_request(languages=[["fr_fr"]]),
        "its languages, [['fr_fr']], are not distinct",
    ),
    "no shard": (build_request(shard_count=0), "its count of shards, 0, lies outside 1 to 16"),
    "more shards than the bound": (
        build_request(shard_count=MAX_SHARD_COUNT + 1),
        "its count of shards, 17, lies outside 1 to 16",
    ),
    "a boolean count of shards": (
        build_request(shard_count=True),
        "its count of shards, True, lies outside 1 to 16",
    ),
    "a fractional count of shards": (
        build_request(shard_count=2.0),
        "its count of shards, 2.0, lies outside 1 to 16",
    ),
    "a negative shard": (build_request(shard=-1), "its shard, -1, lies outside 0 to 2"),
    "a shard beyond its count": (build_request(shard=3), "its shard, 3, lies outside 0 to 2"),
    "a boolean shard": (build_request(shard=False), "its shard, False, lies outside 0 to 2"),
}
PROPERTY_SEED = 20_261_103
PROPERTY_CASE_COUNT = 500
MAX_ENTRY_COUNT = 100


def build_sharding_cases(generator: random.Random) -> list[tuple[int, int]]:
    return [
        (generator.randint(0, MAX_ENTRY_COUNT), generator.randint(1, MAX_SHARD_COUNT))
        for _ in range(PROPERTY_CASE_COUNT)
    ]


class TestTheShardRequest:
    class TestTheParseShardRequestFunction:
        class TestWhenTheCoreWroteTheRequestOfAShard:
            @pytest.fixture
            def shard_request(self) -> ShardRequest:
                return parse_shard_request(render_shard_request(REQUEST))

            def test_should_read_the_request_it_rendered(self, shard_request: ShardRequest) -> None:
                assert shard_request == REQUEST

        @pytest.mark.parametrize(
            "request_at_a_bound",
            list(ACCEPTED_REQUESTS.values()),
            ids=list(ACCEPTED_REQUESTS.keys()),
        )
        class TestWhenTheRequestLiesAtABound:
            @pytest.fixture
            def shard_request(self, request_at_a_bound: ShardRequest) -> ShardRequest:
                return parse_shard_request(render_shard_request(request_at_a_bound))

            def test_should_read_it(
                self, shard_request: ShardRequest, request_at_a_bound: ShardRequest
            ) -> None:
                assert shard_request == request_at_a_bound

        @pytest.mark.parametrize("language", FLEURS_LANGUAGES)
        class TestWhenTheLanguageIsAnyOfFleurs:
            @pytest.fixture
            def shard_request(self, language: str) -> ShardRequest:
                return parse_shard_request(build_request(languages=[language]))

            def test_should_read_it_so_that_a_step_needs_no_list_of_the_bench_s_languages(
                self, shard_request: ShardRequest, language: str
            ) -> None:
                assert shard_request.languages == (language,)

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REQUESTS.values()),
            ids=list(REFUSED_REQUESTS.keys()),
        )
        class TestWhenTheRequestIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], ShardRequest]:
                return partial(parse_shard_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], ShardRequest], reason: str
            ) -> None:
                with pytest.raises(ShardRequestError, match=re.escape(reason)):
                    parse_act()

    class TestTheSelectShardFunction:
        class TestWhenSevenEntriesAreDealtInThreeShards:
            @pytest.fixture
            def second_shard(self) -> list[str]:
                return select_shard(["a", "b", "c", "d", "e", "f", "g"], 1, 3)

            def test_should_deal_the_entries_in_turn(self, second_shard: list[str]) -> None:
                assert second_shard == ["b", "e"]

        class TestWhenGivenAnyCountOfEntriesAndOfShards:
            def test_should_place_every_entry_in_exactly_one_shard(self) -> None:
                generator = random.Random(PROPERTY_SEED)
                cases = build_sharding_cases(generator)
                mismatches = [
                    (entry_count, shard_count)
                    for entry_count, shard_count in cases
                    if sorted(
                        entry
                        for shard in range(shard_count)
                        for entry in select_shard(range(entry_count), shard, shard_count)
                    )
                    != list(range(entry_count))
                ]
                assert mismatches == [], f"seed {PROPERTY_SEED}"
                assert len(cases) == PROPERTY_CASE_COUNT, f"seed {PROPERTY_SEED}"

            def test_should_deal_shards_whose_sizes_differ_by_at_most_one(self) -> None:
                generator = random.Random(PROPERTY_SEED)
                cases = build_sharding_cases(generator)
                shard_sizes = [
                    [
                        len(select_shard(range(entry_count), shard, shard_count))
                        for shard in range(shard_count)
                    ]
                    for entry_count, shard_count in cases
                ]
                mismatches = [
                    (case, sizes)
                    for case, sizes in zip(cases, shard_sizes, strict=True)
                    if max(sizes) - min(sizes) > 1
                ]
                assert mismatches == [], f"seed {PROPERTY_SEED}"
                assert len(cases) == PROPERTY_CASE_COUNT, f"seed {PROPERTY_SEED}"
