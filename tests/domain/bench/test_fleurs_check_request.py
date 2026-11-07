import re
from collections.abc import Callable
from functools import partial
from pathlib import PurePosixPath

import pytest

from mostt_model_builder.domain.bench.fleurs_check_request import (
    BenchFiles,
    FleursCheckRequest,
    FleursCheckRequestError,
    parse_fleurs_check_request,
    render_fleurs_check_request,
)

INDEX = PurePosixPath("/sources/google/fleurs/70bb2e84/data/fr_fr/test.tsv")
ARCHIVE = PurePosixPath("/sources/google/fleurs/70bb2e84/data/fr_fr/audio/test.tar.gz")
REQUEST = FleursCheckRequest((BenchFiles("fr_fr", INDEX, ARCHIVE),))
FRENCH_ENTRY = f'{{"language": "fr_fr", "index": "{INDEX}", "archive": "{ARCHIVE}"}}'
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
REFUSED_REQUESTS = {
    "not json": (b"[", "the request is not JSON"),
    "a document nested 3 000 deep": (b"[" * 3000, "the document nests deeper than 32 levels"),
    "an unknown key": (b'{"languages": [], "x": 1}', "not an object of ['languages']"),
    "no language": (b'{"languages": []}', "the request names no language"),
    "a language with a dash and capitals": (
        f'{{"languages": [{FRENCH_ENTRY.replace("fr_fr", "fr-FR", 1)}]}}'.encode(),
        "'fr-FR' does not name a language as FLEURS does, [a-z]{2,3}(_[a-z0-9]{2,4}){1,2}",
    ),
    "a language that is a path": (
        f'{{"languages": [{FRENCH_ENTRY.replace("fr_fr", "../fr_fr", 1)}]}}'.encode(),
        "'../fr_fr' does not name a language as FLEURS does",
    ),
    "a language that is a number": (
        f'{{"languages": [{FRENCH_ENTRY.replace('"fr_fr"', "7", 1)}]}}'.encode(),
        "7 does not name a language as FLEURS does",
    ),
    "a language twice": (
        f'{{"languages": [{FRENCH_ENTRY}, {FRENCH_ENTRY}]}}'.encode(),
        "names a language twice",
    ),
    "a path outside the sources": (
        f'{{"languages": [{FRENCH_ENTRY.replace(str(INDEX), "/etc/passwd")}]}}'.encode(),
        "'/etc/passwd' is not a path under /sources",
    ),
    "a path that climbs": (
        f'{{"languages": [{FRENCH_ENTRY.replace(str(INDEX), "/sources/../etc/passwd")}]}}'.encode(),
        "is not a path under /sources",
    ),
}


class TestTheFleursCheckRequest:
    class TestTheParseFleursCheckRequestFunction:
        class TestWhenTheHostWroteIt:
            @pytest.fixture
            def request_(self) -> FleursCheckRequest:
                return parse_fleurs_check_request(render_fleurs_check_request(REQUEST))

            def test_should_read_each_languages_files(self, request_: FleursCheckRequest) -> None:
                assert request_ == REQUEST

        @pytest.mark.parametrize("language", FLEURS_LANGUAGES)
        class TestWhenTheLanguageIsAnyOfFleurs:
            @pytest.fixture
            def request_(self, language: str) -> FleursCheckRequest:
                entry = FRENCH_ENTRY.replace("fr_fr", language, 1)
                return parse_fleurs_check_request(f'{{"languages": [{entry}]}}'.encode())

            def test_should_read_it_so_that_the_step_needs_no_list_of_the_bench_s_languages(
                self, request_: FleursCheckRequest, language: str
            ) -> None:
                assert request_.languages[0].language == language

        @pytest.mark.parametrize(
            ("content", "reason"),
            list(REFUSED_REQUESTS.values()),
            ids=list(REFUSED_REQUESTS.keys()),
        )
        class TestWhenTheRequestIsMalformed:
            @pytest.fixture
            def parse_act(self, content: bytes) -> Callable[[], FleursCheckRequest]:
                return partial(parse_fleurs_check_request, content)

            def test_should_refuse_it_and_say_why(
                self, parse_act: Callable[[], FleursCheckRequest], reason: str
            ) -> None:
                with pytest.raises(FleursCheckRequestError, match=re.escape(reason)):
                    parse_act()
