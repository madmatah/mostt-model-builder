---
paths:
  - "tests/**"
  - "recipes/*/tests/**"
  - "pyproject.toml"
---

# Tests

Tests are the specification of the business rules: to learn how a part of the system behaves, read its tests.

## Structure

Every test file follows one structure, so that a rule reads the same way everywhere, and its arrangement, its act and its assertions sit in known places. Its words are those of pytest: a group is a class, nested in its parent, a test is a method, a hook that runs before each test is a fixture of function scope, and a hook that runs once before all of them is a fixture of class scope.

- **Level 1 names the component the file tests,** in domain vocabulary, as the file is named: `TestTheWordErrorRate`, in the tests of the word error rate. A file tests one component; a file of integration or scenario tests may hold several subjects, one level-1 group each.
- **Level 2 names the operation whose rule the tests state:** the function, the method or the constant under test, such as `TestTheWordErrorRateFunction`. Level 2 is there even when the file tests a single function.
  - When the act is a sequence of calls, level 2 names the last one, whose rule the tests state, and the earlier calls are arrangement.
  - When no operation of the component owns the rule, such as a step that a container ends early or a file that a container writes, or when the rule spans several of its operations, level 2 is the situation: `TestWhenTheStepRunsOutOfMemory`.
- **Above the unit level, level 1 names the subject,** such as `TestTheBuildWithAContainerThatWritesALink`, and level 2 the situation.
- **Situations are nested groups named `TestWhen…`,** at most three under level 2, so that the sentence they make stays readable.
- **Hooks arrange and act; each test only asserts.** Teardown aside, a group holds its arrangement and its act in a single fixture. A test may read what the act produced, and project it with a pure query (a map, a filter, a computation that calls nothing under test and changes nothing), but never calls what its group tests, except to assert a refusal or inside a property run, below. An assertion lives in a test only, never in a fixture or in a helper: a helper computes, and the test asserts. A constant under test needs no fixture: its test reads it.
- **What the act produces is named for what it is:** `error_rate`, `source_pin`, never `result`, `input` or `data`, as the fixture that produces it.
- **One expectation per test;** a few tightly related ones are allowed, such as the two bounds of a range. One comparison over an ordered sequence or a composite record is one expectation, and stays one when the order or the combination is the rule.
- **A costly act runs once,** in a fixture of class scope, when its tests only read what it produced: a process started, thousands of iterations, a large file parsed. The act captures, as plain copies, every value its tests read, and nothing under it changes what it produced. A copy lets the act be freed, where a cache, a closure or a fixture that keeps the whole act holds it for the whole run, once per parameter of a property test, until memory runs out. CI can run tests several times slower than a contributor's machine, so where an act has a timeout, such as a process it starts, that timeout is at least twice the act's duration on CI.
- **A sequence of steps nests:** each `TestWhen…` group continues from the state its parent left.
  - A cheap sequence continues in fixtures of function scope, each level adding one step.
  - A costly sequence runs whole in one fixture of class scope, which captures a copy at each step, and its nested groups hold no fixture and read those captures.
  - A chain never mixes the two: a fixture of class scope builds only on the ancestors' fixtures of class scope, since it runs before an ancestor's fixture of function scope: it builds on what an earlier test left, or on nothing, and the ancestor's fixture then rebuilds that state under every test.
- **A value read in the middle of a sequence is captured where it happens,** before the next step changes it, so that a test compares two moments and not the final state with itself.
- **A comparison runs both acts in the same fixture,** each product named for what it is (`float16_error_rate`, `float32_error_rate`), as distinct objects.
- **A refusal names its reason.** When the refusal is raised, as a Python exception, the fixture keeps the act as a function, and the test asserts that calling it fails with the expected error and its message or pattern, so that a failure for another reason fails the test. When it is returned as a value, the act captures it like any product, and the test asserts its type, its value or its message.
- **A property keeps its whole run inside one test:** its generators, its act and its expectation stay in one property run, under a `TestWhenGivenAny…` group that names the generated domain.
- **A check over a collection** gathers the mismatches and asserts that there are none, next to an expectation on the count of what it checked, so that an empty collection fails.
- **Tables:**
  - a parameterized test only for a simple table of data that no act produces, such as the entries of a constant, inside a `TestWhen…` group, each row named so that it states its own rule;
  - a table whose rows each need an act is a parameterized group, with the act in its fixture, since a parameterized test would call the unit in the test;
  - a table of subjects is a parameterized group at level 2;
  - a table built from production data covers a new entry by itself, where a hand-written list would not.
- **Shared data that nothing changes** lives in the narrowest group that holds every test reading it, or at the file's top level when that group is the file's only level 1. An object that an act mutates or writes into is built in each fixture of function scope.
- **Teardown** lives in the group whose fixture set up what it releases, after the fixture's `yield`. The file's top level holds only imports, constant data and helpers that build or compute; everything a test runs happens in its fixtures and its test.
- **A benchmark keeps the form its tool measures,** since the tool measures what runs inside its measured call: this structure does not apply to it.

For example:

```python
import pytest

from mostt_model_builder.domain.word_error_rate import word_error_rate

REFERENCE = "the cat sleeps here"
HYPOTHESIS_WITHOUT_ITS_LAST_WORD = "the cat sleeps"


class TestTheWordErrorRate:
    class TestTheWordErrorRateFunction:
        class TestWhenTheHypothesisDropsOneWordOfFour:
            @pytest.fixture
            def error_rate(self) -> float:
                return word_error_rate(REFERENCE, HYPOTHESIS_WITHOUT_ITS_LAST_WORD)

            def test_should_count_one_deletion_out_of_four_words(self, error_rate: float) -> None:
                assert error_rate == pytest.approx(0.25)
```

## Writing a test

- **A test name is a sentence that states one rule**, in domain vocabulary, starts with `should`, after the `test_` that pytest needs to collect it, and keeps the reason the rule has when its groups do not give it. With the names of its groups, it reads as one sentence: `the word error rate > the word_error_rate function > when the hypothesis drops one word of four > should count one deletion out of four words`.
- **One rule per test.** Neither a test nor its fixtures branch, and every expectation can fail: an expectation on a value that the test built itself checks nothing. Inputs are named values, not unexplained literals.
- **Every test asserts:** the gate fails a test that runs no assertion, wherever the test framework can tell, so that a test whose act never reaches its expectation does not pass.
- **No test is committed focused or skipped:** either one silently takes tests out of the gate. A conditional skip that [pytest](#pytest) allows, with its reason, is the only exception.
- **A bug fix starts with a test** that reproduces the bug.
- **Floating point:** compare with a tolerance, unless bit-exactness is the rule under test.
- **No test depends on timing, the wall clock or the order of the tests,** and randomness is seeded, or reports its seed, so that a failure replays.
- **A deleted test, or a loosened expectation, states its reason** in the change, and so does a dropped refusal message or a weaker assertion.

## Levels

- **Unit tests** live in `tests/`, mirroring the package (`tests/domain/` for `src/mostt_model_builder/domain/`), and, for a recipe, in its own `recipes/<family>/tests/`. They need no network, no Docker and no model file. Every commit runs them, through the gate.
- **Tests of the package as a whole,** such as the one that imports it under an isolated Python, live at the top of `tests/`, and run with the unit tests.
- **A test that needs a real adapter** has a level of its own, apart from the unit tests, so that the unit tests stay fast and need no outside resource. A build on the real containers, images and model files, the one test of that kind, runs by hand, outside the gate.

## Test support

- **A fake lives next to the port it stands in for,** and refuses what the real adapter refuses, so that the code under test behaves the same on it as on the real adapter.
- **Only tests import test support.** It lives in the tests' own directories, `tests/` and each recipe's `tests/`, which no shipped code can import, since neither is on the path of the packages. Review keeps it out of the shipped code.
- **A test that runs git in a repository of its own drops the `GIT_` variables it inherits:** `git rebase --exec`, which checks each commit of a branch, exports `GIT_DIR` when it runs in a worktree, and the test's git would then work on the repository under rebase instead of its own.

## pytest

- **Groups are nested classes:** `TestThe<Component>`, then `TestThe<Function>`, then `TestWhen<Situation>`, and each test is a method named `test_should_<rule>`. Check: review.
- **Hooks are fixtures:** of function scope for an act that runs before each test, of class scope for a costly act that runs once. Check: review.
- **A fixture of class scope is a class method,** `@pytest.fixture(scope="class")` above `@classmethod`, since pytest 9 deprecates one defined on the instance, and warnings are errors. Check: the gate.
- **A refusal is asserted with `pytest.raises(<error>, match=<pattern>)`** on the act the fixture returns as a function, so that a refusal for another reason fails. Check: the gate (`ruff`'s `PT011` refuses a bare `pytest.raises` of a broad exception), review for the others.
- **Markers are strict and warnings are errors:** `--strict-markers`, `--strict-config` and `filterwarnings = ["error"]` in `pyproject.toml`, so that a misspelled marker or a deprecation fails the run. Check: the gate.
- **pytest cannot tell a test that runs no assertion,** so review checks that each test asserts. Check: review.
- **No skip is allowed yet:** the gate runs every test. A conditional skip comes with the test that needs it, and its reason, in this section. Check: review.
