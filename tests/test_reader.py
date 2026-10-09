import mmap
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np
import pytest
import umbi.umb
from bitarray import bitarray
from conftest import doubles, index, rational, u32, u64

import tinyumbi

FIXTURES = Path(__file__).parent / "fixtures" / "umbi"
REFERENCE_FIXTURES = sorted(FIXTURES.glob("*.umb"))
if not REFERENCE_FIXTURES:
    raise RuntimeError(f"Reference UMB fixtures are missing from {FIXTURES}")


def plain(values, n=None):
    if values is None:
        return None
    if isinstance(values, bitarray):
        return list(values[:n])
    if isinstance(values, np.ndarray) and values.dtype.fields:

        def normalized(pair):
            if isinstance(pair[0], tuple):
                return tuple(normalized(endpoint) for endpoint in pair)
            num, den = pair
            divisor = gcd(num, den)
            return num // divisor, den // divisor

        return [normalized(tinyumbi.rational_at(values, i)) for i in range(len(values))]
    return [
        tuple(v.tolist())
        if isinstance(v, np.ndarray)
        else tuple(plain([v.left, v.right]))
        if hasattr(v, "left")
        else (v.numerator, v.denominator)
        if isinstance(v, Fraction)
        else v
        for v in values
    ]


def strings(table):
    return [tinyumbi.string_at(table, i) for i in range(len(table["offsets"]) - 1)]


@pytest.mark.parametrize("path", REFERENCE_FIXTURES, ids=lambda p: p.name)
def test_reference_fixtures(path):
    ref = umbi.umb.read(path)
    model = tinyumbi.read(path)
    assert_reference_matches(model, ref)


@pytest.mark.parametrize("bits", [128, 256, 512])
@pytest.mark.parametrize("interval", [False, True])
def test_rational_reference_agreement(archive, bits, interval):
    kind = "rational-interval" if interval else "rational"
    idx = index()
    idx["annotations"] = {
        "custom": {
            "r": {"type": {"type": kind, "size": bits * (2 if interval else 1)}, "applies-to": ["states"]}
        }
    }
    pairs = [(-2, 6), (0, 3), (3, 3)]
    data = b"".join(rational(num, den, bits // 16) * (2 if interval else 1) for num, den in pairs)
    path = archive(idx, {"annotations/custom/r/states/values.bin": data})
    model = tinyumbi.read(path)
    assert_reference_matches(model, umbi.umb.read(path))
    values = model["annotations"]["custom"]["r"]["states"]["values"]
    expected = [(pair, pair) for pair in pairs] if interval else pairs
    assert [tinyumbi.rational_at(values, i) for i in range(3)] == expected


def assert_reference_matches(model, ref):
    """Compare decoded reader output, also used for live observatory exports."""
    assert type(model) is dict
    for name in (
        "state_is_initial",
        "state_to_player",
        "state_is_markovian",
        "state_to_exit_rate",
        "branch_to_probability",
    ):
        expected = getattr(ref, name)
        if expected is not None:
            assert plain(model[name], ref.index.transition_system.num_states) == plain(expected)
    for name in ("state_to_choices", "choice_to_branches", "branch_to_target"):
        expected = getattr(ref, name)
        if expected is not None:
            assert list(model[name]) == expected
    for group, items in (ref.annotations or {}).items():
        for name, entities in items.items():
            for entity, values in entities.items():
                assert plain(model["annotations"][group][name][entity]["values"], len(values)) == plain(
                    values
                )
    for entity, values in (ref.valuations or {}).items():
        for i, expected in enumerate(values):
            actual = tinyumbi.valuation_at(model["valuations"][entity], i)
            assert actual.keys() == expected.keys()
            for name, value in expected.items():
                if isinstance(value, Fraction):
                    num, den = actual[name]
                    assert num * value.denominator == value.numerator * den
                else:
                    assert actual[name] == value
    for entity, prefix in (("choices", "choice"), ("branches", "branch")):
        if entity in model["actions"]:
            action = model["actions"][entity]
            assert list(action["values"]) == getattr(ref, f"{prefix}_to_{prefix}_action")
            expected = getattr(ref, f"{prefix}_action_to_string")
            assert (None if action["strings"] is None else strings(action["strings"])) == expected


@pytest.mark.parametrize("compression", ["", "gz", "xz", "bz2"])
@pytest.mark.parametrize("prefix", ["", "./", "/"])
def test_compression_lifetime_and_zero_copy(archive, compression, prefix):
    path = archive(
        index(**{"#initial-states": 1}),
        {
            "state-is-initial.bin": b"\x01" + b"\0" * 7,
            "branch-to-target.bin": u64([2, 1, 0]),
        },
        compression,
        prefix,
        index_first=False,
    )
    model = tinyumbi.read(path)
    targets, bits = model["branch_to_target"], model["state_is_initial"]
    assert not targets.flags.writeable
    assert not targets.flags.owndata
    assert isinstance(targets.base.base.obj, mmap.mmap)
    assert bits.buffer_info().imported
    assert bits.buffer_info().readonly
    assert bits.count() == 1
    del model
    assert list(targets) == [2, 1, 0]
    assert bits[0]


def test_defaults_are_constant_space(archive):
    model = tinyumbi.read(archive(index(n=10**12)))
    assert isinstance(model["state_to_choices"], range)
    assert isinstance(model["choice_to_branches"], range)
    assert model["branch_to_target"].strides == (0,)
    assert model["state_is_initial"].strides == (0,)
    assert model["branch_to_target"][-1] == 0
    assert model["branch_to_probability"] is None
    assert model["state_is_markovian"] is None
    assert model["observations"] is None
    assert model["annotations"] == model["valuations"] == model["actions"] == {}


def test_transitions_and_game(archive):
    idx = index(
        **{
            "time": "urgent-stochastic",
            "#players": 2,
            "player-names": ["a", "b"],
            "#choices": 4,
            "#branches": 5,
            "#branch-actions": 2,
            "branch-probability-type": {"type": "double"},
            "exit-rate-type": {"type": "rational", "size": 256},
        }
    )
    members = {
        "state-to-choices.bin": u64([0, 2, 3, 4]),
        "choice-to-branches.bin": u64([0, 2, 3, 4, 5]),
        "branch-to-target.bin": u64([1, 2, 0, 1, 2]),
        "branch-to-probability.bin": doubles([0.5, 0.5, 1, 1, 1]),
        "state-to-player.bin": u32([0, 1, 0]),
        "state-is-markovian.bin": b"\x04" + b"\0" * 7,
        "state-to-exit-rate.bin": b"".join(rational(v, 3, 16) for v in [1, 2, 2**100]),
        "actions/branches/values.bin": u32([0, 1, 0, 1, 0]),
    }
    model = tinyumbi.read(archive(idx, members))
    assert list(model["state_to_choices"]) == [0, 2, 3, 4]
    assert list(model["choice_to_branches"]) == [0, 2, 3, 4, 5]
    assert model["state_is_markovian"][2]
    assert list(model["state_to_player"]) == [0, 1, 0]
    assert plain(model["state_to_exit_rate"]) == [(v, 3) for v in [1, 2, 2**100]]
    assert model["actions"]["branches"]["strings"] is None
    assert list(model["actions"]["branches"]["values"]) == [0, 1, 0, 1, 0]


@pytest.mark.parametrize(
    "name,size,data,expected",
    [
        ("int", 64, np.array([-2, 0, 2**62], dtype="<i8").tobytes(), [-2, 0, 2**62]),
        ("uint", 64, u64([0, 2, 2**64 - 1]), [0, 2, 2**64 - 1]),
        ("double", 64, doubles([-2.5, 0, 2.5]), [-2.5, 0, 2.5]),
        ("bool", 1, b"\x05" + b"\0" * 7, [True, False, True]),
        (
            "int-interval",
            128,
            np.array([[-2, 1], [0, 0], [2, 5]], dtype="<i8").tobytes(),
            [(-2, 1), (0, 0), (2, 5)],
        ),
        ("uint-interval", 128, u64([0, 1, 0, 0, 2, 5]), [(0, 1), (0, 0), (2, 5)]),
        ("double-interval", 128, doubles([-2.5, 1.5, 0, 0, 2, 5]), [(-2.5, 1.5), (0, 0), (2, 5)]),
        (
            "rational",
            128,
            rational(-2, 3) + rational(0, 1) + rational(2, 5),
            [(-2, 3), (0, 1), (2, 5)],
        ),
        (
            "rational",
            512,
            rational(-(2**200), 3, 32) + rational(0, 1, 32) + rational(2, 5, 32),
            [(-(2**200), 3), (0, 1), (2, 5)],
        ),
        (
            "rational-interval",
            512,
            b"".join(rational(v, 3, 16) for v in [-2, 1, 0, 0, 2, 5]),
            [((-2, 3), (1, 3)), ((0, 1), (0, 1)), ((2, 3), (5, 3))],
        ),
    ],
)
def test_annotation_types(archive, name, size, data, expected):
    idx = index()
    idx["annotations"] = {"custom": {"a": {"type": {"type": name, "size": size}, "applies-to": ["states"]}}}
    model = tinyumbi.read(archive(idx, {"annotations/custom/a/states/values.bin": data}))
    values = model["annotations"]["custom"]["a"]["states"]["values"]
    assert plain(values, 3) == expected
    if not isinstance(values, bitarray):
        assert plain(values[::-1]) == expected[::-1]


def test_string_dictionary_and_defaults(archive):
    idx = index(**{"#choice-actions": 2, "#observations": 2, "observations-apply-to": "states"})
    idx["annotations"] = {
        "custom": {
            "label": {
                "type": {"type": "string"},
                "#strings": 3,
                "applies-to": ["states", "choices", "branches", "observations", "players"],
            }
        }
    }
    members = {
        "actions/choices/string-mapping.bin": u64([0, 0, 4]),
        "actions/choices/strings.bin": "🌳".encode(),
    }
    for entity in idx["annotations"]["custom"]["label"]["applies-to"]:
        path = f"annotations/custom/label/{entity}"
        members[f"{path}/string-mapping.bin"] = u64([0, 0, 4, 6])
        members[f"{path}/strings.bin"] = "🌳é".encode()
        if entity in {"states", "choices", "branches"}:
            members[f"{path}/values.bin"] = u64([2, 1, 2])
    model = tinyumbi.read(archive(idx, members))
    annotation = model["annotations"]["custom"]["label"]
    state = annotation["states"]
    assert [tinyumbi.string_at(state["strings"], int(i)) for i in state["values"]] == ["é", "🌳", "é"]
    assert list(annotation["observations"]["values"]) == [0, 0]
    assert list(annotation["players"]["values"]) == [0]
    assert list(model["actions"]["choices"]["values"]) == [0, 0, 0]
    assert strings(model["actions"]["choices"]["strings"]) == ["", "🌳"]
    assert list(model["observations"]["values"]) == [0, 0, 0]


@pytest.mark.parametrize("entity", ["states", "branches"])
@pytest.mark.parametrize("prob_type", ["double", "rational", "double-interval", "rational-interval"])
def test_stochastic_observations(archive, entity, prob_type):
    idx = index(
        **{
            "#observations": 3,
            "observations-apply-to": entity,
            "observation-probability-type": {"type": prob_type},
        }
    )
    p = [0.25, 0.75, 1, 1]
    if prob_type.startswith("double"):
        data = doubles([x for v in p for x in ([v, v] if prob_type.endswith("interval") else [v])])
    else:
        data = b"".join(
            rational(v, 4) for v in [1, 3, 4, 4] for _ in range(2 if prob_type.endswith("interval") else 1)
        )
    model = tinyumbi.read(
        archive(
            idx,
            {
                f"observations/{entity}/distribution-mapping.bin": u64([0, 2, 3, 4]),
                f"observations/{entity}/values.bin": u64([0, 1, 2, 1]),
                f"observations/{entity}/probabilities.bin": data,
            },
        )
    )
    obs = model["observations"]
    start, end = obs["offsets"][:2]
    assert list(obs["values"][start:end]) == [0, 1]
    expected = [(1, 4), (3, 4)] if prob_type.startswith("rational") else p[:2]
    if prob_type.endswith("interval"):
        expected = [(v, v) for v in expected]
    assert plain(obs["probabilities"][start:end]) == expected


@pytest.mark.parametrize("kind", ["int", "bool", "string"])
def test_stochastic_annotations(archive, kind):
    idx = index()
    desc = {
        "type": {"type": kind},
        "applies-to": ["states"],
        "probability-type": {"type": "double"},
        "#probabilities": 4,
    }
    idx["annotations"] = {"custom": {"a": desc}}
    path = "annotations/custom/a/states"
    members = {f"{path}/distribution-mapping.bin": u64([0, 2, 2, 4])}
    if kind == "bool":
        members[f"{path}/values.bin"] = b"\x05" + b"\0" * 7
        expected = [True, False]
    elif kind == "int":
        members[f"{path}/values.bin"] = np.array([-1, 1, 2, 3], dtype="<i8").tobytes()
        expected = [-1, 1]
    else:
        desc["#strings"] = 2
        members[f"{path}/values.bin"] = u64([0, 1, 1, 0])
        members[f"{path}/strings.bin"] = b"ab"
        members[f"{path}/string-mapping.bin"] = u64([0, 1, 2])
        expected = [0, 1]
    model = tinyumbi.read(archive(idx, members))
    dist = model["annotations"]["custom"]["a"]["states"]
    assert list(dist["offsets"]) == [0, 2, 2, 4]
    assert list(dist["values"][:2]) == expected
    assert list(dist["probabilities"][:2]) == [1, 1]


def test_dirac_and_interval_defaults(archive):
    idx = index(
        **{
            "time": "stochastic",
            "exit-rate-type": {"type": "rational-interval", "size": 512},
            "branch-probability-type": {"type": "double-interval"},
        }
    )
    idx["annotations"] = {
        "custom": {
            "a": {
                "type": {"type": "uint-interval"},
                "applies-to": ["states"],
                "probability-type": {"type": "rational-interval"},
                "#probabilities": 3,
            }
        }
    }
    model = tinyumbi.read(archive(idx))
    dist = model["annotations"]["custom"]["a"]["states"]
    assert isinstance(dist["offsets"], range)
    assert plain(dist["values"]) == [(0, 0)] * 3
    assert plain(dist["probabilities"]) == [((1, 1), (1, 1))] * 3
    assert plain(model["state_to_exit_rate"]) == [((0, 1), (0, 1))] * 3
    assert plain(model["branch_to_probability"]) == [(0, 0)] * 3


def test_native_bits(archive):
    idx = index(n=65, **{"#initial-states": 3})
    model = tinyumbi.read(archive(idx, {"state-is-initial.bin": b"\x81" + b"\0" * 7 + b"\x01" + b"\0" * 7}))
    bits = model["state_is_initial"]
    assert type(bits) is bitarray
    assert bits.count() == 3
    assert bits.count(False, 0, 65) == 62
    assert list(bits.search(bitarray("1"))) == [0, 7, 64]
    assert bits.buffer_info().nbytes == 16


def test_raw_extensions_and_pax_long_names(archive):
    name = "extensions/" + "x" * 150 + ".bin"
    path = archive(index(), {name: b"hello"})
    raw = tinyumbi.read_archive(path)
    assert type(raw) is dict
    assert raw[name].tobytes() == b"hello"
    assert raw[name].readonly


def test_deterministic_branch_observations_and_metadata(archive):
    idx = index(**{"#observations": 2, "observations-apply-to": "branches", "#choice-actions": 1})
    idx["model-data"] = {"name": "example", "authors": ["a"], "description": "☃"}
    idx["file-data"] = {"tool": "test", "creation-date": 2**64 - 1, "parameters": [1, "x"]}
    idx["extension"] = {"custom": True}
    model = tinyumbi.read(archive(idx, {"observations/branches/values.bin": u64([1, 0, 1])}))
    assert model["index"] == idx
    assert list(model["observations"]["values"]) == [1, 0, 1]
    assert "offsets" not in model["observations"]
    assert model["actions"]["choices"]["strings"] is None


def test_empty_entity_arrays(archive):
    idx = index(n=1, **{"#choices": 0, "#branches": 0})
    idx["annotations"] = {
        "custom": {"a": {"type": {"type": "rational-interval"}, "applies-to": ["branches"]}}
    }
    model = tinyumbi.read(
        archive(
            idx,
            {
                "state-to-choices.bin": u64([0, 0]),
                "choice-to-branches.bin": u64([0]),
                "branch-to-target.bin": b"",
                "annotations/custom/a/branches/values.bin": b"",
            },
        )
    )
    assert model["branch_to_target"].shape == (0,)
    assert model["annotations"]["custom"]["a"]["branches"]["values"].shape == (0, 2)
