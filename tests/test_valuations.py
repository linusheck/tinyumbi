import struct
from fractions import Fraction

import numpy as np
import pytest
from conftest import index, u32, u64

import tinyumbi


def field(name, kind, bits, **extra):
    return {"name": name, "type": {"type": kind, "size": bits}, **extra}


def test_heterogeneous_optional_strings_offsets(archive):
    idx = index()
    idx["valuations"] = {
        "states": {
            "unique": True,
            "#strings": 3,
            "classes": [
                {
                    "variables": [
                        field("a", "bool", 2),
                        field("b", "uint", 6, **{"is-optional": True}),
                        {"padding": 5},
                        field("s", "string", 64),
                        field("c", "bool", 3),
                        {"padding": 7},
                    ]
                },
                {"variables": [field("u", "uint", 8, offset=-128), field("i", "int", 16)]},
            ],
        }
    }
    # Pack Boolean, optional uint, string ID, Boolean; then the spec's offset example.
    first = bytes([0b11011101, 0b01000000, 0, 0, 0, 0, 0, 0, 0, 0b01000000, 0])
    second = bytes([0b10111000, 0b11111111, 0b10111101])
    absent = bytes([first[0] & ~(1 << 2)]) + first[1:]
    model = tinyumbi.read(
        archive(
            idx,
            {
                "valuations/states/valuations.bin": first + second + absent,
                "valuations/states/valuation-to-class.bin": u32([0, 1, 0]),
                "valuations/states/string-mapping.bin": u64([0, 3, 5, 6]),
                "valuations/states/strings.bin": b"xyzabc",
            },
        )
    )
    table = model["valuations"]["states"]
    assert list(table["offsets"]) == [0, 11, 14, 25]
    assert tinyumbi.valuation_at(table, 0) == {"a": True, "b": 27, "s": "ab", "c": True}
    assert tinyumbi.valuation_at(table, 1) == {"u": 56, "i": -16897}
    assert tinyumbi.valuation_at(table, 2) == {"a": True, "b": None, "s": "ab", "c": True}
    assert list(tinyumbi.column(table, "b")) == [27, None, None]


@pytest.mark.parametrize("entity", ["states", "choices", "branches", "observations", "players"])
def test_aligned_zero_copy_columns(archive, entity):
    idx = index(**{"#observations": 3, "observations-apply-to": "states"})
    idx["valuations"] = {
        entity: {
            "unique": False,
            "classes": [
                {"variables": [field("a", "uint", 8), field("b", "int", 16), field("d", "double", 64)]}
            ],
        }
    }
    n = idx["transition-system"][f"#{entity}"]
    data = b"".join(struct.pack("<Bhd", i, -i, i + 0.25) for i in range(n))
    table = tinyumbi.read(archive(idx, {f"valuations/{entity}/valuations.bin": data}))["valuations"][entity]
    assert isinstance(table["offsets"], range)
    for name, expected in (
        ("a", list(range(n))),
        ("b", [-i for i in range(n)]),
        ("d", [i + 0.25 for i in range(n)]),
    ):
        column = tinyumbi.column(table, name)
        assert isinstance(column, np.ndarray)
        assert not column.flags.writeable
        assert not column.flags.owndata
        assert column.tolist() == expected
        assert np.shares_memory(column, np.frombuffer(table["data"], dtype="u1"))


@pytest.mark.parametrize("bits", [1, 3, 7, 9, 24, 65, 127, 256])
@pytest.mark.parametrize("kind", ["int", "uint"])
def test_arbitrary_bit_widths_and_offsets(archive, bits, kind):
    values = [
        0,
        1 if kind == "uint" or bits > 1 else -1,
        (2**bits - 1) if kind == "uint" else -(2 ** (bits - 1)),
    ]
    width = (bits + 7) // 8
    variables = [field("x", kind, bits, offset=-42)]
    if bits % 8:
        variables.append({"padding": 8 - bits % 8})
    idx = index()
    idx["valuations"] = {"states": {"unique": False, "classes": [{"variables": variables}]}}
    data = b"".join((v % 2**bits).to_bytes(width, "little") for v in values)
    table = tinyumbi.read(archive(idx, {"valuations/states/valuations.bin": data}))["valuations"]["states"]
    assert list(tinyumbi.column(table, "x")) == [v - 42 for v in values]


@pytest.mark.parametrize("bits", [4, 14, 128, 256])
def test_bit_packed_rationals_match_reference(archive, bits):
    import umbi.binary
    from umbi.binary import SizedType, StructAttribute, StructPadding, StructType
    from umbi.datatypes import NumericPrimitiveType

    fields = [StructAttribute("x", SizedType(NumericPrimitiveType.RATIONAL, bits))]
    variables = [field("x", "rational", bits)]
    if bits % 8:
        padding = 8 - bits % 8
        fields.append(StructPadding(padding))
        variables.append({"padding": padding})
    values = [Fraction(-1), Fraction(1), Fraction(0)]
    data = umbi.binary.vector_to_bytes([{"x": v} for v in values], StructType(fields))
    idx = index()
    idx["valuations"] = {"states": {"unique": True, "classes": [{"variables": variables}]}}
    table = tinyumbi.read(archive(idx, {"valuations/states/valuations.bin": data}))["valuations"]["states"]
    assert list(tinyumbi.column(table, "x")) == [(-1, 1), (1, 1), (0, 1)]


def test_empty_valuation_classes(archive):
    idx = index()
    idx["valuations"] = {"states": {"unique": False, "classes": [{"variables": []}]}}
    table = tinyumbi.read(archive(idx, {"valuations/states/valuations.bin": b""}))["valuations"]["states"]
    assert tinyumbi.valuation_at(table, 2) == {}


def test_unaligned_double_and_optional_absence(archive):
    idx = index()
    idx["valuations"] = {
        "states": {
            "unique": False,
            "classes": [
                {
                    "variables": [
                        {"padding": 3},
                        field("d", "double", 64, **{"is-optional": True}),
                        {"padding": 4},
                    ]
                }
            ],
        }
    }
    data = b""
    for present, value in [(1, -1.25), (0, 99), (1, 2.5)]:
        raw = int.from_bytes(struct.pack("<d", value), "little")
        data += ((raw << 4) | (present << 3)).to_bytes(9, "little")
    table = tinyumbi.read(archive(idx, {"valuations/states/valuations.bin": data}))["valuations"]["states"]
    assert list(tinyumbi.column(table, "d")) == [-1.25, None, 2.5]
