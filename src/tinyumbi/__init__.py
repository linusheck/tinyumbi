"""UMB 1.0 reader."""

import bz2
import gzip
import lzma
import mmap
import shutil
import struct
import tarfile
import tempfile
from contextlib import ExitStack

import numpy as np
import orjson
from bitarray import bitarray

__all__ = ["column", "rational_at", "read", "read_archive", "string_at", "valuation_at"]
__version__ = "0.1.0"


# Type descriptors.
BOOL = {"type": "bool"}
DEFAULT_BITS = {"bool": 1, "int": 64, "uint": 64, "double": 64, "rational": 128, "string": 64}
U32 = {"type": "uint", "size": 32}
U64 = {"type": "uint"}


# Type and buffer helpers.


def type_info(desc):
    name = desc["type"]
    base = name.removesuffix("-interval")
    parts = 2 if name.endswith("-interval") else 1
    return base, desc.get("size", DEFAULT_BITS[base] * parts), parts


def dtype_for(desc):
    base, bits, parts = type_info(desc)
    width = bits // parts
    if base == "rational":
        half = width // 16
        numerator = f"<i{half}" if half in {1, 2, 4, 8} else f"V{half}"
        denominator = f"<u{half}" if half in {1, 2, 4, 8} else f"V{half}"
        return np.dtype([("numerator", numerator), ("denominator", denominator)])
    kind = "f" if base == "double" else "i" if base == "int" else "u"
    return np.dtype(f"<{kind}{width // 8}") if width in {8, 16, 32, 64} else np.dtype(f"V{width // 8}")


def scalar(raw, desc):
    """Decode a bit-packed valuation scalar, already shifted and masked."""
    base, bits, _ = type_info(desc)
    if base == "bool":
        return bool(raw)
    if base == "int":
        return raw - (1 << bits) if raw >> (bits - 1) else raw
    if base == "double":
        return struct.unpack("<d", raw.to_bytes(8, "little"))[0]
    if base == "rational":
        half = bits // 2
        num = raw >> half
        if num >> (half - 1):
            num -= 1 << half
        return num, raw & ((1 << half) - 1)
    return raw


def array(data, desc, n, default=0):
    base, bits, parts = type_info(desc)
    if base == "bool" and bits == 1:
        return bitarray(buffer=data, endian="little") if data is not None else np.broadcast_to(False, (n,))
    dtype = dtype_for(desc)
    shape = (n, 2) if parts == 2 else (n,)
    if data is None:
        scalar = np.zeros((), dtype=dtype)
        if base == "rational":
            for field, value in (("numerator", default), ("denominator", 1)):
                if dtype[field].kind == "V":
                    scalar[field] = np.void(
                        value.to_bytes(dtype[field].itemsize, "little", signed=field == "numerator")
                    )
                else:
                    scalar[field] = value
        else:
            scalar[...] = default
        return np.broadcast_to(scalar, shape)
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def offsets(data, n):
    return range(n + 1) if data is None else np.frombuffer(data, dtype="<u8")


# Table construction and decoding helpers.


def _strings(files, path):
    mapping = files.get(f"{path}/string-mapping.bin")
    return (
        None
        if mapping is None
        else {"offsets": np.frombuffer(mapping, dtype="<u8"), "data": files[f"{path}/strings.bin"]}
    )


def _annotation(files, path, desc, n):
    result = {"type": desc["type"]}
    ptype = desc.get("probability-type")
    if ptype is not None:
        result["offsets"] = offsets(files.get(f"{path}/distribution-mapping.bin"), n)
        n = int(result["offsets"][-1])
        result["probabilities"] = array(files.get(f"{path}/probabilities.bin"), ptype, n, 1)
    result["values"] = array(files.get(f"{path}/values.bin"), desc["type"], n)
    if desc["type"]["type"] == "string":
        result["strings"] = _strings(files, path)
    return result


def _valuations(files, path, desc, n):
    layouts, sizes = [], []
    for cls in desc["classes"]:
        layout, bit = {}, 0
        for field in cls["variables"]:
            if "padding" in field:
                bit += field["padding"]
            else:
                layout[field["name"]] = {**field, "bit": bit}
                bit += type_info(field["type"])[1] + field.get("is-optional", False)
        layouts.append(layout)
        sizes.append(bit // 8)
    ids = files.get(f"{path}/valuation-to-class.bin")
    if ids is None:
        ids = np.broadcast_to(np.uint32(0), (n,))
        positions = range(0, (n + 1) * sizes[0], sizes[0]) if sizes[0] else np.broadcast_to(0, (n + 1,))
    else:
        ids = np.frombuffer(ids, dtype="<u4")
        positions = np.empty(n + 1, dtype="u8")
        positions[0] = 0
        np.cumsum(np.asarray(sizes, dtype="u8")[ids], out=positions[1:])
        positions.flags.writeable = False
    return {
        "data": files[f"{path}/valuations.bin"],
        "classes": layouts,
        "sizes": sizes,
        "class_ids": ids,
        "offsets": positions,
        "strings": _strings(files, path),
    }


def _value(data, field, strings):
    bit = field["bit"]
    _, bits, _ = type_info(field["type"])
    optional = field.get("is-optional", False)
    width = (bit % 8 + bits + optional + 7) // 8
    raw = int.from_bytes(data[bit // 8 : bit // 8 + width], "little") >> (bit % 8)
    if optional:
        if not raw & 1:
            return None
        raw >>= 1
    value = scalar(raw & ((1 << bits) - 1), field["type"])
    if field["type"]["type"] == "string":
        return string_at(strings, value)
    return value + field["offset"] if field.get("offset") else value


def _column_values(table, name):
    for i, cls in enumerate(table["class_ids"]):
        cls = int(cls)
        field = table["classes"][cls].get(name)
        if field is None:
            yield None
        else:
            start = int(table["offsets"][i])
            yield _value(table["data"][start : start + table["sizes"][cls]], field, table["strings"])


# Public API.


def read_archive(path):
    """Return {name: memoryview}. Compressed archives use an anonymous disk spool."""
    with ExitStack() as stack:
        source = stack.enter_context(open(path, "rb"))
        magic = source.read(6)
        source.seek(0)
        for signature, codec in (
            (b"\x1f\x8b", gzip.open),
            (b"\xfd7zXZ\0", lzma.LZMAFile),
            (b"BZh", bz2.BZ2File),
        ):
            if magic.startswith(signature):
                spool = stack.enter_context(tempfile.TemporaryFile())
                with codec(source, mode="rb") as decoded:
                    shutil.copyfileobj(decoded, spool, length=1024 * 1024)
                spool.seek(0)
                source = spool
                break
        mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        members = {}
        with tarfile.open(fileobj=source, mode="r:") as tar:
            for member in tar:
                if member.isfile():
                    name = member.name
                    while name.startswith("./"):
                        name = name[2:]
                    members[name.removeprefix("/")] = memoryview(mapping)[
                        member.offset_data : member.offset_data + member.size
                    ]
        return members


def read(path):
    """Return native arrays and plain dictionaries. Assumes a valid UMB 1.0 file."""
    files = read_archive(path)
    idx = orjson.loads(files["index.json"])
    ts = idx["transition-system"]
    counts = {e: ts[f"#{e}"] for e in ("states", "choices", "branches", "observations", "players")}
    ns, nc, nb = counts["states"], counts["choices"], counts["branches"]
    result = {
        "index": idx,
        "state_to_choices": offsets(files.get("state-to-choices.bin"), ns),
        "choice_to_branches": offsets(files.get("choice-to-branches.bin"), nc),
        "branch_to_target": array(files.get("branch-to-target.bin"), U64, nb),
        "state_is_initial": array(files.get("state-is-initial.bin"), BOOL, ns),
        "state_to_player": array(files.get("state-to-player.bin"), U32, ns) if counts["players"] else None,
        "state_is_markovian": array(files.get("state-is-markovian.bin"), BOOL, ns)
        if ts["time"] == "urgent-stochastic"
        else None,
    }
    for key, type_key, n in (
        ("branch_to_probability", "branch-probability-type", nb),
        ("state_to_exit_rate", "exit-rate-type", ns),
    ):
        result[key] = (
            array(files.get(f"{key.replace('_', '-')}.bin"), ts[type_key], n) if type_key in ts else None
        )
    result["actions"] = {}
    for entity, key in (("choices", "#choice-actions"), ("branches", "#branch-actions")):
        if ts[key]:
            path = f"actions/{entity}"
            result["actions"][entity] = {
                "values": array(files.get(f"{path}/values.bin"), U32, counts[entity]),
                "strings": _strings(files, path),
            }
    result["annotations"] = {
        group: {
            name: {
                entity: _annotation(files, f"annotations/{group}/{name}/{entity}", desc, counts[entity])
                for entity in desc["applies-to"]
            }
            for name, desc in items.items()
        }
        for group, items in idx.get("annotations", {}).items()
    }
    result["observations"] = None
    if counts["observations"]:
        entity = ts["observations-apply-to"]
        desc = {"type": U64}
        if "observation-probability-type" in ts:
            desc["probability-type"] = ts["observation-probability-type"]
        result["observations"] = _annotation(files, f"observations/{entity}", desc, counts[entity])
    result["valuations"] = {
        entity: _valuations(files, f"valuations/{entity}", desc, counts[entity])
        for entity, desc in idx.get("valuations", {}).items()
    }
    return result


def string_at(table, i):
    """Decode one UTF-8 string from {'offsets', 'data'}."""
    start, end = int(table["offsets"][i]), int(table["offsets"][i + 1])
    return bytes(table["data"][start:end]).decode("utf-8")


def rational_at(values, i):
    """Return (numerator, denominator), or a pair of those tuples for an interval."""
    value = values[i]
    if isinstance(value, np.ndarray):
        return tuple(rational_at(value, j) for j in range(len(value)))
    num, den = value["numerator"], value["denominator"]
    if isinstance(num, np.void):
        num = int.from_bytes(num.tobytes(), "little", signed=True)
        den = int.from_bytes(den.tobytes(), "little")
    return int(num), int(den)


def valuation_at(table, i):
    """Decode one entity's valuation to a plain dict."""
    cls = int(table["class_ids"][i])
    start = int(table["offsets"][i])
    data = table["data"][start : start + table["sizes"][cls]]
    return {name: _value(data, field, table["strings"]) for name, field in table["classes"][cls].items()}


def column(table, name):
    """Zero-copy aligned numeric column; other layouts return an iterator."""
    ids = table["class_ids"]
    if ids.strides == (0,) and len(ids):
        cls = int(ids[0])
        field = table["classes"][cls][name]
        base, bits, _ = type_info(field["type"])
        if (
            not field.get("is-optional")
            and not field.get("offset")
            and field["bit"] % 8 == 0
            and base in {"int", "uint", "double"}
            and bits in {8, 16, 32, 64}
        ):
            return np.ndarray(
                (len(ids),),
                dtype=dtype_for(field["type"]),
                buffer=table["data"],
                offset=field["bit"] // 8,
                strides=(table["sizes"][cls],),
            )
    return _column_values(table, name)
