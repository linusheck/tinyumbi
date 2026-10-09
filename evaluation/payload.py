"""Touch native numeric arrays and decode strings and valuation rows."""

import numpy as np
from bitarray import bitarray

import tinyumbi


def _touch(value):
    if isinstance(value, dict):
        if set(value) == {"offsets", "data"}:
            for i in range(len(value["offsets"]) - 1):
                tinyumbi.string_at(value, i)
        else:
            for item in value.values():
                _touch(item)
    elif isinstance(value, np.ndarray):
        if value.dtype.fields:
            for name in value.dtype.names:
                _touch(value[name])
        elif value.dtype.kind == "V":
            # Wide integers retain their exact bytes; this view also supports
            # strided fields, interval endpoints, and broadcast defaults.
            value.view((np.uint8, value.dtype.itemsize)).sum()
        else:
            value.sum()
    elif isinstance(value, (memoryview, range)):
        sum(value)
    elif isinstance(value, bitarray):
        value.count()


def touch_model(model):
    for key, value in model.items():
        if key not in {"index", "valuations"}:
            _touch(value)
    for table in model["valuations"].values():
        for i in range(len(table["class_ids"])):
            tinyumbi.valuation_at(table, i)
