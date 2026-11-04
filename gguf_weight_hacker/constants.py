"""GGUF / GGML constants: value types, tensor types and quant block geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian
GGUF_DEFAULT_ALIGNMENT = 32
FP16_MAX = 65504.0


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# struct format char + byte size for scalar KV value types
SCALAR_FMT = {
    GGUFValueType.UINT8: ("B", 1),
    GGUFValueType.INT8: ("b", 1),
    GGUFValueType.UINT16: ("H", 2),
    GGUFValueType.INT16: ("h", 2),
    GGUFValueType.UINT32: ("I", 4),
    GGUFValueType.INT32: ("i", 4),
    GGUFValueType.FLOAT32: ("f", 4),
    GGUFValueType.BOOL: ("?", 1),
    GGUFValueType.UINT64: ("Q", 8),
    GGUFValueType.INT64: ("q", 8),
    GGUFValueType.FLOAT64: ("d", 8),
}


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39


@dataclass(frozen=True)
class Field:
    """A named field inside one quant block (byte offset, length, numpy dtype, count)."""
    name: str
    offset: int
    dtype: str          # numpy dtype string, e.g. "<f2", "u1", "i1"
    count: int = 1
    role: str = "payload"   # "scale" | "min" | "subscale" | "payload" | "hbits"

    @property
    def nbytes(self) -> int:
        import numpy as np
        return np.dtype(self.dtype).itemsize * self.count


@dataclass(frozen=True)
class BlockLayout:
    type: GGMLType
    block_elems: int
    block_bytes: int
    fields: tuple = field(default_factory=tuple)

    @property
    def is_float(self) -> bool:
        return self.type in (GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.F64)

    @property
    def scale_fields(self):
        """FP16 fields that linearly scale the whole block ("d", "dmin", "m")."""
        return tuple(f for f in self.fields if f.role in ("scale", "min") and f.dtype == "<f2")

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.type.name} has no field {name!r}; fields: {[f.name for f in self.fields]}")


def _L(t, ne, nb, *fields):
    return BlockLayout(t, ne, nb, tuple(fields))


F = Field
BLOCK_LAYOUTS: dict[GGMLType, BlockLayout] = {
    GGMLType.F32: _L(GGMLType.F32, 1, 4, F("value", 0, "<f4")),
    GGMLType.F16: _L(GGMLType.F16, 1, 2, F("value", 0, "<f2")),
    GGMLType.BF16: _L(GGMLType.BF16, 1, 2, F("value", 0, "<u2")),
    GGMLType.F64: _L(GGMLType.F64, 1, 8, F("value", 0, "<f8")),
    GGMLType.I8: _L(GGMLType.I8, 1, 1, F("value", 0, "i1")),
    GGMLType.I16: _L(GGMLType.I16, 1, 2, F("value", 0, "<i2")),
    GGMLType.I32: _L(GGMLType.I32, 1, 4, F("value", 0, "<i4")),
    GGMLType.I64: _L(GGMLType.I64, 1, 8, F("value", 0, "<i8")),
    GGMLType.Q4_0: _L(GGMLType.Q4_0, 32, 18, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "u1", 16)),
    GGMLType.Q4_1: _L(GGMLType.Q4_1, 32, 20, F("d", 0, "<f2", 1, "scale"), F("m", 2, "<f2", 1, "min"),
                      F("qs", 4, "u1", 16)),
    GGMLType.Q5_0: _L(GGMLType.Q5_0, 32, 22, F("d", 0, "<f2", 1, "scale"), F("qh", 2, "u1", 4, "hbits"),
                      F("qs", 6, "u1", 16)),
    GGMLType.Q5_1: _L(GGMLType.Q5_1, 32, 24, F("d", 0, "<f2", 1, "scale"), F("m", 2, "<f2", 1, "min"),
                      F("qh", 4, "u1", 4, "hbits"), F("qs", 8, "u1", 16)),
    GGMLType.Q8_0: _L(GGMLType.Q8_0, 32, 34, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "i1", 32)),
    GGMLType.Q8_1: _L(GGMLType.Q8_1, 32, 40, F("d", 0, "<f4", 1, "scale"), F("s", 4, "<f4", 1, "min"),
                      F("qs", 8, "i1", 32)),
    GGMLType.Q2_K: _L(GGMLType.Q2_K, 256, 84, F("scales", 0, "u1", 16, "subscale"), F("qs", 16, "u1", 64),
                      F("d", 80, "<f2", 1, "scale"), F("dmin", 82, "<f2", 1, "min")),
    GGMLType.Q3_K: _L(GGMLType.Q3_K, 256, 110, F("hmask", 0, "u1", 32, "hbits"), F("qs", 32, "u1", 64),
                      F("scales", 96, "u1", 12, "subscale"), F("d", 108, "<f2", 1, "scale")),
    GGMLType.Q4_K: _L(GGMLType.Q4_K, 256, 144, F("d", 0, "<f2", 1, "scale"), F("dmin", 2, "<f2", 1, "min"),
                      F("scales", 4, "u1", 12, "subscale"), F("qs", 16, "u1", 128)),
    GGMLType.Q5_K: _L(GGMLType.Q5_K, 256, 176, F("d", 0, "<f2", 1, "scale"), F("dmin", 2, "<f2", 1, "min"),
                      F("scales", 4, "u1", 12, "subscale"), F("qh", 16, "u1", 32, "hbits"),
                      F("qs", 48, "u1", 128)),
    GGMLType.Q6_K: _L(GGMLType.Q6_K, 256, 210, F("ql", 0, "u1", 128), F("qh", 128, "u1", 64, "hbits"),
                      F("scales", 192, "i1", 16, "subscale"), F("d", 208, "<f2", 1, "scale")),
    GGMLType.Q8_K: _L(GGMLType.Q8_K, 256, 292, F("d", 0, "<f4", 1, "scale"), F("qs", 4, "i1", 256),
                      F("bsums", 260, "<i2", 16)),
    GGMLType.IQ4_NL: _L(GGMLType.IQ4_NL, 32, 18, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "u1", 16)),
    GGMLType.IQ4_XS: _L(GGMLType.IQ4_XS, 256, 136, F("d", 0, "<f2", 1, "scale"),
                        F("scales_h", 2, "<u2", 1, "subscale"), F("scales_l", 4, "u1", 4, "subscale"),
                        F("qs", 8, "u1", 128)),
    GGMLType.IQ2_XXS: _L(GGMLType.IQ2_XXS, 256, 66, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "<u2", 32)),
    GGMLType.IQ2_XS: _L(GGMLType.IQ2_XS, 256, 74, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "<u2", 32),
                        F("scales", 66, "u1", 8, "subscale")),
    GGMLType.IQ2_S: _L(GGMLType.IQ2_S, 256, 82, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "u1", 64),
                       F("qh", 66, "u1", 8, "hbits"), F("scales", 74, "u1", 8, "subscale")),
    GGMLType.IQ3_XXS: _L(GGMLType.IQ3_XXS, 256, 98, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "u1", 96)),
    GGMLType.IQ3_S: _L(GGMLType.IQ3_S, 256, 110, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "u1", 64),
                       F("qh", 66, "u1", 8, "hbits"), F("signs", 74, "u1", 32),
                       F("scales", 106, "u1", 4, "subscale")),
    GGMLType.IQ1_S: _L(GGMLType.IQ1_S, 256, 50, F("d", 0, "<f2", 1, "scale"), F("qs", 2, "u1", 32),
                       F("qh", 34, "<u2", 8, "hbits")),
    GGMLType.IQ1_M: _L(GGMLType.IQ1_M, 256, 56, F("qs", 0, "u1", 32), F("qh", 32, "u1", 16, "hbits"),
                       F("scales", 48, "u1", 8, "subscale")),
    GGMLType.TQ1_0: _L(GGMLType.TQ1_0, 256, 54, F("qs", 0, "u1", 48), F("qh", 48, "u1", 4, "hbits"),
                       F("d", 52, "<f2", 1, "scale")),
    GGMLType.TQ2_0: _L(GGMLType.TQ2_0, 256, 66, F("qs", 0, "u1", 64), F("d", 64, "<f2", 1, "scale")),
    GGMLType.MXFP4: _L(GGMLType.MXFP4, 32, 17, F("e", 0, "u1", 1, "subscale"), F("qs", 1, "u1", 16)),
}

# llama.cpp "file types" (general.file_type) -> human name. Note Q4_K_M etc. are *mixes*
# of tensor types, not tensor types themselves.
LLAMA_FTYPE_NAMES = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S",
    25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
    36: "MOSTLY_TQ1_0", 37: "MOSTLY_TQ2_0", 38: "MOSTLY_MXFP4_MOE",
}


def type_from_any(t) -> GGMLType:
    if isinstance(t, GGMLType):
        return t
    if isinstance(t, int):
        return GGMLType(t)
    s = str(t).upper().removeprefix("GGML_TYPE_")
    return GGMLType[s]


def layout_for(t) -> BlockLayout:
    t = type_from_any(t)
    try:
        return BLOCK_LAYOUTS[t]
    except KeyError:
        raise NotImplementedError(f"No block layout for {t.name}") from None
