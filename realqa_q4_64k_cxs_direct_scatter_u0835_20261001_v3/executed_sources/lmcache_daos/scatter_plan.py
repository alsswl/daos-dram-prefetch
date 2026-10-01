"""CPU-only address planner. No tensor allocations or payload copies.

Persistent order: [K/V, layer, token, hidden]. Physical GPU order is
per-layer [block, K/V, token, head, dim], or [K/V, block, token, head, dim].
Partial prefix reads select array extents instead of writing cached tokens.
"""
from dataclasses import dataclass

MAX_IOVS = 65536


@dataclass(frozen=True)
class Segment:
    pointer: int
    length: int
    offset: int


def plan_segments(base_ptrs, num_blocks, block_size, hidden, itemsize,
                  slots, *, skip=0, layout="NB_TWO", max_iovs=MAX_IOVS):
    bases, slots = tuple(base_ptrs), tuple(slots)
    if (not bases or any(type(p) is not int or p <= 0 for p in bases)
            or any(type(v) is not int or v <= 0
                   for v in (num_blocks, block_size, hidden, itemsize))):
        raise ValueError("Invalid KV geometry or base pointers")
    if layout not in {"NB_TWO", "TWO_NB"}:
        raise ValueError("Only contiguous NHD KV layouts are supported")
    if type(skip) is not int or not 0 <= skip <= len(slots):
        raise ValueError("Invalid prefix skip")
    active = slots[skip:]
    if any(type(s) is not int or not 0 <= s < num_blocks * block_size for s in active):
        raise ValueError("Invalid destination slot")
    if len(set(active)) != len(active):
        raise ValueError("Aliased destination slots")
    # A destination must never alias any known cached prefix slot.
    if set(active).intersection(s for s in slots[:skip] if s >= 0):
        raise ValueError("Destination aliases cached prefix")
    row = hidden * itemsize
    layers, tokens = len(bases), len(slots)
    result = []
    for kv in range(2):
        for layer, base in enumerate(bases):
            t = skip
            while t < tokens:
                block, within = divmod(slots[t], block_size)
                count = 1
                while (t + count < tokens and within + count < block_size
                       and slots[t + count] == slots[t] + count):
                    count += 1
                if layout == "NB_TWO":
                    dest_row = (block * 2 + kv) * block_size + within
                else:
                    dest_row = (kv * num_blocks + block) * block_size + within
                seg = Segment(base + dest_row * row, count * row,
                              ((kv * layers + layer) * tokens + t) * row)
                if seg.pointer + seg.length > 2**64 - 1:
                    raise ValueError("Pointer overflow")
                result.append(seg)
                if len(result) > max_iovs:
                    raise ValueError("SGL too large; reduce chunk size")
                t += count
    return result
