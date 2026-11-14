# SPDX-License-Identifier: Apache-2.0
"""Stream raw GLM ModelOpt expert tensors directly into stacked MLX banks."""

import fcntl
import json
import re
import struct
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.generate import wired_limit

from ..deepseek_v41.storage import (
    RESIDENT_READ_BYTES,
    SAFETENSORS_NUMPY_DTYPES,
    metal_backed_array,
)

_EXPERT = re.compile(
    r"(model\.language_model\.layers\.\d+\.mlp)\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)"
)


def _read_into(reader, origin, entry, value):
    target = memoryview(value).cast("B")
    start, end = entry["data_offsets"]
    if end - start != len(target):
        raise ValueError("GLM ModelOpt tensor byte count does not match its shape")
    reader.seek(origin + start)
    offset = 0
    while offset < len(target):
        count = reader.readinto(target[offset : offset + RESIDENT_READ_BYTES])
        if not count:
            raise ValueError("Truncated GLM ModelOpt safetensors data")
        offset += count


def _read_weights(model_path: Path, index: dict, num_experts: int) -> dict:
    with ExitStack() as stack:
        sources = {}
        for shard in sorted(set(index.values())):
            reader = stack.enter_context((model_path / shard).open("rb"))
            (length,) = struct.unpack("<Q", reader.read(8))
            header = json.loads(reader.read(length))
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(reader.fileno(), fcntl.F_NOCACHE, 1)
            sources[shard] = reader, 8 + length, header

        def entry_for(key):
            reader, origin, header = sources[index[key]]
            return reader, origin, header[key]

        groups = defaultdict(dict)
        weights = {}
        for key in index:
            match = _EXPERT.fullmatch(key)
            if match:
                prefix, expert, projection, field = match.groups()
                groups[f"{prefix}.switch_mlp.{projection}.{field}"][int(expert)] = key
                continue
            reader, origin, entry = entry_for(key)
            value = np.empty(
                entry["shape"], dtype=SAFETENSORS_NUMPY_DTYPES[entry["dtype"]]
            )
            _read_into(reader, origin, entry, value)
            array = mx.array(value)
            weights[key] = (
                array.view(mx.bfloat16) if entry["dtype"] == "BF16" else array
            )

        for dest, experts in groups.items():
            if experts.keys() != set(range(num_experts)):
                raise ValueError(f"Incomplete GLM ModelOpt expert bank: {dest}")
            _, _, first = entry_for(experts[0])
            shape, dtype = first["shape"], first["dtype"]
            # The final bank is writable shared Metal memory. Fill it in place
            # to avoid holding the per-expert buffers and a second bank.
            array = metal_backed_array(
                (num_experts, *shape), np.dtype(SAFETENSORS_NUMPY_DTYPES[dtype])
            )
            values = np.asarray(array)
            for expert, key in experts.items():
                reader, origin, entry = entry_for(key)
                if entry["shape"] != shape or entry["dtype"] != dtype:
                    raise ValueError(f"Inconsistent GLM ModelOpt expert tensor: {key}")
                _read_into(reader, origin, entry, values[expert : expert + 1])
            if dtype == "BF16":
                array = array.view(mx.bfloat16)
            weights[dest] = array
        return weights


@contextmanager
def stream_glm5_modelopt_weights(model_path: Path):
    """Feed pre-stacked raw tensors into the normal mlx-vlm strict loader."""
    config = json.loads((model_path / "config.json").read_text())
    text_config = config.get("text_config") or {}
    quant = config.get("quantization_config") or text_config.get("quantization_config")
    manifest = model_path / "model.safetensors.index.json"
    if (
        not isinstance(quant, dict)
        or quant.get("quant_method") != "modelopt"
        or quant.get("quant_algo") != "NVFP4"
        or not manifest.is_file()
    ):
        yield False
        return

    index = json.loads(manifest.read_text())["weight_map"]
    if not any(
        key.endswith(".weight_scale_2") and _EXPERT.fullmatch(key) for key in index
    ):
        yield False
        return

    from mlx_vlm import utils as vlm_utils

    shards = set(index.values())
    first = min(shards)
    target = model_path.resolve()
    original = vlm_utils._load_safetensors

    def load_shard(path):
        path = Path(path)
        shard = path.name
        if path.parent.resolve() != target or shard not in shards:
            return original(str(path))
        if shard != first:
            return {}
        # Match mlx-vlm's sorted manifest traversal. One read can fill a
        # complete bank even when it spans several shard files.
        with wired_limit(nn.Module()):
            return _read_weights(model_path, index, text_config["n_routed_experts"])

    vlm_utils._load_safetensors = load_shard
    try:
        yield True
    finally:
        vlm_utils._load_safetensors = original
