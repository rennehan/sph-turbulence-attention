"""Loader for .sph snapshot bundles produced by the dashboard / headless runner.

Container layout (little-endian):
    b"SPHD" | u32 version | u32 manifestLen | manifest(JSON utf8) | pad->4 | payload

The manifest holds run metadata and, per snapshot, each array's
{dtype, offset, length}; offsets are relative to the (4-byte-aligned)
payload start. f4 = float32, i4 = int32.

Nothing here is derived/filtered: snapshots carry raw primitives
(x, y, vx, vy, rho, P) plus the neighbour list in CSR form
(nbr_offsets length N+1, nbr_indices flattened, self excluded). The SGS
filter, target, and invariant features are built on top of this.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPE = {"f4": "<f4", "i4": "<i4"}


@dataclass
class Snapshot:
    seed: int
    index: int
    t: float
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    rho: np.ndarray
    P: np.ndarray
    nbr_offsets: np.ndarray  # int32, length N+1
    nbr_indices: np.ndarray  # int32, flattened neighbour ids (self excluded)

    def neighbors(self, i: int) -> np.ndarray:
        """Neighbour particle ids of particle i (within the dumped 4h radius)."""
        return self.nbr_indices[self.nbr_offsets[i]:self.nbr_offsets[i + 1]]


@dataclass
class Bundle:
    meta: dict
    snapshots: list[Snapshot]

    def by_seed(self, seed: int) -> list[Snapshot]:
        return [s for s in self.snapshots if s.seed == seed]


def load(path: str | Path) -> Bundle:
    raw = Path(path).read_bytes()
    if raw[:4] != b"SPHD":
        raise ValueError(f"bad magic {raw[:4]!r} (not an .sph bundle)")
    version, manifest_len = struct.unpack_from("<II", raw, 4)
    if version != 1:
        raise ValueError(f"unsupported version {version}")
    manifest = json.loads(raw[12:12 + manifest_len].decode("utf-8"))
    header_len = 12 + manifest_len
    payload_start = (header_len + 3) & ~3  # align to 4

    def read(spec: dict) -> np.ndarray:
        off = payload_start + spec["offset"]
        return np.frombuffer(raw, dtype=_DTYPE[spec["dtype"]],
                             count=spec["length"], offset=off)

    snaps = []
    for s in manifest["snapshots"]:
        a = s["arrays"]
        snaps.append(Snapshot(
            seed=s["seed"], index=s["index"], t=s["t"],
            x=read(a["x"]), y=read(a["y"]),
            vx=read(a["vx"]), vy=read(a["vy"]),
            rho=read(a["rho"]), P=read(a["P"]),
            nbr_offsets=read(a["nbr_offsets"]), nbr_indices=read(a["nbr_indices"]),
        ))
    return Bundle(meta=manifest["meta"], snapshots=snaps)


if __name__ == "__main__":
    import sys
    b = load(sys.argv[1])
    m = b.meta
    print(f"meta: N={m['N']} L={m['L']} h={m['h']:.5g} mode={m['mode']} "
          f"M={m['mach']:.3g} seeds={m['seeds']} radius={m['nbr_radius_h']:.2g}h")
    print(f"snapshots: {len(b.snapshots)} "
          f"({m['n_snapshots_per_seed']}/seed x {len(m['seeds'])} seeds)")
    s0 = b.snapshots[0]
    deg = np.diff(s0.nbr_offsets)
    print(f"snap0: seed={s0.seed} t={s0.t:.3g} N={s0.x.size} "
          f"neighbours/particle mean={deg.mean():.1f} min={deg.min()} max={deg.max()}")
    print(f"checks: rho mean={s0.rho.mean():.4g} (rho0={m['rho0']}) "
          f"x in [{s0.x.min():.3g},{s0.x.max():.3g}] of L={m['L']}")
