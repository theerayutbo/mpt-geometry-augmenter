"""Run MPT-Calculator on one sample and harvest its Results/ CSVs.

``run_mpt`` is the in-process call. ``python -m mpt_aug.runner`` is the worker
entry point used by the batch generator to isolate each FEM run in its own
process (netgen/ngsolve can segfault or hang; a subprocess gives crash isolation
and a hard timeout).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .geometry_augmenter import SampleSpec


@contextlib.contextmanager
def redirect_fds(log_path: str | Path) -> Iterator[None]:
    """Send fd 1/2 (incl. C++ netgen/ngsolve output) to ``log_path``."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log_path, "a") as fh:
        os.dup2(fh.fileno(), 1)
        os.dup2(fh.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _num(x: float) -> float | int:
    """MPT-Calculator formats sweep bounds into folder names; keep integers integral."""
    return int(x) if float(x).is_integer() else float(x)


def run_mpt(
    spec: SampleSpec,
    mpt_root: str | Path,
    order: int = 2,
    freq: tuple[float, float, int] = (1, 6, 12),
    pod_points: int = 7,
    cpus: int = 2,
) -> dict[str, Any]:
    """Call MPT-Calculator ``main()`` for an already-written ``OCC_Geometry/aug_<id>.py``.

    ``freq = (log10 omega_start, log10 omega_stop, n_points)``. Returns metadata
    (sweep folder, element/dof counts, wall time). cwd is restored afterwards.
    """
    mpt_root = Path(mpt_root).resolve()
    cwd = os.getcwd()
    os.chdir(mpt_root)
    if str(mpt_root) not in sys.path:
        sys.path.insert(0, str(mpt_root))
    try:
        from main import main  # MPT-Calculator's main.py (needs cwd = root)

        t0 = time.perf_counter()
        out = main(
            h="coarse",
            order=order,
            geometry=spec.geometry_name,
            use_OCC=True,
            use_POD=True,
            use_parallel=False,
            start_stop=(_num(freq[0]), _num(freq[1]), int(freq[2])),
            N_POD_points=pod_points,
            cpus=cpus,
        )
        wall = time.perf_counter() - t0
    finally:
        os.chdir(cwd)
    ndof = out.get("NDOF", ())
    ndof = list(ndof) if isinstance(ndof, (tuple, list)) else [int(ndof)]
    return {
        "sweep_dir": str(mpt_root / "Results" / out["SweepName"]),
        "n_elements": int(out["NElements"]),
        "ndofs": [int(x) for x in ndof],
        "wall_time_s": wall,
    }


# --------------------------------------------------------------------------- harvest
def _read_complex_csv(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append([complex(tok.strip()) for tok in line.split(",")])
    return np.asarray(rows, dtype=complex)


_PAIRS = (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2))


def harvest_sweep(data_dir: str | Path) -> list[dict[str, float]]:
    """Read Frequencies/Tensors/N0/Invariants CSVs into one dict per frequency.

    Tensor convention (MPT-Calculator): M = (N0 + R) + i I, with Tensors.csv holding
    the 9 entries of M row-major. We emit the 6 independent entries of M, the
    eigenvalues of R = Re(M) - N0 and of I = Im(M) (ascending), the 6 entries of N0,
    and the invariants from Invariants.csv.
    """
    d = Path(data_dir)
    omega = np.loadtxt(d / "Frequencies.csv", ndmin=1)
    T = _read_complex_csv(d / "Tensors.csv").reshape(-1, 3, 3)
    N0 = np.loadtxt(d / "N0.csv", delimiter=",").reshape(3, 3)
    inv = _read_complex_csv(d / "Invariants.csv").reshape(-1, 3)
    if not (len(omega) == len(T) == len(inv)):
        raise ValueError(f"inconsistent row counts in {d}: {len(omega)}, {len(T)}, {len(inv)}")
    rows = []
    for k, w in enumerate(omega):
        M = T[k]
        R = 0.5 * ((M.real - N0) + (M.real - N0).T)
        I = 0.5 * (M.imag + M.imag.T)
        row: dict[str, float] = {"omega": float(w)}
        for name, i, j in _PAIRS:
            row[f"M_{name}_re"] = float(M[i, j].real)
            row[f"M_{name}_im"] = float(M[i, j].imag)
        for n, (er, ei) in enumerate(zip(np.linalg.eigvalsh(R), np.linalg.eigvalsh(I)), 1):
            row[f"eig_R_{n}"] = float(er)
            row[f"eig_I_{n}"] = float(ei)
        for name, i, j in _PAIRS:
            row[f"N0_{name}"] = float(N0[i, j])
        for n in range(3):
            row[f"inv_{n + 1}_re"] = float(inv[k, n].real)
            row[f"inv_{n + 1}_im"] = float(inv[k, n].imag)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- worker
def _worker() -> int:
    p = argparse.ArgumentParser(description="MPT-Calculator worker for one sample (internal).")
    p.add_argument("--spec", required=True)
    p.add_argument("--mpt-root", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--freq", type=float, nargs=3, default=(1, 6, 12))
    p.add_argument("--pod-points", type=int, default=7)
    p.add_argument("--cpus", type=int, default=2)
    a = p.parse_args()
    spec = SampleSpec.from_json(Path(a.spec).read_text())
    try:
        with redirect_fds(a.log):
            res = run_mpt(spec, a.mpt_root, a.order, tuple(a.freq), a.pod_points, a.cpus)
        res["ok"] = True
    except Exception as exc:  # report, don't crash the batch
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    Path(a.result).write_text(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_worker())
