"""Batch-generate augmented geometries, run MPT-Calculator, and build a tidy dataset.

Example
-------
    python -m mpt_aug.batch_generator --mpt-root ../ext/MPT-Calculator \\
        --n 5 --out dataset/run1 --order 2 --pod-points 7 --freq 1 6 12 --seed 0

Pipeline per slot i (family = families[i % len(families)], so families are balanced):
  draw next QMC point u + per-draw seed -> SampleSpec -> pre-mesh gate -> mesh
  in-process -> post-mesh gate -> (accept) write OCC file + sidecar -> FEM in a
  subprocess (timeout) -> harvest Results CSVs.
Rejected draws are resampled (next QMC point) up to ``--max-attempts`` per slot.

Outputs in --out:
  samples.jsonl      accepted specs + gate diagnostics + mesh info + run status
  rejections.jsonl   every gate rejection with its reason (for tuning)
  failures.jsonl     FEM/harvest failures, and slots that exhausted their attempts
  mpt_dataset.parquet (or .csv)  long table, one row per (sample, omega)
  specs/aug_<id>.json, logs/aug_<id>.log, run_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.stats import qmc

from .geometry_augmenter import (
    FAMILIES,
    N_UNIT_DIMS,
    GateConfig,
    SampleSpec,
    SamplerConfig,
    mesh_spec,
    nu_of,
    spec_from_unit,
    validate_mesh,
    validate_spec,
    write_sample_files,
)
from .runner import harvest_sweep, redirect_fds


def unit_stream(sampler: str, seed: int, d: int = N_UNIT_DIMS, pool: int = 1024) -> Iterator[np.ndarray]:
    """Yield points of a scrambled Sobol sequence (or an LHS design) in [0,1)^d.

    The pool is drawn once (Sobol: a power of two, so the full pool is balanced);
    rejection-resampling just consumes the next point.
    """
    if sampler == "sobol":
        eng = qmc.Sobol(d, scramble=True, seed=seed)
        m = max(1, math.ceil(math.log2(pool)))
        pts = eng.random_base2(m)
    elif sampler == "lhs":
        pts = qmc.LatinHypercube(d, seed=seed).random(pool)
    elif sampler == "random":
        pts = np.random.default_rng(seed).random((pool, d))
    else:
        raise ValueError(f"unknown sampler {sampler!r}")
    yield from pts


def draw_seed(seed: int, k: int) -> int:
    """Independent per-draw RNG seed derived from (global seed, draw index)."""
    return int(np.random.SeedSequence([seed, k]).generate_state(1, dtype=np.uint64)[0] >> 1)


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


def _run_fem_subprocess(
    spec_path: Path, args: argparse.Namespace, out: Path, sample_id: str
) -> dict[str, Any]:
    res_path = out / "logs" / f"aug_{sample_id}.result.json"
    log_path = out / "logs" / f"aug_{sample_id}.log"
    cmd = [
        sys.executable, "-m", "mpt_aug.runner",
        "--spec", str(spec_path), "--mpt-root", str(args.mpt_root),
        "--result", str(res_path), "--log", str(log_path),
        "--order", str(args.order), "--freq", *map(str, args.freq),
        "--pod-points", str(args.pod_points), "--cpus", str(args.cpus),
    ]  # fmt: skip
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, timeout=args.timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {args.timeout}s", "wall_time_s": time.perf_counter() - t0}
    if res_path.exists():
        res = json.loads(res_path.read_text())
    else:
        res = {"ok": False, "error": f"worker exited {proc.returncode} without result: {proc.stderr[-2000:]}"}
    res.setdefault("wall_time_s", time.perf_counter() - t0)
    res["process_wall_time_s"] = time.perf_counter() - t0
    return res


def _rows_for(spec: SampleSpec, harvested: list[dict[str, float]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "sample_id": spec.sample_id,
        "family": spec.family,
        "ops": ",".join(spec.op_kinds) or "none",
        "n_ops": len(spec.ops),
        "mur": spec.mur,
        "sigma": spec.sigma,
        "alpha": spec.alpha,
        "nu_max": spec.nu_max,
        "maxh": spec.maxh,
        "bl_layers": spec.bl_layers,
        **meta,
    }
    rows = []
    for h in harvested:
        r = dict(base)
        r["omega"] = h["omega"]
        r["nu"] = float(nu_of(h["omega"], spec.sigma, spec.alpha))
        r.update({k: v for k, v in h.items() if k != "omega"})
        rows.append(r)
    return rows


def save_table(rows: list[dict[str, Any]], out: Path) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows)
    try:
        import pyarrow  # noqa: F401

        path = out / "mpt_dataset.parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = out / "mpt_dataset.csv"
        df.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mpt_aug.batch_generator", description=__doc__.split("\n\n")[0])
    p.add_argument("--mpt-root", required=True, type=Path, help="MPT-Calculator checkout (contains main.py)")
    p.add_argument("--n", type=int, default=5, help="number of accepted samples")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--pod-points", type=int, default=7)
    p.add_argument("--freq", type=float, nargs=3, default=(1, 6, 12), metavar=("LOG10_W0", "LOG10_W1", "N"),
                   help="omega sweep: log10 start, log10 stop, points (rad/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    p.add_argument("--sampler", default="sobol", choices=("sobol", "lhs", "random"))
    p.add_argument("--nu-range", type=float, nargs=2, default=(2.0, 4.0), metavar=("LO", "HI"),
                   help="log10 range of nu at omega_max")
    p.add_argument("--maxh", type=float, default=0.3, help="object maxh in object units (max half-extent = 1)")
    p.add_argument("--maxh-min", type=float, default=0.15, help="floor when maxh adapts to thin bodies")
    p.add_argument("--bl-layers", default="auto",
                   help="prism layers (int) or 'auto' (stack spans delta(omega_max)..maxh/2; recommended)")
    p.add_argument("--grading", type=float, default=0.3,
                   help="netgen mesh grading (0.3 default; 0.5 roughly halves the air-dominated element count)")
    p.add_argument("--max-elements", type=int, default=60_000)
    p.add_argument("--delta-factor", type=float, default=3.0, help="t_min >= delta_factor * skin depth")
    p.add_argument("--max-attempts", type=int, default=50, help="gate attempts per slot before giving up")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--timeout", type=float, default=900.0, help="seconds per FEM run")
    p.add_argument("--dry-run", action="store_true", help="generate + gate + mesh only, no FEM")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.mpt_root = args.mpt_root.resolve()
    if not (args.mpt_root / "main.py").exists() or not (args.mpt_root / "OCC_Geometry").is_dir():
        raise SystemExit(f"{args.mpt_root} does not look like an MPT-Calculator checkout")
    out: Path = args.out.resolve()
    for sub in ("specs", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for name in ("samples.jsonl", "rejections.jsonl", "failures.jsonl"):
        (out / name).unlink(missing_ok=True)

    scfg = SamplerConfig(
        families=tuple(args.families),
        max_frequency=10.0 ** args.freq[1],
        log10_nu_max=tuple(args.nu_range),
        maxh=args.maxh,
        maxh_min=args.maxh_min,
        delta_factor=args.delta_factor,
        bl_layers=None if str(args.bl_layers) == "auto" else int(args.bl_layers),
        mesh_grading=args.grading,
    )
    gcfg = GateConfig(max_elements=args.max_elements, delta_factor=args.delta_factor)
    pool = max(64, args.n * args.max_attempts)
    stream = unit_stream(args.sampler, args.seed, pool=pool)

    draws = accepted = 0
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for slot in range(args.n):
        family = args.families[slot % len(args.families)]
        spec = None
        for _ in range(args.max_attempts):
            u = next(stream)
            spec_i = spec_from_unit(u, family, draw_seed(args.seed, draws), scfg)
            draws += 1
            g = validate_spec(spec_i, gcfg)
            if g.ok:
                t0 = time.perf_counter()
                try:
                    with redirect_fds(out / "logs" / "meshing.log"):
                        _, info = mesh_spec(spec_i.to_dict())
                except Exception as exc:
                    g.ok, g.reason = False, f"meshing failed: {exc}"
                else:
                    g.diagnostics["mesh"] = info
                    g.diagnostics["mesh_time_s"] = time.perf_counter() - t0
                    gm = validate_mesh(spec_i, info, gcfg)
                    if not gm.ok:
                        g.ok, g.reason = False, gm.reason
            if g.ok:
                spec = spec_i
                break
            _append_jsonl(out / "rejections.jsonl", {"slot": slot, "reason": g.reason, "spec": spec_i.to_dict()})
        if spec is None:
            _append_jsonl(out / "failures.jsonl", {"slot": slot, "family": family, "stage": "gate",
                                                   "error": f"no valid spec in {args.max_attempts} attempts"})
            print(f"[slot {slot}] {family}: gave up after {args.max_attempts} attempts", flush=True)
            continue
        accepted += 1
        mesh = g.diagnostics["mesh"]
        rec: dict[str, Any] = {"slot": slot, "spec": spec.to_dict(), "nu_max": spec.nu_max, "gate": g.diagnostics}
        msg = (f"[slot {slot}] {spec.sample_id} {family:9s} ops={','.join(spec.op_kinds) or 'none':24s} "
               f"mur={spec.mur:7.2f} nu_max={spec.nu_max:9.1f} elements={mesh['n_elements']} "
               f"(prisms {mesh['n_prisms']}) vol_ratio={g.diagnostics['volume_ratio']:.3f}")
        print(msg, flush=True)

        if args.dry_run:
            rec["status"] = "dry_run"
            _append_jsonl(out / "samples.jsonl", rec)
            summary.append({"sample_id": spec.sample_id, "family": family, "ops": spec.op_kinds, **mesh})
            continue

        _, spec_path = write_sample_files(spec, args.mpt_root / "OCC_Geometry", out / "specs")
        res = _run_fem_subprocess(spec_path, args, out, spec.sample_id)
        rec["fem"] = res
        if res.get("ok"):
            try:
                harvested = harvest_sweep(Path(res["sweep_dir"]) / "Data")
                ndofs = res.get("ndofs", [])
                meta = {
                    "n_elements": res["n_elements"],
                    "n_prisms": mesh["n_prisms"],
                    "ndofs_0": ndofs[0] if ndofs else None,
                    "ndofs_1": ndofs[1] if len(ndofs) > 1 else None,
                    "order": args.order,
                    "pod_points": args.pod_points,
                    "wall_time_s": res["wall_time_s"],
                }
                rows.extend(_rows_for(spec, harvested, meta))
                rec["status"] = "ok"
            except Exception as exc:
                res["ok"], res["error"] = False, f"harvest failed: {exc}"
        if not res.get("ok"):
            rec["status"] = "failed"
            _append_jsonl(out / "failures.jsonl", {"slot": slot, "sample_id": spec.sample_id, "stage": "fem",
                                                   "error": res.get("error"), "traceback": res.get("traceback")})
        _append_jsonl(out / "samples.jsonl", rec)
        print(f"    -> {rec['status']} in {res.get('wall_time_s', float('nan')):.1f}s "
              f"(FEM elements={res.get('n_elements')}, ndofs={res.get('ndofs')})"
              + (f" ERROR {res.get('error')}" if not res.get("ok") else ""), flush=True)
        summary.append({"sample_id": spec.sample_id, "family": family, "ops": spec.op_kinds,
                        "status": rec["status"], "wall_time_s": res.get("wall_time_s"), **mesh})

    table = save_table(rows, out) if rows else None
    run = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "sampler_config": asdict(scfg),
        "gate_config": asdict(gcfg),
        "draws": draws,
        "accepted": accepted,
        "acceptance_rate": accepted / draws if draws else float("nan"),
        "total_wall_time_s": time.perf_counter() - t_start,
        "table": str(table) if table else None,
        "samples": summary,
    }
    (out / "run_summary.json").write_text(json.dumps(run, indent=2, default=float))
    print(f"accepted {accepted}/{draws} draws (acceptance {run['acceptance_rate']:.2%}); "
          f"{len(rows)} rows -> {table}; summary {out / 'run_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # scipy Sobol balance warnings
        raise SystemExit(main())
