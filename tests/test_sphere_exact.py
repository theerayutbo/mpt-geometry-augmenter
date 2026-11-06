"""Slow end-to-end check: augmenter sphere -> MPT-Calculator -> exact Mie-type solution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mpt_aug.geometry_augmenter import SampleSpec, SamplerConfig, spec_from_unit, validate_spec, write_sample_files
from mpt_aug.runner import harvest_sweep

FREQ = (1, 6, 12)
# Measured on this sphere (mu_r=10, nu_max=1e3, maxh=0.25, order 2, 16k elements):
#   2 prism layers sized for delta(omega_max) only, 7 POD pts : max err 5.7 % (mid-band mesh gap)
#   auto layers (stack spans delta..maxh/2),       7 POD pts : max err 1.4 % (POD, top decade)
#   auto layers,                                   13 POD pts : max err 0.26 %
# So this test uses auto layers + 13 snapshots (~7 min on 2 cores).
POD_POINTS = 13


@pytest.mark.slow
def test_sphere_matches_exact_solution(mpt_root: Path, tmp_path: Path):
    # mu_r = 10 (u[2] maps into the magnetic branch), nu_max = 1e3, no ops.
    cfg = SamplerConfig(log10_nu_max=(3.0, 3.0), n_ops_probs=(1.0,), p_nonmagnetic=0.0, mur_range=(10.0, 10.0),
                        maxh=0.25)
    spec = spec_from_unit([0.5, 0.5, 0.5, 0.5, 0.5], "sphere", seed=20261106, cfg=cfg)
    spec = SampleSpec.from_dict({**spec.to_dict(), "sample_id": "spheretest"})
    assert spec.ops == [] and spec.mur == pytest.approx(10.0)
    assert validate_spec(spec).ok

    py, js = write_sample_files(spec, mpt_root / "OCC_Geometry", tmp_path)
    res_path, log_path = tmp_path / "result.json", tmp_path / "run.log"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    cmd = [sys.executable, "-m", "mpt_aug.runner", "--spec", str(js), "--mpt-root", str(mpt_root),
           "--result", str(res_path), "--log", str(log_path), "--order", "2",
           "--freq", *map(str, FREQ), "--pod-points", str(POD_POINTS), "--cpus", "2"]
    subprocess.run(cmd, env=env, timeout=900, check=False)
    res = json.loads(res_path.read_text())
    assert res["ok"], res.get("error")

    rows = harvest_sweep(Path(res["sweep_dir"]) / "Data")
    sys.path.insert(0, str(mpt_root / "Functions" / "Helper_Functions"))
    from exact_sphere import exact_sphere  # noqa: E402

    # The prism layer is sized for delta(omega_max), so every omega <= omega_max is
    # resolved; check the whole sweep. Sphere => M = m*I, compare each diagonal entry.
    errs = []
    for r in rows:
        ex = exact_sphere(spec.alpha, 0.0, spec.mur, spec.sigma, r["omega"])
        for c in ("xx", "yy", "zz"):
            fem = r[f"M_{c}_re"] + 1j * r[f"M_{c}_im"]
            errs.append(abs(fem - ex) / abs(ex))
        off = max(abs(r[f"M_{c}_re"]) for c in ("xy", "xz", "yz"))
        assert off < 1e-2 * abs(ex)
    print(f"max relative error vs exact sphere: {max(errs):.3e} over {len(rows)} frequencies")
    assert max(errs) < 1e-2
