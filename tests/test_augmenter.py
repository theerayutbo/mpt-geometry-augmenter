"""Fast tests: spec JSON round-trip, determinism, gate rejections, OCC file rendering."""

from __future__ import annotations

import json
import os
import runpy
from dataclasses import replace

import numpy as np
import pytest

from mpt_aug.batch_generator import draw_seed, unit_stream
from mpt_aug.geometry_augmenter import (
    FAMILIES,
    GateConfig,
    SampleSpec,
    SamplerConfig,
    min_feature_report,
    nu_of,
    skin_depth,
    spec_from_unit,
    validate_mesh,
    validate_spec,
    write_sample_files,
)

U = [0.4, 0.6, 0.7, 0.5, 0.5]


def _block(**kw) -> SampleSpec:
    """2 x 1.2 x 0.8 aluminium-ish block with no ops; low mu_r keeps delta small."""
    base = dict(
        sample_id="testblk0",
        seed=0,
        family="cuboid",
        params={"lx": 2.0, "ly": 1.2, "lz": 0.8},
        ops=[],
        mur=1.0,
        sigma=3.5e7,
        alpha=0.01,
        max_frequency=1e6,
        maxh=0.2,
    )
    base.update(kw)
    return SampleSpec(**base)


def _hole(radius: float, offset=(0.0, 0.0)) -> dict:
    return {"kind": "through_hole", "axis": "z", "radius": radius, "offset": list(offset)}


# --------------------------------------------------------------------------- spec
@pytest.mark.parametrize("family", FAMILIES)
def test_spec_json_roundtrip(family):
    spec = spec_from_unit(U, family, seed=123)
    again = SampleSpec.from_json(spec.to_json())
    assert again == spec
    assert json.loads(again.to_json()) == json.loads(spec.to_json())


def test_spec_is_deterministic_given_seed():
    a = spec_from_unit(U, "cuboid", seed=7)
    b = spec_from_unit(U, "cuboid", seed=7)
    c = spec_from_unit(U, "cuboid", seed=8)
    assert a.to_json() == b.to_json()
    assert a.sample_id != c.sample_id


def test_stream_and_seeds_are_deterministic():
    s1 = np.array([next(unit_stream("sobol", 3)) for _ in range(1)])
    s2 = np.array([next(unit_stream("sobol", 3)) for _ in range(1)])
    assert np.array_equal(s1, s2)
    assert draw_seed(0, 5) == draw_seed(0, 5) != draw_seed(1, 5)


def test_similarity_law_parametrisation():
    """alpha is derived so that nu(omega_max) hits the sampled nu_max exactly."""
    cfg = SamplerConfig(log10_nu_max=(3.0, 3.0))
    spec = spec_from_unit(U, "sphere", seed=1, cfg=cfg)
    assert spec.nu_max == pytest.approx(1e3, rel=1e-12)
    assert nu_of(cfg.max_frequency, spec.sigma, spec.alpha) == pytest.approx(1e3, rel=1e-12)
    # delta/alpha = sqrt(2/(mu_r nu_max))
    expected = np.sqrt(2.0 / (spec.mur * spec.nu_max))
    assert spec.delta_min_units == pytest.approx(expected, rel=1e-12)
    assert skin_depth(spec.max_frequency, spec.mur, spec.sigma) == pytest.approx(expected * spec.alpha)


# --------------------------------------------------------------------------- gate
def test_plain_block_passes():
    res = validate_spec(_block())
    assert res.ok, res.reason
    assert res.diagnostics["volume_ratio"] == pytest.approx(1.0)


def test_rejects_hole_larger_than_block():
    res = validate_spec(_block(ops=[_hole(0.7)]))  # block half-width in y is 0.6
    assert not res.ok
    assert "hole_wall" in res.reason


def test_rejects_wall_too_thin():
    # 0.45 radius hole in a 0.6 half-width block leaves a 0.15 wall < 2*maxh = 0.4
    res = validate_spec(_block(ops=[_hole(0.45)]))
    assert not res.ok
    assert "op0_hole_wall" in res.reason


def test_skin_depth_term_can_dominate():
    spec = _block(ops=[_hole(0.2)], maxh=0.15)
    assert validate_spec(spec).ok
    # Very low sigma -> large delta; 3*delta exceeds every feature.
    thick_skin = replace(spec, sigma=1e5, alpha=1e-3)
    rep = min_feature_report(thick_skin)
    assert rep["t_skin"] > rep["t_mesh"]
    assert not validate_spec(thick_skin).ok


def test_rejects_disconnected_result():
    # A "fused" stem that starts 0.2 outside the body (negative overlap) does not
    # touch it -> two solids; the analytic feature rule cannot see this.
    stem = {"kind": "fuse", "primitive": "box", "axis": "x", "side": 1, "size": 0.25, "length": 0.8, "overlap": -0.2}
    res = validate_spec(_block(ops=[stem]))
    assert not res.ok
    assert "single solid" in res.reason


def test_rejects_volume_ratio():
    gate = GateConfig(maxh_factor=0.0, delta_factor=0.0)
    big = {"kind": "notch", "axis": "z", "along": "y", "side": 1, "depth": 0.75, "width": 1.9, "position": 0.0}
    res = validate_spec(_block(ops=[big]), gate)
    assert not res.ok
    assert "volume ratio" in res.reason


def test_rejects_out_of_range_alpha():
    assert "alpha" in validate_spec(_block(alpha=5.0)).reason


def test_mesh_gate():
    spec = _block()
    assert not validate_mesh(spec, {"n_elements": 10**6, "n_prisms": 10}).ok
    assert "no prisms" in validate_mesh(spec, {"n_elements": 100, "n_prisms": 0}).reason
    assert validate_mesh(spec, {"n_elements": 100, "n_prisms": 10}).ok


def test_sampled_specs_mostly_pass_gate():
    """Sampling is feasibility-aware, so the gate should accept the large majority."""
    rng_pts = unit_stream("sobol", 0, pool=64)
    ok = 0
    for k in range(24):
        spec = spec_from_unit(next(rng_pts), FAMILIES[k % 4], draw_seed(0, k))
        ok += validate_spec(spec).ok
    assert ok >= 18


# --------------------------------------------------------------------------- writer
def test_occ_file_is_self_contained(tmp_path):
    """The generated file meshes from its embedded spec without importing mpt_aug."""
    (tmp_path / "OCC_Geometry").mkdir()
    (tmp_path / "VolFiles").mkdir()
    spec = _block(ops=[_hole(0.25, (0.3, 0.0))], maxh=0.4)
    py, js = write_sample_files(spec, tmp_path / "OCC_Geometry", tmp_path / "specs")
    assert SampleSpec.from_json(js.read_text()) == spec
    src = py.read_text()
    assert "mpt_aug" not in src.split("# ---- end inlined mpt_aug/occ_builder.py ----")[1]
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        ns = runpy.run_path(str(py))
    finally:
        os.chdir(cwd)
    assert ns["material_name"] == ["metal"] and ns["mur"] == [1.0] and ns["alpha"] == 0.01
    mats = {ns["nmesh"].GetMaterial(i) for i in range(1, ns["nmesh"].GetNDomains() + 1)}
    assert mats == {"metal", "air"}
    assert ns["MESH_INFO"]["n_prisms"] > 0
    assert (tmp_path / "VolFiles" / "aug_testblk0.vol").exists()
