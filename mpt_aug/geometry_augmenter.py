"""Parametric + irregular metal geometries for MPT-Calculator, with a validity gate.

Physics that shapes the design
------------------------------
The MPT of a conductor obeys the similarity law

    M(alpha, omega, sigma, mu_r, B) = alpha^3 * M_hat(nu, mu_r, B),   nu = omega*mu0*sigma*alpha^2

so for a fixed dimensionless shape ``B`` only ``(nu, mu_r)`` matter. We therefore
sample ``log10(nu_max)`` (nu at the top of the frequency sweep) and ``mu_r``
directly. ``sigma`` is drawn from a realistic metal range purely so the metadata
looks like a real object; ``alpha`` is then *derived*,
``alpha = sqrt(nu_max / (omega_max*mu0*sigma))``, so sigma and alpha never vary
independently of nu. In object units the skin depth at omega_max is
``delta/alpha = sqrt(2 / (mu_r*nu_max))``, which is what sets the prism-layer
thickness and the skin-depth part of the minimum-feature rule.

Shapes are normalised so the largest half-extent of the base primitive is 1 unit;
``alpha`` is then (approximately) the physical half-size in metres.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import occ_builder
from .occ_builder import AXES, MU0, build_solid, half_extents, mesh_spec  # re-exported

__all__ = [
    "FAMILIES",
    "OPS",
    "SampleSpec",
    "SamplerConfig",
    "GateConfig",
    "GateResult",
    "skin_depth",
    "nu_of",
    "spec_from_unit",
    "sample_spec",
    "min_feature_report",
    "validate_spec",
    "validate_mesh",
    "build_solid",
    "mesh_spec",
    "render_occ_file",
    "write_sample_files",
    "N_UNIT_DIMS",
]

FAMILIES: tuple[str, ...] = ("sphere", "cylinder", "cuboid", "ellipsoid")
OPS: tuple[str, ...] = ("aspect", "corner_cut", "through_hole", "notch", "fuse")
_OP_ORDER = {k: i for i, k in enumerate(("aspect", "fuse", "corner_cut", "notch", "through_hole"))}
#: Dimension of the unit-hypercube point consumed by :func:`spec_from_unit`:
#: (shape u1, shape u2, mu_r, log nu_max, log sigma).
N_UNIT_DIMS = 5


# --------------------------------------------------------------------------- physics
def skin_depth(omega: float, mur: float, sigma: float) -> float:
    """Skin depth in metres, delta = sqrt(2 / (omega*mu0*mu_r*sigma))."""
    return math.sqrt(2.0 / (omega * MU0 * mur * sigma))


def nu_of(omega: float | np.ndarray, sigma: float, alpha: float) -> float | np.ndarray:
    """MPT-Calculator's dimensionless frequency nu = omega*mu0*sigma*alpha^2."""
    return omega * MU0 * sigma * alpha**2


# --------------------------------------------------------------------------- spec
@dataclass
class SampleSpec:
    """Everything needed to rebuild, mesh and solve one sample. JSON round-trippable.

    Lengths (``params``, op geometry, ``maxh``, ``air_half``) are in object units;
    ``alpha`` is metres per unit. ``max_frequency`` is the *angular* frequency
    omega_max [rad/s] at the top of the sweep (MPT-Calculator sweeps omega).
    """

    sample_id: str
    seed: int
    family: str
    params: dict[str, float]
    ops: list[dict[str, Any]]
    mur: float
    sigma: float
    alpha: float
    max_frequency: float
    maxh: float = 0.3
    maxh_air: float = 1000.0
    air_half: float = 1000.0
    bl_layers: int = 2
    bl_cap: float = 0.25
    material_name: str = "metal"
    unit_point: list[float] | None = None

    @property
    def geometry_name(self) -> str:
        """OCC file name; MPT-Calculator keys all outputs by it."""
        return f"aug_{self.sample_id}.py"

    @property
    def nu_max(self) -> float:
        return float(nu_of(self.max_frequency, self.sigma, self.alpha))

    @property
    def delta_min_units(self) -> float:
        """Skin depth at omega_max in object units (= sqrt(2/(mu_r*nu_max)))."""
        return skin_depth(self.max_frequency, self.mur, self.sigma) / self.alpha

    @property
    def op_kinds(self) -> list[str]:
        return [o["kind"] for o in self.ops]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SampleSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "SampleSpec":
        return cls.from_dict(json.loads(s))


@dataclass
class SamplerConfig:
    """Ranges for spec sampling. Defaults target cheap but non-trivial FEM runs."""

    families: tuple[str, ...] = FAMILIES
    max_frequency: float = 1e6
    log10_nu_max: tuple[float, float] = (2.0, 4.0)
    log10_sigma: tuple[float, float] = (6.0, math.log10(6e7))
    p_nonmagnetic: float = 0.4  # P(mu_r = 1); otherwise log-uniform in mur_range
    mur_range: tuple[float, float] = (2.0, 100.0)
    min_aspect: float = 0.3  # smallest secondary half-extent of the base primitive
    n_ops_probs: tuple[float, ...] = (0.2, 0.5, 0.3)  # P(0 ops), P(1), P(2)
    op_weights: dict[str, float] = field(
        default_factory=lambda: {
            "aspect": 1.0,
            "corner_cut": 1.0,
            "through_hole": 1.0,
            "notch": 1.0,
            "fuse": 1.0,
        }
    )
    maxh: float = 0.3  # upper bound on object maxh
    maxh_min: float = 0.15  # lower bound when adapting maxh to thin bodies
    maxh_factor: float = 2.0  # sampler's copy of the gate floors (hints only)
    delta_factor: float = 3.0
    bl_layers: int = 2
    bl_cap: float = 0.25


@dataclass
class GateConfig:
    """Validity-gate thresholds.

    Minimum feature rule: t_min = max(maxh_factor*maxh, delta_factor*delta_min/alpha).
    See :func:`min_feature_report` for what is (and is not) measured.
    """

    vol_ratio: tuple[float, float] = (0.3, 1.0)
    maxh_factor: float = 2.0
    delta_factor: float = 3.0
    max_elements: int = 60_000
    alpha_range: tuple[float, float] = (5e-4, 0.2)
    sigma_range: tuple[float, float] = (1e5, 1e8)
    air_fraction: float = 0.05  # object bbox must lie inside air_fraction * air box


@dataclass
class GateResult:
    ok: bool
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- sampling
def _lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * float(u)


def _sample_id(seed: int) -> str:
    return hashlib.sha1(str(seed).encode()).hexdigest()[:8]


def _base_params(family: str, u1: float, u2: float, min_aspect: float) -> dict[str, float]:
    if family == "sphere":
        return {"r": 1.0}
    if family == "cylinder":
        rho = _lerp(0.25, 1.0, u1)
        # u2 picks rod (long axis z) vs disc (radius dominant).
        return {"r": rho, "h": 2.0} if u2 < 0.5 else {"r": 1.0, "h": 2.0 * rho}
    if family == "cuboid":
        return {"lx": 2.0, "ly": 2.0 * _lerp(min_aspect, 1.0, u1), "lz": 2.0 * _lerp(min_aspect, 1.0, u2)}
    if family == "ellipsoid":
        return {"a": 1.0, "b": _lerp(min_aspect, 1.0, u1), "c": _lerp(min_aspect, 1.0, u2)}
    raise ValueError(f"unknown family {family!r}")


def _apply_aspect(family: str, params: dict[str, float], f: Sequence[float]) -> dict[str, float]:
    p = dict(params)
    if family == "cylinder":
        p["r"] *= f[0]
        p["h"] *= f[2]
    elif family == "cuboid":
        p["lx"], p["ly"], p["lz"] = p["lx"] * f[0], p["ly"] * f[1], p["lz"] * f[2]
    elif family == "ellipsoid":
        p["a"], p["b"], p["c"] = p["a"] * f[0], p["b"] * f[1], p["c"] * f[2]
    else:
        raise ValueError(f"aspect op not defined for {family!r}")
    return p


def _support(family: str, H: Sequence[float], n: Sequence[float]) -> float:
    """Support function h(n) = max_{x in body} n.x of the base primitive."""
    if family in ("sphere", "ellipsoid"):
        return math.sqrt(sum((ni * hi) ** 2 for ni, hi in zip(n, H)))
    if family == "cylinder":
        return H[0] * math.hypot(n[0], n[1]) + H[2] * abs(n[2])
    return sum(abs(ni) * hi for ni, hi in zip(n, H))


def _local_half(family: str, H: Sequence[float], a: int, offsets: dict[int, float]) -> float:
    """Half-thickness of the base body along axis ``a`` at the given offsets on other axes."""
    if family in ("sphere", "ellipsoid"):
        s = 1.0 - sum((o / H[i]) ** 2 for i, o in offsets.items())
        return H[a] * math.sqrt(max(s, 0.0))
    if family == "cylinder":
        radial = {i: o for i, o in offsets.items() if i != 2}
        if a == 2:
            return H[2] if math.hypot(*radial.values() or [0.0]) <= H[0] else 0.0
        other = [o for i, o in radial.items()]
        return math.sqrt(max(H[0] ** 2 - sum(o * o for o in other), 0.0))
    return H[a]


def _sample_op(
    kind: str,
    family: str,
    H: Sequence[float],
    rng: np.random.Generator,
    t_min: float,
    t_mesh: float,
) -> dict[str, Any] | None:
    """Draw one op's parameters inside the range the gate can accept.

    ``t_min``/``t_mesh`` are the gate's feature floors (see :func:`min_feature_report`);
    sampling within them keeps the gate from silently biasing the dataset against
    holes/notches. Returns ``None`` if the body is too small for this op. The gate
    remains the authority (curved-body sections, op interactions, OCC failures).
    """
    if kind == "corner_cut":
        w = rng.uniform(0.3, 1.0, 3) * rng.choice([-1.0, 1.0], 3)
        n = w / np.linalg.norm(w)
        frac = float(rng.uniform(0.1, 0.35))
        return {
            "kind": kind,
            "normal": [float(x) for x in n],
            "frac": frac,
            "offset": (1.0 - frac) * _support(family, H, n),
        }
    if kind == "through_hole":
        # Prefer drilling along the thin direction (washer-like), weight ~ min perp extent.
        perp_min = np.array([min(H[k] for k in range(3) if k != a) for a in range(3)])
        r_hi = perp_min - t_min
        ok = r_hi >= 0.5 * t_mesh
        if not ok.any():
            return None
        wts = np.where(ok, perp_min, 0.0)
        a = int(rng.choice(3, p=wts / wts.sum()))
        i, j = [k for k in range(3) if k != a]
        r_lo = 0.5 * t_mesh
        r = r_lo + float(rng.uniform(0.0, 0.7)) * (r_hi[a] - r_lo)
        off = [float(rng.choice([-1, 1]) * rng.uniform(0.0, 0.5) * max(H[k] - r - t_min, 0.0)) for k in (i, j)]
        return {"kind": kind, "axis": "xyz"[a], "radius": r, "offset": off}
    if kind == "notch":
        a = int(rng.integers(3))
        b, c = sorted((k for k in range(3) if k != a), key=lambda k: H[k])  # c = wider axis
        depth_hi = min(0.4 * 2 * H[a], 2 * H[a] - t_min)
        width_hi = min(0.3 * 2 * H[c], 2 * H[c] - 2 * t_min)
        depth_lo, width_lo = 0.1 * 2 * H[a], t_mesh
        if depth_hi < depth_lo or width_hi < width_lo:
            return None
        depth = float(rng.uniform(depth_lo, depth_hi))
        width = float(rng.uniform(width_lo, width_hi))
        slack = H[c] - 0.5 * width - t_min
        pos = float(rng.choice([-1, 1]) * rng.uniform(0.0, 0.5) * slack)
        return {
            "kind": kind,
            "axis": "xyz"[a],
            "along": "xyz"[b],
            "side": int(rng.choice([-1, 1])),
            "depth": depth,
            "width": width,
            "position": pos,
        }
    if kind == "fuse":
        a = int(rng.integers(3))
        b, c = [k for k in range(3) if k != a]
        prim = str(rng.choice(["cylinder", "box", "sphere"]))
        size_lo, size_hi = 0.5 * t_min, 0.5 * min(H[b], H[c])
        if size_hi < size_lo:
            return None
        size = float(rng.uniform(max(size_lo, 0.25 * min(H[b], H[c])), size_hi)) if size_hi > size_lo else size_lo
        length = size if prim == "sphere" else float(rng.uniform(0.3, 0.8)) * max(H)
        # Penetrate deep enough that the stem's rim is inside a curved body.
        rim = min(_local_half(family, H, a, {b: size}), _local_half(family, H, a, {c: size}))
        overlap = (H[a] - rim) + float(rng.uniform(0.1, 0.3)) * H[a]
        return {
            "kind": kind,
            "primitive": prim,
            "axis": "xyz"[a],
            "side": int(rng.choice([-1, 1])),
            "size": size,
            "length": length,
            "overlap": overlap,
        }
    raise ValueError(f"unknown op {kind!r}")


def spec_from_unit(
    u: Sequence[float],
    family: str,
    seed: int,
    cfg: SamplerConfig | None = None,
) -> SampleSpec:
    """Map a unit-hypercube point (from Sobol/LHS) plus a per-sample RNG seed to a spec.

    Continuous, physics-relevant dims come from ``u`` (so QMC stratifies them):
    ``u = (shape1, shape2, mu_r, log nu_max, log sigma)``. Discrete structure and op
    parameters come from ``numpy.random.default_rng(seed)``.
    """
    cfg = cfg or SamplerConfig()
    if len(u) != N_UNIT_DIMS:
        raise ValueError(f"expected {N_UNIT_DIMS} unit dims, got {len(u)}")
    rng = np.random.default_rng(seed)
    params = _base_params(family, u[0], u[1], cfg.min_aspect)

    # mu_r: point mass at 1 (non-magnetic metals) + log-uniform magnetic branch.
    if u[2] < cfg.p_nonmagnetic:
        mur = 1.0
    else:
        v = (u[2] - cfg.p_nonmagnetic) / (1.0 - cfg.p_nonmagnetic)
        mur = float(10 ** _lerp(*np.log10(cfg.mur_range), v))
    nu_max = 10 ** _lerp(*cfg.log10_nu_max, u[3])
    sigma = 10 ** _lerp(*cfg.log10_sigma, u[4])
    alpha = math.sqrt(nu_max / (cfg.max_frequency * MU0 * sigma))

    n_ops = int(rng.choice(len(cfg.n_ops_probs), p=np.asarray(cfg.n_ops_probs) / sum(cfg.n_ops_probs)))
    kinds = [k for k in OPS if not (k == "aspect" and family == "sphere")]
    w = np.array([cfg.op_weights.get(k, 0.0) for k in kinds], dtype=float)
    # Random priority order over all kinds; take the first n_ops that are feasible.
    order = [str(k) for k in rng.choice(kinds, size=int((w > 0).sum()), replace=False, p=w / w.sum())]

    delta_units = math.sqrt(2.0 / (mur * nu_max))  # skin depth at omega_max / alpha
    if "aspect" in order[:n_ops]:
        f = [float(x) for x in np.exp(rng.uniform(-0.2, 0.2, 3))]
        params = _apply_aspect(family, params, f)
    H = half_extents(family, params)
    # maxh adapts down so the thinnest base dimension spans >= maxh_factor elements.
    maxh = max(cfg.maxh_min, min(cfg.maxh, 2 * min(H) / cfg.maxh_factor))
    t_mesh = cfg.maxh_factor * maxh
    t_min = max(t_mesh, cfg.delta_factor * delta_units)

    ops: list[dict[str, Any]] = []
    for kind in order:
        if len(ops) >= n_ops:
            break
        if kind == "aspect":
            if "aspect" in order[:n_ops]:
                ops.append({"kind": "aspect", "factors": f})
            continue
        op = _sample_op(kind, family, H, rng, t_min, t_mesh)
        if op is not None:
            ops.append(op)
    ops.sort(key=lambda o: _OP_ORDER[o["kind"]])

    return SampleSpec(
        sample_id=_sample_id(seed),
        seed=int(seed),
        family=family,
        params={k: float(v) for k, v in params.items()},
        ops=ops,
        mur=mur,
        sigma=float(sigma),
        alpha=float(alpha),
        max_frequency=float(cfg.max_frequency),
        maxh=float(maxh),
        bl_layers=cfg.bl_layers,
        bl_cap=cfg.bl_cap,
        unit_point=[float(x) for x in u],
    )


def sample_spec(rng: np.random.Generator, family: str | None = None, cfg: SamplerConfig | None = None) -> SampleSpec:
    """Plain Monte-Carlo convenience wrapper: draw u and a seed from ``rng``."""
    cfg = cfg or SamplerConfig()
    fam = family or str(rng.choice(cfg.families))
    return spec_from_unit(rng.random(N_UNIT_DIMS), fam, int(rng.integers(2**63 - 1)), cfg)


# --------------------------------------------------------------------------- gate
def _section_wall(family: str, H: Sequence[float], a: int, centre: Sequence[float], r: float) -> float:
    """Ligament between a hole (axis ``a``, radius r) and the outer surface, measured in
    the mid cross-section perpendicular to the hole. Exact for rectangles/ellipses."""
    i, j = [k for k in range(3) if k != a]
    ci, cj = centre
    elliptic = family in ("sphere", "ellipsoid") or (family == "cylinder" and a == 2)
    if not elliptic:  # rectangle (cuboid, or a cylinder cut perpendicular to its axis)
        return min(H[i] - abs(ci), H[j] - abs(cj)) - r
    t = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
    xi, xj = H[i] * np.cos(t), H[j] * np.sin(t)
    if (ci / H[i]) ** 2 + (cj / H[j]) ** 2 >= 1.0:
        return -r
    return float(np.min(np.hypot(xi - ci, xj - cj))) - r


def min_feature_report(spec: SampleSpec, gate: GateConfig | None = None) -> dict[str, Any]:
    """Estimate thin features from the spec parameters (no geometry kernel involved).

    Checked, in object units:
      * base thickness: the smallest full extent of the primitive;
      * through_hole: mid-section ligament between hole and outer surface, and the
        hole diameter (must be meshable: >= maxh_factor*maxh);
      * notch: remaining thickness under the notch at its position, the side
        ligaments next to it (a ligament of exactly 0 is a step, not a wall, but we
        still require >= t_min to stay conservative), and the slot width (an air
        gap: only the maxh rule applies);
      * fuse: stem diameter / side.
    NOT checked: interactions between ops (e.g. a corner cut passing close to a
    hole), wedge-shaped edges where a hole exits a curved surface, and the
    corner-cut sliver itself. The post-build single-solid/volume checks and the
    post-mesh element cap are the backstop for those.
    """
    gate = gate or GateConfig()
    H = half_extents(spec.family, spec.params)
    t_mesh = gate.maxh_factor * spec.maxh
    t_skin = gate.delta_factor * spec.delta_min_units
    t_min = max(t_mesh, t_skin)
    feats: list[tuple[str, float, float]] = [("base_thickness", 2 * min(H), t_min)]
    for k, op in enumerate(spec.ops):
        kind = op["kind"]
        if kind == "through_hole":
            a = AXES[op["axis"]]
            feats.append((f"op{k}_hole_wall", _section_wall(spec.family, H, a, op["offset"], op["radius"]), t_min))
            feats.append((f"op{k}_hole_diameter", 2 * op["radius"], t_mesh))
        elif kind == "notch":
            a, b = AXES[op["axis"]], AXES[op["along"]]
            c = 3 - a - b
            half_loc = _local_half(spec.family, H, a, {c: op["position"]})
            cut = max(0.0, op["depth"] - (H[a] - half_loc))
            feats.append((f"op{k}_notch_floor", 2 * half_loc - cut, t_min))
            lig = H[c] - abs(op["position"]) - 0.5 * op["width"]
            feats.append((f"op{k}_notch_ligament", lig, t_min))
            feats.append((f"op{k}_notch_width", op["width"], t_mesh))
        elif kind == "fuse":
            feats.append((f"op{k}_fuse_size", 2 * op["size"], t_min))
    worst = min(feats, key=lambda f: f[1] - f[2])
    return {
        "t_min": t_min,
        "t_mesh": t_mesh,
        "t_skin": t_skin,
        "features": {n: v for n, v, _ in feats},
        "worst": worst[0],
        "worst_margin": worst[1] - worst[2],
        "violations": [f"{n}={v:.4g}<{req:.4g}" for n, v, req in feats if v < req],
    }


def validate_spec(spec: SampleSpec, gate: GateConfig | None = None) -> GateResult:
    """Pre-mesh gate: physical ranges, min-feature heuristic, then OCC build checks."""
    gate = gate or GateConfig()
    diag: dict[str, Any] = {"nu_max": spec.nu_max, "delta_min_units": spec.delta_min_units}
    if not gate.sigma_range[0] <= spec.sigma <= gate.sigma_range[1]:
        return GateResult(False, f"sigma {spec.sigma:.3g} outside {gate.sigma_range}", diag)
    if not gate.alpha_range[0] <= spec.alpha <= gate.alpha_range[1]:
        return GateResult(False, f"alpha {spec.alpha:.3g} outside {gate.alpha_range}", diag)

    feat = min_feature_report(spec, gate)
    diag["min_feature"] = feat
    if feat["violations"]:
        return GateResult(False, "thin feature: " + "; ".join(feat["violations"]), diag)

    d = spec.to_dict()
    try:
        additive = build_solid(d, stage="additive")
        solid = build_solid(d)
    except Exception as exc:  # OCC raises a zoo of exception types
        return GateResult(False, f"occ build failed: {exc}", diag)
    n_solids = len(solid.solids)
    diag["n_solids"] = n_solids
    if n_solids != 1:
        return GateResult(False, f"not a single solid after booleans (n_solids={n_solids})", diag)
    v0, v = additive.mass, solid.mass
    ratio = v / v0 if v0 > 0 else float("nan")
    diag.update(volume=v, volume_additive=v0, volume_ratio=ratio)
    lo, hi = gate.vol_ratio
    if not (lo <= ratio <= hi + 1e-9):
        return GateResult(False, f"volume ratio {ratio:.3f} outside [{lo}, {hi}]", diag)
    (x0, y0, z0), (x1, y1, z1) = solid.bounding_box
    ext = max(abs(c) for c in (x0, y0, z0, x1, y1, z1))
    diag["bbox"] = [[x0, y0, z0], [x1, y1, z1]]
    if ext > gate.air_fraction * spec.air_half:
        return GateResult(False, f"bbox extent {ext:.3g} exceeds {gate.air_fraction}*air_half", diag)
    return GateResult(True, "", diag)


def validate_mesh(spec: SampleSpec, info: dict[str, int], gate: GateConfig | None = None) -> GateResult:
    """Post-mesh gate: element cap and a non-empty prism layer (if layers requested)."""
    gate = gate or GateConfig()
    if info["n_elements"] > gate.max_elements:
        return GateResult(False, f"too many elements {info['n_elements']} > {gate.max_elements}", dict(info))
    if spec.bl_layers > 0 and info["n_prisms"] == 0:
        return GateResult(False, "boundary layer requested but mesh has no prisms", dict(info))
    return GateResult(True, "", dict(info))


# --------------------------------------------------------------------------- writer
_TEMPLATE = '''"""Auto-generated by mpt-geometry-augmenter - DO NOT EDIT.

sample_id: {sample_id}   family: {family}   ops: {ops}
mu_r={mur:.6g}  sigma={sigma:.6g} S/m  alpha={alpha:.6g} m  omega_max={omega:.6g} rad/s  nu_max={nu:.6g}

Self-contained: the netgen-OCC builder below is inlined verbatim from
mpt_aug/occ_builder.py, and SPEC_JSON is the full SampleSpec. Executed by
MPT-Calculator (Functions/MeshMaking/Generate_From_Python.py) via runpy with
cwd = MPT-Calculator root; it must leave material_name, mur, sigma, alpha, nmesh.
"""
# ---- begin inlined mpt_aug/occ_builder.py ----
{builder_src}
# ---- end inlined mpt_aug/occ_builder.py ----

import json as _json

SPEC_JSON = {spec_json!r}
SPEC = _json.loads(SPEC_JSON)

material_name = [SPEC["material_name"]]
mur = [SPEC["mur"]]
sigma = [SPEC["sigma"]]
alpha = SPEC["alpha"]

nmesh, MESH_INFO = mesh_spec(SPEC, vol_path="VolFiles/{stem}.vol")
print(f"[mpt-aug] {stem}: {{MESH_INFO}}")
'''


def render_occ_file(spec: SampleSpec) -> str:
    """Source text of the self-contained OCC geometry file for ``spec``."""
    builder_src = inspect.getsource(occ_builder).rstrip()
    return _TEMPLATE.format(
        sample_id=spec.sample_id,
        family=spec.family,
        ops=",".join(spec.op_kinds) or "none",
        mur=spec.mur,
        sigma=spec.sigma,
        alpha=spec.alpha,
        omega=spec.max_frequency,
        nu=spec.nu_max,
        builder_src=builder_src,
        spec_json=spec.to_json(sort_keys=True),
        stem=Path(spec.geometry_name).stem,
    )


def write_sample_files(spec: SampleSpec, occ_dir: str | Path, sidecar_dir: str | Path) -> tuple[Path, Path]:
    """Write ``<occ_dir>/aug_<id>.py`` and ``<sidecar_dir>/aug_<id>.json``.

    ``occ_dir`` must be MPT-Calculator's ``OCC_Geometry`` folder (path is hardcoded
    there). The JSON sidecar is kept out of the MPT-Calculator checkout.
    """
    occ_dir, sidecar_dir = Path(occ_dir), Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    py = occ_dir / spec.geometry_name
    js = sidecar_dir / (Path(spec.geometry_name).stem + ".json")
    py.write_text(render_occ_file(spec))
    js.write_text(spec.to_json(indent=2, sort_keys=True) + "\n")
    return py, js
