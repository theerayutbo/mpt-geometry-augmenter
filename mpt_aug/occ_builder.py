"""Netgen-OCC construction and meshing of an augmented sample from a plain spec dict.

This module is deliberately dependency-light (stdlib + netgen only) because its
source is inlined verbatim into every generated ``OCC_Geometry/aug_<id>.py`` file.
That makes each geometry file self-contained: MPT-Calculator executes it with
``runpy`` and it rebuilds exactly the solid that the augmenter validated, without
needing ``mpt_aug`` to be importable from the MPT-Calculator process.

Conventions
-----------
* Geometry is in dimensionless object units; the physical size is ``alpha`` metres
  per unit (MPT-Calculator's scaling). Base primitives are centred at the origin
  and normalised so that their largest half-extent is ~1.
* ``spec`` is the JSON form of :class:`mpt_aug.geometry_augmenter.SampleSpec`.
* Op order is canonical: additive ops (``fuse``) first, then subtractive ops in
  list order. ``aspect`` is provenance only - its factors are already baked into
  ``spec["params"]``.
"""

import math

from netgen.meshing import BoundaryLayerParameters
from netgen.occ import (
    X,
    Y,
    Z,
    Axes,
    Axis,
    Box,
    Cylinder,
    Ellipsoid,
    Glue,
    OCCGeometry,
    Pnt,
    Sphere,
    Vec,
)

MU0 = 4.0e-7 * math.pi
AXES = {"x": 0, "y": 1, "z": 2}
_DIRS = (X, Y, Z)


def half_extents(family, params):
    """Half-extents (hx, hy, hz) of the base primitive in object units."""
    if family == "sphere":
        r = params["r"]
        return (r, r, r)
    if family == "cylinder":  # axis along z
        return (params["r"], params["r"], 0.5 * params["h"])
    if family == "cuboid":
        return (0.5 * params["lx"], 0.5 * params["ly"], 0.5 * params["lz"])
    if family == "ellipsoid":
        return (params["a"], params["b"], params["c"])
    raise ValueError(f"unknown family {family!r}")


def _box(lo, hi):
    return Box(Pnt(*lo), Pnt(*hi))


def base_solid(family, params):
    """Primitive solid for a shape family, centred at the origin."""
    if family == "sphere":
        return Sphere(Pnt(0, 0, 0), r=params["r"])
    if family == "cylinder":
        h = params["h"]
        return Cylinder(Pnt(0, 0, -0.5 * h), Z, r=params["r"], h=h)
    if family == "cuboid":
        hx, hy, hz = half_extents(family, params)
        return _box((-hx, -hy, -hz), (hx, hy, hz))
    if family == "ellipsoid":
        # Native OCC ellipsoid (exact quadric) rather than a GTrsf-scaled sphere,
        # which OCC would convert to a B-spline surface.
        return Ellipsoid(Axes(Pnt(0, 0, 0), Z, X), params["a"], params["b"], params["c"])
    raise ValueError(f"unknown family {family!r}")


def _slab_along(axis, lo, hi, across, centre=(0.0, 0.0, 0.0), widths=None):
    """Axis-aligned box spanning [lo, hi] along ``axis`` and +-across elsewhere.

    ``widths`` optionally overrides the half-width on the other two axes
    (dict axis_index -> (centre, half_width)).
    """
    a = AXES[axis]
    lo3 = [c - across for c in centre]
    hi3 = [c + across for c in centre]
    lo3[a], hi3[a] = lo, hi
    for i, (c, hw) in (widths or {}).items():
        lo3[i], hi3[i] = c - hw, c + hw
    return _box(lo3, hi3)


def op_solid(op, H):
    """Tool solid for one op, given the base half-extents ``H``."""
    big = 4.0 * max(H) + 1.0
    kind = op["kind"]
    if kind == "corner_cut":
        # Half-space-like box whose inner face is the plane n.x = d.
        n = op["normal"]
        d = op["offset"]
        tool = _box((-big, -big, 0.0), (big, big, 2 * big))
        nz = max(-1.0, min(1.0, n[2]))
        ang = math.degrees(math.acos(nz))
        rx, ry = -n[1], n[0]  # z x n
        if math.hypot(rx, ry) > 1e-12 and ang > 1e-9:
            tool = tool.Rotate(Axis(Pnt(0, 0, 0), Vec(rx, ry, 0.0)), ang)
        elif nz < 0:
            tool = tool.Rotate(Axis(Pnt(0, 0, 0), X), 180.0)
        return tool.Move(Vec(d * n[0], d * n[1], d * n[2]))
    if kind == "through_hole":
        a = AXES[op["axis"]]
        p = [0.0, 0.0, 0.0]
        others = [i for i in range(3) if i != a]
        p[others[0]], p[others[1]] = op["offset"]
        p[a] = -big
        return Cylinder(Pnt(*p), _DIRS[a], r=op["radius"], h=2 * big)
    if kind == "notch":
        a, b = AXES[op["axis"]], AXES[op["along"]]
        c = 3 - a - b
        side = op["side"]
        top = side * H[a]
        inner = top - side * op["depth"]
        lo, hi = sorted((inner, top + side * big))
        return _slab_along(
            op["axis"], lo, hi, big, widths={c: (op["position"], 0.5 * op["width"])}
        )
    if kind == "fuse":
        a = AXES[op["axis"]]
        side = op["side"]
        start = side * (H[a] - op["overlap"])
        end = side * (H[a] + op["length"])
        size = op["size"]
        if op["primitive"] == "cylinder":
            p = [0.0, 0.0, 0.0]
            p[a] = min(start, end)
            return Cylinder(Pnt(*p), _DIRS[a], r=size, h=abs(end - start))
        if op["primitive"] == "box":
            lo, hi = sorted((start, end))
            return _slab_along(op["axis"], lo, hi, size)
        if op["primitive"] == "sphere":
            p = [0.0, 0.0, 0.0]
            p[a] = side * H[a]
            return Sphere(Pnt(*p), r=size)
        raise ValueError(f"unknown fuse primitive {op['primitive']!r}")
    raise ValueError(f"unknown op {kind!r}")


def build_solid(spec, stage="final"):
    """Build the conducting solid for ``spec``.

    ``stage="additive"`` returns the solid after fuse ops only (the reference
    volume for the volume-ratio gate); ``stage="final"`` applies all cuts too.
    """
    family, params = spec["family"], spec["params"]
    H = half_extents(family, params)
    solid = base_solid(family, params)
    ops = [o for o in spec["ops"] if o["kind"] != "aspect"]
    for op in (o for o in ops if o["kind"] == "fuse"):
        solid = solid + op_solid(op, H)
    if stage == "additive":
        return solid
    for op in (o for o in ops if o["kind"] != "fuse"):
        solid = solid - op_solid(op, H)
    return solid


def boundary_layer_thicknesses(spec):
    """Prism layer thicknesses in object units.

    First layer = skin depth at the max frequency, delta/alpha = sqrt(2/(mur*nu_max)),
    capped at ``bl_cap * maxh`` (when delta exceeds the element size the tets resolve
    it and a thick prism layer only risks invalid elements). Layers grow by 2x, as in
    the MPT-Calculator examples.
    """
    omega = spec["max_frequency"]
    delta = math.sqrt(2.0 / (omega * MU0 * spec["mur"] * spec["sigma"]))
    tau = min(delta / spec["alpha"], spec["bl_cap"] * spec["maxh"])
    return [tau * 2**n for n in range(spec["bl_layers"])]


def mesh_spec(spec, vol_path=None):
    """Glue the solid into the air box, mesh with prism layers, optionally save .vol.

    Returns ``(nmesh, info)`` where info holds element counts by type. The air
    region gets material 'air' and bc 'outer'; any domain netgen reports as
    'default' is renamed to 'air' (netgen >= 6.2.2301 quirk noted in the
    MPT-Calculator OCC examples).
    """
    mat = spec["material_name"]
    solid = build_solid(spec)
    solid.bc("default")
    solid.mat(mat)
    solid.maxh = spec["maxh"]
    L = spec["air_half"]
    air = _box((-L, -L, -L), (L, L, L))
    air.mat("air")
    air.bc("outer")
    air.maxh = spec["maxh_air"]
    air = air - solid
    geo = OCCGeometry(Glue([solid, air]))
    layers = []
    if spec["bl_layers"] > 0:
        layers = [
            BoundaryLayerParameters(
                boundary=".*",
                thickness=boundary_layer_thicknesses(spec),
                new_material=mat,
                domain=mat,
                outside=False,
                disable_curving=False,
            )
        ]
    nmesh = geo.GenerateMesh(boundary_layers=layers) if layers else geo.GenerateMesh()
    for i in range(1, nmesh.GetNDomains() + 1):
        if nmesh.GetMaterial(i) == "default":
            nmesh.SetMaterial(i, "air")
    n_prism = n_tet = n_other = 0
    for el in nmesh.Elements3D():
        nv = len(el.vertices)
        if nv == 4:
            n_tet += 1
        elif nv == 6:
            n_prism += 1
        else:
            n_other += 1
    info = {"n_elements": nmesh.ne, "n_tets": n_tet, "n_prisms": n_prism, "n_other": n_other}
    if vol_path is not None:
        nmesh.Save(str(vol_path))
    return nmesh, info
