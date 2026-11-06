# mpt-geometry-augmenter

Generate parametric and irregular metal geometries for
[MPT-Calculator](https://github.com/MPT-Calculator/MPT-Calculator) (magnetic
polarizability tensors via NGSolve FEM), batch-run them, and collect the results
into one tidy table for ML.

- **Shape families:** sphere, cylinder (rod or disc), cuboid, ellipsoid (native OCC quadric).
- **Irregular ops** (seeded per sample): aspect-ratio perturbation, corner cut
  (planar chamfer-like Box cut), through-hole (Cylinder cut), notch/groove (Box
  cut), fuse of a second primitive (cylinder/box stem or spherical bump).
- **Validity gate** before any FEM: physical ranges, a minimum-feature check, a
  single connected solid after the booleans, volume ratio, bounding box. After
  meshing it checks an element cap and that the prism layer is non-empty.
- **Reproducible:** every sample is a JSON `SampleSpec`. The generated
  `OCC_Geometry/aug_<id>.py` embeds that JSON plus the builder source, so the file
  rebuilds the identical solid without this package installed.

## Why sample in (shape, μr, log ν)

The MPT obeys the similarity law

    M(α, ω, σ, μr, B) = α³ · M̂(ν, μr, B),    ν = ω μ0 σ α²

So for a fixed dimensionless shape B, only ν and μr matter. Sampling σ and α
independently wastes samples on equivalent points and leaves gaps in ν. Instead
we sample:

| dim | distribution |
|---|---|
| shape params | Sobol/LHS over aspect ratios (largest half-extent normalised to 1) |
| μr | point mass at 1 (p = 0.4, non-magnetic) + log-uniform on [2, 100] |
| log10 ν_max | uniform on `--nu-range` (default [2, 4]); ν_max = ν at ω_max |
| log10 σ | uniform on [1e6, 6e7] S/m. This is only for realistic metadata; it does not affect M̂ |

Then **α is derived**: α = sqrt(ν_max / (ω_max μ0 σ)). Each row records σ, α, ω
and ν. In object units the skin depth at ω_max is δ/α = sqrt(2/(μr ν_max)). This
sets the prism-layer thickness and the skin-depth term of the feature rule. Because
the sweep covers several decades of ω, each sample spans several decades of ν below
ν_max.

The continuous dimensions come from a scrambled Sobol sequence (or LHS). Families
are assigned round-robin, so they stay balanced. Discrete choices and op parameters
come from `numpy.random.default_rng(seed)`, with a per-draw seed derived from
`(--seed, draw index)`. When the gate rejects a draw, the generator takes the next
QMC point.

## Install

Use the same environment as MPT-Calculator:

```bash
pip install -r requirements.txt        # ngsolve/netgen 6.2.2608, numpy==2.3.3, scipy, pandas, pyarrow
pip install -e .                        # optional; or run from the repo root
```

- **numpy must stay at 2.3.3.** numpy 2.5 breaks the MPT-Calculator solver.
- **Netgen 6.2.2608 .vol files** make
  `MPT-Calculator/Functions/Helper_Functions/count_prismatic_elements.py` loop
  forever. The old loop waits for a `# surfid` comment that newer Netgen no
  longer writes. Patch it locally to read exactly the element count declared after
  `volumeelements`, counting lines with 4 or 6 nodes, and to stop at EOF. The
  augmenter itself never modifies the MPT-Calculator checkout. It only adds
  `OCC_Geometry/aug_*.py`, and MPT-Calculator writes its own `GeoFiles/`,
  `VolFiles/` and `Results/`.

## Usage

```bash
# Tune maxh / check acceptance: generate + gate + mesh only, no FEM
python -m mpt_aug.batch_generator --mpt-root ../ext/MPT-Calculator --n 20 --out dataset/dry --dry-run

# Real run
python -m mpt_aug.batch_generator --mpt-root ../ext/MPT-Calculator \
    --n 5 --out dataset/run1 --order 2 --pod-points 7 --freq 1 6 12 --seed 0 \
    [--families sphere cuboid] [--maxh 0.3] [--bl-layers auto|N] [--sampler sobol|lhs]
```

`--freq a b n` sweeps ω (rad/s) over `logspace(a, b, n)`. Each FEM run executes in
a subprocess with `--timeout` (default 900 s), so a solver crash or hang only fails
that sample. MPT-Calculator's console output goes to `logs/aug_<id>.log`.

The library can also be used directly:

```python
from mpt_aug.geometry_augmenter import spec_from_unit, validate_spec, write_sample_files
spec = spec_from_unit([0.4, 0.6, 0.7, 0.5, 0.5], "cuboid", seed=1)
print(validate_spec(spec).reason or "ok", spec.to_json())
```

Tests: `pytest -m "not slow"` takes about 2 s. `pytest -m slow` runs the
FEM-vs-exact sphere check and takes about 7 min. It needs `MPT_ROOT` or a checkout
at `../ext/MPT-Calculator`.

## Output layout (`--out`)

| file | content |
|---|---|
| `mpt_dataset.parquet` | long table, one row per (sample, ω). Written as CSV if pyarrow is missing |
| `samples.jsonl` | accepted specs + gate diagnostics + mesh counts + FEM status |
| `rejections.jsonl` | every gate rejection with its reason. Use it for tuning |
| `failures.jsonl` | FEM/harvest failures, and slots that ran out of attempts |
| `specs/aug_<id>.json` | spec sidecars |
| `logs/` | per-sample MPT-Calculator logs |
| `run_summary.json` | args, configs, acceptance rate, per-sample timing |

### Dataset schema

| column(s) | meaning |
|---|---|
| `sample_id, family, ops, n_ops` | identity. `ops` is a comma-joined op list |
| `mur, sigma, alpha, nu_max` | material/scale. `alpha` is metres per object unit |
| `omega, nu` | angular frequency (rad/s) and ν = ω μ0 σ α² |
| `maxh, bl_layers, n_elements, n_prisms, ndofs_0, ndofs_1, order, pod_points` | discretisation. `ndofs_*` is MPT-Calculator's `NDOF` tuple |
| `wall_time_s` | MPT-Calculator `main()` wall time |
| `M_{xx,yy,zz,xy,xz,yz}_{re,im}` | the 6 independent entries of M = (N0 + R) + iI, in m³ |
| `eig_R_{1,2,3}, eig_I_{1,2,3}` | ascending eigenvalues of R = Re(M) − N0 and of I = Im(M) |
| `N0_{xx,...,yz}` | the frequency-independent N0 |
| `inv_{1,2,3}_{re,im}` | invariants as written by MPT-Calculator (`Invariants.csv`) |

For a scale-free target, use M̂ = M/α³ against (log ν, μr). Objects are axis-aligned
in a canonical orientation. Rotate the tensors yourself if you want orientation
augmentation; eigenvalues and invariants are rotation-invariant.

## Validity gate: what is checked

The minimum feature size is t_min = max(2·maxh, 3·δ_min/α). It is estimated from
the spec parameters:

- base thickness
- hole ligament in the mid cross-section (exact for rectangular and elliptic sections)
- hole diameter
- notch floor at the notch position
- notch side ligaments
- slot width (an air gap, so only the maxh rule applies)
- fuse stem size

The sampler draws op parameters inside these limits, so the gate does not quietly
remove most holes and notches. Seed 0 gives about 90 % acceptance. The remaining
rejections are mostly Netgen meshing failures on near-tangent booleans.

After the booleans, the gate checks:

- exactly one solid
- volume within [0.3, 1.0] × the volume after fuse ops. Additive ops define the
  reference, so fused samples are not penalised
- bounding box inside 5 % of the air box (±1000 units)

After meshing, it checks the element cap (60k by default) and that prisms exist.

## Known limits

- **The feature rule is a heuristic.** It does not check interactions between ops
  (for example a corner cut passing near a hole), or the wedge edges where a hole
  leaves a curved surface. The single-solid, volume and element-cap checks catch
  the worst of these. The 3δ term is conservative: physically, thin walls only
  need the prism stacks from opposite faces not to collide. That condition is
  enforced separately (stack ≤ 0.9·maxh, and t_min ≥ 2·maxh). At low ν the 3δ
  term only biases the dataset towards chunkier shapes. Use `--delta-factor 0` to
  disable it.
- **The boundary-layer stack must cover mid-band frequencies.** With prism layers
  sized only for δ(ω_max) (for example `--bl-layers 2`), the band where δ is thicker
  than the stack but thinner than the tets is under-resolved. Measured max error
  against the exact sphere solution (μr = 10, ν_max = 1e3, order 2, 16k elements):
  - 2 layers, 7 POD points: 5.7 %
  - `auto` layers (stack spans δ(ω_max) to maxh/2), 7 POD points: 1.4 %. This error
    is POD interpolation in the top decade
  - `auto` layers, 13 POD points: 0.26 %
- **POD with 7 snapshots** is cheap but costs about 1 % accuracy between snapshots.
  Use ≥ 13 for production data.
- **The air box dominates the element count** (~11k of ~13–23k at maxh 0.3), so
  making the object maxh coarser barely reduces cost but rejects holes and notches.
  maxh adapts down to 0.15 for thin bodies.
- **Meshing runs twice per accepted sample:** once in-process for the gate, and
  once inside MPT-Calculator from the OCC file. This takes a few seconds per sample.
- **Single conducting material per object** for now; MPT-Calculator itself
  supports several.

## License

MIT, © 2026 Theerayut Boonkird.
