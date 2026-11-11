# NX convergence campaign for element-WENO M1PN

Independent study branch; **nothing is submitted automatically**. Existing solver code, historic NX campaigns, legacy study plans and numerical results remain untouched. Primary files:

- `scripts/element_weno/nx_convergence.py` — freezes binaries and analysis, generates a reproducible manifest, executes and validates runs, submits NUMA-pinned NX packs, collects observations.
- `scripts/element_weno/nx_plot_convergence.py` — derives figures (PNG/PDF) and a LaTeX table.
- `nx/batches/element_weno_pack.sbatch` — **reused**, not modified. Four 16-core slots on one NX node via `scripts/element_weno/pack_node.py`.
- `nx/scripts/analysis.py` — frozen and reused for moment conventions, kinetic crossing reference, exact-coordinate matching and structured-grid interpolation.
- `scripts/line_source/reference.py` — frozen and reused for the Ganapol/Bennett–McClarren line-source reference.

## 1. Prerequisites and usage

Run from the repository root on the NX login host after building both programs with the existing MFEM/MPI/Eigen environment:

~~~bash
cmake --build tmp/build --target m1pn_fast m1pn_element_weno \
  m1pn_element_weno_tests m1pn_element_weno_fix_tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ctest --test-dir tmp/build -R 'element_weno' --output-on-failure

# Check exact task inventory without creating any files:
python3 scripts/element_weno/nx_convergence.py prepare \
  --profile paper --campaign "$PWD/nx/tmp/ew_convergence_paper" --dry-run

# A small, separate shake-down campaign:
python3 scripts/element_weno/nx_convergence.py prepare \
  --profile smoke --campaign "$PWD/nx/tmp/ew_convergence_smoke"
python3 scripts/element_weno/nx_convergence.py submit \
  --campaign "$PWD/nx/tmp/ew_convergence_smoke"   # DRY RUN
python3 scripts/element_weno/nx_convergence.py submit \
  --campaign "$PWD/nx/tmp/ew_convergence_smoke" --execute

# After confirming smoke logs and results, prepare the full paper matrix:
python3 scripts/element_weno/nx_convergence.py prepare \
  --profile paper --campaign "$PWD/nx/tmp/ew_convergence_paper"
python3 scripts/element_weno/nx_convergence.py submit \
  --campaign "$PWD/nx/tmp/ew_convergence_paper"
python3 scripts/element_weno/nx_convergence.py submit \
  --campaign "$PWD/nx/tmp/ew_convergence_paper" --execute
~~~

**Review `--dry-run` before using `--execute`.** Only `submit --execute` invokes `sbatch`; no automatic background jobs are created by code generation or repository checkout.

The current pack script requests `--constraint=nx`, `--partition=med`, one exclusive 64-core node, 256 GiB and eight hours. It uses four NUMA-aware 16-core slots, two per socket. `--tasks-per-job 4` by default. To avoid flooding NX with whole-node reservations, the submitter defaults to **one active pack job at a time** using `afterany` dependency chains; `--max-concurrent-packs 2` explicitly opts into two concurrent node-sized packs. Source code and executables are copied into each immutable campaign folder and SHA256-checked before every task; commit and exact command arguments are recorded.

To inspect a single run or reprocess results without cluster submission:

~~~bash
python3 nx/tmp/ew_convergence_smoke/study.py run \
  --campaign "$PWD/nx/tmp/ew_convergence_smoke" \
  --task smooth_P3_p1_E8_spatial --threads 1
python3 nx/tmp/ew_convergence_smoke/study.py collect \
  --campaign "$PWD/nx/tmp/ew_convergence_smoke"
~~~

`collect` reports missing jobs as incomplete and returns nonzero; it never creates synthetic success records. The script refuses to overwrite existing campaign folders or solver output directories. Failed jobs require deliberate inspection and a new clean campaign or explicit manual recovery. The submission record prevents a duplicate blind submission.

## 2. Scientific matrix

The full profile contains **81 independent tasks** (21 in smoke mode), organised into:

| Family | Variable | Reference / caveat |
|---|---|---|
| Smooth periodic 1D | Qp p=1,2,3,4; E=8,16,32,64 | Standalone `m1pn_element_weno` with L2-projected initial field and **analytic PN density** |
| Smooth dt check | finest E=64, half CFL and half requested dt | Check temporal contamination before claiming spatial Qp order |
| Open-boundary smooth transport | P3,P7; Q2; E=8,16,32,64 | Analytic kinetic inflow; finite PN modeling error persists |
| Crossing Beams | P3,P7; Q1,Q2,Q3; E=8,16,32,64 | Kinetic density **and** fixed-PN next-grid self difference reported separately |
| GHS Line Source | P3,P7; Q1,Q2; E=32,64,128 | Exact infinite-medium reference with documented rescaling: sigma_s=3, T=1/3, Gaussian width 0.01, background 9e-4 |
| Temporal splitting | Smooth periodic with sigma_s=2,sigma_a=0.5; crossing with sigma_s=2,sigma_a=0.5, filter 0.15 | Fixed grid/PN, dt through dt/16; finest solution is **self reference** |
| Ablation | WENO on/off and 1e-3 vs 0.2 weights; limiter on/off; SSPRK3 vs SSPRK(4,3) | No-limiter failures are valid findings, not silently excluded |
| Baseline | `m1pn_mass_correction` coupled vs `pn-only`, SSPRK3 | **Q1 / 128 elements vs element-WENO Q2 / 64 elements:** same 129² sample points, different spatial scheme; no unimplemented pure-M1 claim |

Production studies use SSPRK(4,3) **explicitly**, `--angular-reduction off` and `--mirror-reduction off` for transparent comparability (optimized variants can be compared in a separate performance campaign). Typical `--cfl .2` and the solver's current feasibility/stability step bound are used; time refinement makes a separate zero-time probe to determine a consistent sequence.

### Interpretation restrictions

1. Periodic smooth exact-PN study is the main test of formal spatial order; it uses `m1pn_element_weno` with L2 initialization, not point interpolation.
2. Kinetic errors in Crossing, Line Source and boundary cases contain **PN truncation error**. Their plateau does not by itself disprove high spatial order.
3. Temporal refinements use a numerical finest-dt reference on an identical grid. Slopes nearest to the finite reference are biased and should not be presented as proof of asymptotic order.
4. Strang splitting of noncommuting transport and reaction is generally second order; this study **measures** its impact, not a theorem about all problems.
5. At high polynomial degrees, SSPRK(4,3) temporal error may dominate the smooth spatial convergence. Use paired time-step tests before claiming EOC.
6. C3 synchronization energy and C7 Onsager boundary consistency remain unresolved. Cone preservation does not imply dissipativity of all PN harmonics.
7. `m1pn_fast` baseline uses another spatial scheme, and the repository has no verified pure-M1 equivalent under identical conditions.

## 3. Frozen outputs

Each campaign contains:

- `manifest.json`, `tasks.json`: exact experiment design, commit, source and binary hashes.
- `packs/tasks_*.tsv`, `packs/*.log`: deterministic SLURM commands and logs.
- `runs/<task>/config.json`, `summary.json`, `initial.csv`, `final.csv`, `nodal_mass.csv` where supported.
- `status/<task>.json`: success/failure and SHA256 of saved outputs.
- `references/ganapol_GH.csv`: independently generated kinetic radial reference.
- `results.csv`, `results.json`: classified error metrics and computed EOCs.
- `REPORT.md`: completion and methodology notes; missing tasks marked explicitly.
- `figures/*.pdf`, `figures/*.png`, `tables.tex`: publication figures and tables, subject to verifying the classification.

Only runs whose saved checksums, metadata, Bernstein-coefficient cone margin and conical sampled moments validate are admitted to the aggregate tables. Validation is a numerical check, not a proof of exact real-arithmetic cone preservation.

## 4. Testing and extension

~~~bash
python3 -m py_compile scripts/element_weno/nx_convergence.py \
  scripts/element_weno/nx_plot_convergence.py tests/test_nx_convergence.py
python3 -m unittest discover -s tests -p 'test_nx_convergence.py' -v
~~~

The scripts intentionally do **not** modify production solver code, change default integrators, add a new third-order reaction splitting, or launch automatically. For any publication figure, retain the frozen campaign and its git SHA.

## 5. Heterogeneous and 3D profiles (format nx_convergence_v2)

`--profile hetero-smoke` (5 tasks) and `--profile hetero-paper` (63 tasks) add the lattice (Paul (44)-(47)), the
3D shadow (Weih et al. 4.2.1 geometry, own beam profile) and the own 3D lattice extension; see
`docs/ELEMENT_HETEROGENEOUS.md`. The v1 profiles `smoke`/`paper` and their task matrices are unchanged.

~~~bash
python3 scripts/element_weno/nx_convergence.py plan --profile hetero-paper      # resource estimate, no files
python3 scripts/element_weno/nx_convergence.py prepare --profile hetero-paper \
  --campaign "$PWD/nx/tmp/ew_hetero_paper"
python3 nx/tmp/ew_hetero_paper/study.py submit --campaign "$PWD/nx/tmp/ew_hetero_paper"            # DRY RUN
python3 nx/tmp/ew_hetero_paper/study.py submit --campaign "$PWD/nx/tmp/ew_hetero_paper" --execute
# smoke on the short partition:
python3 <campaign>/study.py submit --campaign <campaign> --partition short --time 00:30:00 --execute
~~~

Validation per run: material integrals against the expected values (lattice 49/1/110/38, 3D lattice
343/1/620/429, shadow sphere within 5e-3), expected mirror directions, particle balance (|relative| <= 1e-9),
numerical cone margin. Recorded metadata: affinity, NUMA nodes, OMP variables, compiler, moments, dofs, dt, steps,
RSS, phase times (incl. reaction), ledger. `submit` packs tasks by slot class (`slot_cores` 16 or 32 from the
cost model); `--partition`/`--time` override the pack defaults. `collect` writes `results.csv/json` and `REPORT.md`
with lattice deviations from Schotthoefer/Hauck (Table 1 / 2 pi) and shadow Beer-Lambert metrics from `slices.csv`.
Historical path-B lattice runs used an older binary; `lattice_B_*_B_new` recomputes them with the current one.
