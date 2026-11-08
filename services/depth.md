# RoboDojo precision depth

One entry: `scripts/precision_depth.py`. Five fixed lanes: RGB, GT, DAv2, TradSL, NSL.
Dry-run by default; `--execute` runs native RoboDojo evaluation. Native task budgets unchanged.

```bash
source config/env.sh
export PATH=/path/to/miniconda3/bin:$PATH
python scripts/precision_depth.py \
  --robo "$ROBODOJO_ROOT" --depth-root /path/to/Depth --data-root /path/to/data \
  --policy-dir "$ROBODOJO_ROOT/XPolicyLab/policy/HexaShell" \
  --gpus 1 2 4 5 6 --seed 1 --output /path/to/new-run --execute
```

`POLICY_ENV`, `EVAL_ENV`, SAM3 and gateway settings must be configured first. The native
policy alias must resolve to this checkout's `xpolicylab/HexaShell`. `--tasks insert_tubes`
selects a subset; omit it for the native eight precision tasks. `--eval-num` defaults to1.
Results/exit records are under output; native RoboDojo scoring/video remains authoritative.

Packaged sensor YAML in `config/precision-depth` targets dual-X5/arx_x5. GT uses optical-axis
distance_to_image_plane. TradSL/NSL use rendered passive RGB stereo (+5cm local-x), not active
IR. Metadata hashes, upstream commits and model hashes in `sources.json` are checked before
execution. `--env-configs` accepts five explicit alternatives; review metadata changes first.

Expected inputs: Depth-Anything-V2, TradSL and nsl_infer_server repositories under depth-root;
model interpreters under data-root/envs/{dav2,tradsl,nsl-infer}/bin/python and checkpoints under
data-root/models. See sources.json for source revisions, hashes and license caveats.

Depth geometry/perception reuses HexaShell's Camera, RGB View masks, DepthView/DepthScene and
spot. RGB uses original calibration. Depth reads observed OpenGL-to-world K/T and owns paired
RGB/depth snapshots. Stale Masks must be found again; to_world uses last delivered pictures.

Only trusted operator-owned loopback pickle endpoints19122–19124 and native NSL19130 are used.
GT is excluded from prediction requests. Wrong backend/token/shape, failed inference and
occupied ports fail closed; no GT fallback or custom depth scaling. Prediction HTTP timeout
110s is below native RPC120s; NSL cold startup45s/inference15s per camera can fail on slow hosts.

Validation: five real insert_tubes scripted episodes, three actions each, camera/look/
to_world/measure and model predictions completed without policy/cell errors. This is an
integration check, not task-success or depth-accuracy acceptance. DAv2 shows large metric
bias on this rendered scene (also present before refactoring); outputs are left unchanged.
