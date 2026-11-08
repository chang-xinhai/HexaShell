# Native RoboDojo depth entry

RGB remains the default. Set `HEXASHELL_DEPTH=true` to select the packaged RGBD session.
`HEXASHELL_DEPTH_BACKEND` is `gt` (default), `dav2`, `tradsl`, or `nsl`.
Predicted modes additionally require `HEXASHELL_DEPTH_URL=http://127.0.0.1:PORT`.
No setup directory, startup injection, global parser patch, or camera-axis conversion is used.

The policy boundary must already provide WXYZ robot poses and declare
`additional_info.camera_extrinsic_convention=opengl_to_world`. Camera-to-world matrices
are copied without conversion. All three cameras must carry RGB, K and T.
Depth is aligned float32 optical-axis metres, not ray length. Invalid samples are excluded;
missing depth fails closed. Predictions receive only RGB and calibration, never GT depth.
RGB screenshots remain RGB; motion and task success criteria are unchanged.

`services/robodojo_depth.py --help` documents the persistent local worker. It reuses the
existing Depth model implementations. The service uses trusted-loopback pickle: never
expose/forward it to untrusted clients. NSL requires its separate native inference server.
TradSL/NSL currently consume passive RGB stereo rendered at a 5cm local-x baseline,
converted to grayscale; this is NOT a simulation of projected active IR patterns.

The only experiment entry is `scripts/run_precision_depth.py` (default dry-run).
Actual evaluation requires explicit `--execute`, five distinct GPU IDs and a new output
folder. Each lane serializes the canonical precision tasks and uses native task budgets.
Do not use the old `setup/.../sitecustomize.py` launchers with this entry.

Offline tests do not establish simulator rendering equivalence or real task success.
An online acceptance run remains required before drawing benchmark conclusions.
