# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A temporal video effects toolkit. Standalone Python scripts that process video frames over time to create visual effects (motion trails, slit-scan, optical flow visualization, etc) and retime footage via optical-flow frame interpolation.

## Scripts

- **`temporal_fx.py`** — Main effect engine. 31 effects applied via `-e <name>`. Each effect is a `fx_*` function that receives frames + params and returns processed frames. The `EFFECTS` dict maps CLI names to functions. Use `-e all` to run every effect. Most effects support `--cores N` for shared-memory multicore processing. The `echo` effect supports `--anchor` (window position relative to current frame) and `--step` (frame stride); `gaussian` accepts `--sigma` to control the bell curve spread.
- **`retime.py`** — Optical-flow slow motion / speed-up by frame interpolation. Computes dense flow between frame pairs in both directions, warps and blends. Backends via `--flow`: `dis` (default, fast CPU), `farneback`, `raft` (AI model, needs PyTorch). `--factor` >1 slows, <1 speeds up; `--accel-frames`/`--decel-frames` ease in/out from freezes.
- **`batch_random.py`** — Generates 50 randomized effect jobs from source videos. Runs 4 in parallel (RAFT jobs run sequentially due to GPU memory). Calls `temporal_fx.py` as a subprocess.

## Running

```bash
# Single effect
python3 temporal_fx.py video.mp4 -e echo -n 60

# Multicore echo/gaussian blending
python3 temporal_fx.py video.mp4 -e echo -n 30 --cores 4
python3 temporal_fx.py video.mp4 -e gaussian -n 30 --cores 4 --sigma 5.0

# Memory modes (auto picks based on available RAM)
python3 temporal_fx.py video.mp4 -e echo -n 60 --memory auto      # default
python3 temporal_fx.py video.mp4 -e echo -n 60 --memory streaming  # force low-memory
python3 temporal_fx.py video.mp4 -e echo -n 60 --memory ram        # force all-in-RAM

# Animated parameters (kcurve keyframes or math expressions)
python3 temporal_fx.py video.mp4 -e echo -n "10@0L:60@200L"
python3 temporal_fx.py video.mp4 -e echo -n "sin(f*0.05)*30+30"
python3 temporal_fx.py video.mp4 -e decay --decay "0.8@0L:0.99@200S"

# Preview animation curves
python -m kcurve "10@0L:60@200L"

# Slow motion via optical-flow interpolation (50% speed)
python3 retime.py video.mp4 --factor 2

# Batch 50 random effects from source/ directory
python3 batch_random.py
```

No build step, no tests, no linter configured. These are standalone scripts run directly.

## Dependencies

- Python 3.10+, OpenCV (`opencv-python`), NumPy
- `kcurve` package (animated parameter curves; pip-installed, not part of this repo)
- FFmpeg on PATH (H.264 re-encode + audio mux)
- PyTorch + torchvision >= 0.22.0 (only for `flow-raft` and `retime.py --flow raft`, lazy-imported)

## Architecture Patterns

**Pipeline**: Two modes controlled by `--memory auto|ram|streaming`. **RAM mode** (original): VideoCapture → load all frames into memory → apply effect → write mp4v → FFmpeg re-encode. **Streaming mode**: VideoCapture with lazy `FrameBuffer` → process frame-by-frame with sliding window → write each frame to VideoWriter immediately → FFmpeg re-encode. Streaming uses O(window) memory instead of O(total). **Auto mode** (default): probes video dimensions and frame count, estimates RAM needed (`estimate_memory_bytes()`), compares to available RAM (`get_available_memory()` via `sysctl`/`vm_stat` on macOS, `/proc/meminfo` on Linux), and picks the mode. Streaming does not support `--reverse` (falls back to RAM) or `--cores` (single-threaded). The FFmpeg step is try/excepted so scripts work without it.

**Adding a new effect to temporal_fx.py**: Write a `fx_name(frames, n, ...)` function, add it to the `EFFECTS` dict, add a default N value to `DEFAULT_N` if the effect uses a window size, and add a corresponding block in `_stream_frame()` for streaming mode support. CLI args are passed through via the `args` namespace.

**Shared code is duplicated** across scripts (progress_bar, FFmpeg mux pattern) rather than imported from a shared module. Each script is self-contained.

**CLAHE equalization** appears in two forms: `--pre-eq` (equalizes input frames before processing) and `--post-eq` (equalizes output frames after processing). Both are parallelized when `--cores` > 1. Several effects have built-in EQ variants (e.g. `brightest-eq`, `darkest-eq`, `bitwise-nor-eq`).

**Animated parameters** via the `kcurve` package. Any numeric CLI parameter (`-n`, `--decay`, `--step`, `--sigma`, `--anchor`, `--pre-eq`, `--post-eq`, `--gamma`, `--sharpen`, `--sharpen-radius`, `--orig-mix`, `--edge-preserve`, `--edge-thickness`, `--edge-gamma`) accepts either a static number or a kcurve spec. Two formats: keyframe strings (`value@frame[L|S|B]` colon-separated, L=linear S=spline B=bezier) and math expressions (`f` = frame number, with sin/cos/noise/fbm/lerp/etc). Parameters are pre-computed to per-frame numpy arrays in `run_effect()`, then resolved per-frame via `_at(param, i)` in effect loops. Parallel workers resolve via `_process_one()`.

## Directories

- `examples/` — Short H.265 result clips, one per technique, with a README mapping each to the command that produced it (committed)
- `source/` — Input videos (not committed)
- `output/` — Generated results (not committed)
