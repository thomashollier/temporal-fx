# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A temporal video effects toolkit. Three standalone Python scripts that process video frames over time to create visual effects (motion trails, slit-scan, optical flow visualization, etc).

## Scripts

- **`temporal_fx.py`** — Main effect engine. 31 effects applied via `-e <name>`. Each effect is a `fx_*` function that receives frames + params and returns processed frames. The `EFFECTS` dict maps CLI names to functions. Use `-e all` to run every effect.
- **`frame_blend.py`** — Weighted temporal frame averaging (uniform or gaussian). Uses `multiprocessing.shared_memory` to parallelize across cores. Has `--post-eq` for CLAHE contrast restoration after blending.
- **`batch_random.py`** — Generates 50 randomized effect jobs from source videos. Runs 4 in parallel (RAFT jobs run sequentially due to GPU memory). Calls `temporal_fx.py` as a subprocess.

## Running

```bash
# Single effect
python3 temporal_fx.py video.mp4 -e echo -n 60

# Frame blending (4-core parallel)
python3 frame_blend.py video.mp4 -n 30 --post-eq 2.0 --cores 4

# Batch 50 random effects from source/ directory
python3 batch_random.py
```

No build step, no tests, no linter configured. These are standalone scripts run directly.

## Dependencies

- Python 3.10+, OpenCV (`opencv-python`), NumPy
- FFmpeg on PATH (H.264 re-encode + audio mux)
- PyTorch + torchvision >= 0.22.0 (only for `flow-raft`, lazy-imported)

## Architecture Patterns

**Pipeline**: VideoCapture → load all frames into memory → apply effect → write mp4v → FFmpeg re-encode to H.264 (CRF 18) + copy audio from source. The FFmpeg step is try/excepted so scripts work without it.

**Adding a new effect to temporal_fx.py**: Write a `fx_name(frames, n, ...)` function, add it to the `EFFECTS` dict, and add a default N value to `DEFAULT_N` if the effect uses a window size. CLI args are passed through via the `args` namespace.

**Shared code is duplicated** across scripts (progress_bar, FFmpeg mux pattern) rather than imported from a shared module. Each script is self-contained.

**CLAHE equalization** appears in two forms: `--pre-eq` on temporal_fx.py (equalizes input frames before processing) and `--post-eq` on frame_blend.py (equalizes output frames after blending). Several effects have built-in EQ variants (e.g. `brightest-eq`, `darkest-eq`, `bitwise-nor-eq`).

## Directories

- `source/` — Input videos (not committed)
- `output/` — Generated results (not committed)
