# Temporal Video Effects Suite

Tools for processing video frames over time.

- **`temporal_fx.py`** — the effect engine. Applies one of 31 temporal effects (motion trails, slit-scan, bitwise blends, optical-flow visualization, …) to a video. Basic use: `python3 temporal_fx.py video.mp4 -e echo`. Almost every numeric option can be animated over time with keyframes or math expressions (see [Animated parameters](#animated-parameters-kcurve)).
- **`retime.py`** — optical-flow retimer. Slows down (or speeds up) a video by synthesizing in-between frames with dense optical flow, instead of duplicating or dropping frames. Basic use: `python3 retime.py video.mp4 --factor 2` for 50% speed.
- **`batch_random.py`** — batch driver that generates 50 randomized effect jobs from videos in `source/` and runs them in parallel through `temporal_fx.py`.

The two main tools compose well: slow footage down with `retime.py`, run a temporal effect over the extra frames, then speed it back up (`--factor 0.5` and below speed up).

## Requirements

- Python 3.10+
- OpenCV (`pip install opencv-python`)
- NumPy
- `kcurve` package (animated parameter curves — install it, or place a `kcurve/` checkout next to the scripts)
- FFmpeg (on PATH, for H.264 re-encoding and audio muxing; scripts still work without it, skipping that step)
- PyTorch + torchvision >= 0.22.0 (only needed for the RAFT flow backend / `flow-raft` effect)

---

# temporal_fx.py

```
python3 temporal_fx.py <input_video> -e <effect> [options]
```

## Options

### Core

| Flag | Description |
|---|---|
| `-e, --effect` | Effect name or `all` to run every effect (required) |
| `-n, --frames` | Temporal window size in frames (overrides the effect's default). Animatable. |
| `-o, --output` | Output file path (auto-generated next to the input if omitted) |
| `--reverse` | Blend with *following* frames instead of previous (reverses input, processes, reverses output) |
| `--hflip` | Mirror frames horizontally before processing |
| `--no-audio` | Skip muxing audio from the source |

### Effect-specific

| Flag | Description |
|---|---|
| `--decay` | Decay factor for the `decay` effect (default: 0.92). Animatable. |
| `--step` | Frame stride for `strobe` (default 4) and `echo` (default 1). For echo, `-n` is the window *span* and `n/step` frames are blended. Animatable. |
| `--sigma` | Gaussian sigma for the `gaussian` effect (default: `n/4`). Smaller = sharper peak, larger = flatter. Animatable. |
| `--anchor` | Window anchor for the `echo` effect: `1.0` = current frame is the *end* of the window (trails behind, the default), `0.0` = current frame is the *start* (trails ahead), values between shift the window proportionally. Animatable. |
| `-q, --quality` | Quality preset for `flow-farneback`: `low`, `medium`, `high` (default: low) |

### Pre/post-processing

These work with any effect and are applied around the effect itself.

| Flag | Description |
|---|---|
| `--pre-eq CLIP` | CLAHE histogram equalization on input frames *before* processing (clip limit, e.g. `2.0`). Expands dynamic range of the source. Animatable. |
| `--post-eq CLIP` | CLAHE equalization *after* processing, to restore contrast lost to heavy blending (e.g. `2.0`; higher = stronger). Animatable. |
| `--edge-preserve S` | Re-inject Sobel edges from the source frames into the processed output; strength 0.0–1.0 controls the blend. Animatable. |
| `--edge-thickness T` | Edge line thickness for `--edge-preserve` (default: 3). Animatable. |
| `--edge-gamma G` | Gamma applied per frame after the edge-preserve pass (only active with `--edge-preserve`). Animatable. |
| `--gamma G` | Gamma correction after processing (>1 brightens midtones, <1 darkens). Applied after post-eq. Animatable. |
| `--sharpen AMOUNT` | Unsharp-mask sharpen after processing (~0.5–1.5 typical). Animatable. |
| `--sharpen-radius PX` | Gaussian blur radius for `--sharpen` (default: 3.0). Animatable. |
| `--orig-mix FRAC` | Blend a fraction of the original frame back over the final result (0.0–1.0). Applied last, on top of everything. Animatable. |

### Performance

| Flag | Description |
|---|---|
| `--cores N` | CPU cores for shared-memory parallel processing (default: 4). Most effects are parallelized, as are `--pre-eq`/`--post-eq`. Not parallelized: `decay`, `feedback` (sequential accumulation), the optical-flow effects, and the trivially fast consecutive-frame ops (`diff`, `bitwise-xor`). |
| `--memory MODE` | `auto` (default), `ram`, or `streaming`. See below. |

#### Memory modes

- **`ram`** — load every frame into memory, process, write. Fastest, but needs O(total frames) RAM.
- **`streaming`** — lazy frame buffer with a sliding window; each output frame is written immediately. Uses O(window) memory, so arbitrarily long videos work. Does not support `--reverse` (falls back to RAM) or `--cores` (single-threaded).
- **`auto`** — probes the video, estimates peak RAM needed, compares it to available system memory, and picks `ram` or `streaming` accordingly.

## Animated parameters (kcurve)

Every flag marked "Animatable" above (`-n`, `--decay`, `--step`, `--sigma`, `--anchor`, `--pre-eq`, `--post-eq`, `--edge-preserve`, `--edge-thickness`, `--edge-gamma`, `--gamma`, `--sharpen`, `--sharpen-radius`, `--orig-mix`) accepts either a plain number or a **kcurve spec** — a string that evaluates to a different value on every frame. Values are pre-computed into a per-frame array before processing, so animation works in both single-core and multicore modes.

### Keyframe strings

Format: `value@frame[interp]`, colon-separated. The interpolation letter controls how the curve moves *toward the next keyframe*:

- `L` — linear
- `S` — spline (Catmull-Rom, smooth through the points)
- `B` — bezier (ease in/out)

```bash
# Echo window grows linearly from 10 frames at frame 0 to 60 at frame 200
python3 temporal_fx.py video.mp4 -e echo -n "10@0L:60@200L"

# Decay factor eases smoothly from 0.8 to 0.99
python3 temporal_fx.py video.mp4 -e decay --decay "0.8@0L:0.99@200S"

# Echo trails swing from behind the subject (anchor 1) to ahead of it (anchor 0)
python3 temporal_fx.py video.mp4 -e echo -n 30 --anchor "1@0L:0@300L"

# Pre-eq pulses: flat, spike to 4.0 at frame 290, snap back, spike again at 700
python3 temporal_fx.py video.mp4 -e echo \
  --pre-eq "1@1L:4.0@290S:1.0@300S:1@600S:5@700L"
```

Multiple animated parameters can be combined in one run, each with its own set of keyframes.

### Math expressions

Any expression using `f` as the frame number, with `sin`, `cos`, `lerp`, `clamp`, `smoothstep`, `noise`, `fbm`, `pi`, and friends:

```bash
# Echo window oscillates between 0 and 60 frames
python3 temporal_fx.py video.mp4 -e echo -n "sin(f*0.05)*30+30"

# Organic wandering sigma via fractal noise
python3 temporal_fx.py video.mp4 -e gaussian --sigma "fbm(f*0.01)*8+2"
```

### Previewing curves

Plot a curve before committing to a render:

```bash
python -m kcurve "10@0L:60@100S:10@200L"     # keyframe specs
python -m kcurve "sin(f*0.1)*30+30" 0 200    # expressions need start/end frames
```

## Effects

### Temporal blending

| Effect | Default n | Description |
|---|---|---|
| `echo` | 30 | Blend previous N frames with equal weight (motion trails). Supports `--anchor`, `--step`, `--cores` |
| `gaussian` | 30 | Gaussian-weighted blend across N frames (bell curve falloff, see `--sigma`) |
| `median` | 15 | Median pixel across N frames (removes moving objects) |
| `decay` | — | Exponential persistence (use `--decay` to control) |
| `time-ramp` | 60 | Blend window grows from 1 to N over the clip |
| `strobe` | 30 | Blend every Kth frame across wider span (use `--step`) |
| `ping-pong` | 30 | Average forward + time-reversed frames |
| `brightest` | 45 | Keep brightest pixel across N frames (light trails) |
| `darkest` | 45 | Keep darkest pixel across N frames |
| `brightest-eq` | 45 | Brightest pixel + CLAHE histogram equalization |
| `darkest-eq` | 45 | Darkest pixel + CLAHE histogram equalization |
| `brightest-edge` | 45 | Brightest pixel + CLAHE + Canny edge overlay |
| `darkest-edge` | 45 | Darkest pixel + CLAHE + Canny edge overlay |
| `screen` | 30 | Screen blend across N frames (double-exposure, combines light) |
| `multiply` | 30 | Multiply blend across N frames (shadow combine, normalized) |
| `hue-trails` | 30 | Echo with progressive hue shift (rainbow motion trails) |

### Scanline / time-displacement

| Effect | Default n | Description |
|---|---|---|
| `slit-scan` | 120 | Each row of pixels comes from a different frame in time |
| `rolling-shutter` | 30 | Each scanline offset by 1 frame (rolling shutter simulation) |
| `time-mosaic` | 60 | 8x8 grid of tiles, each from a different moment in the N-frame window |

### Bitwise operations

| Effect | Default n | Description |
|---|---|---|
| `bitwise-or` | 15 | Bitwise OR across N frames (accumulates lit pixels) |
| `bitwise-and` | 15 | Bitwise AND across N frames (keeps persistent pixels) |
| `bitwise-nor` | 15 | Bitwise NOR across N frames (inverse OR, keeps unlit pixels) |
| `bitwise-nor-eq` | 15 | Bitwise NOR + CLAHE histogram equalization |
| `bitwise-xor` | — | Bitwise XOR between consecutive frames (highlights per-bit changes) |

### Feedback / motion

| Effect | Default n | Description |
|---|---|---|
| `feedback` | — | Recursive blend with slight zoom (video feedback loop) |
| `motion-streak` | — | Directional blur along optical flow vectors (DIS-based) |

### Frame analysis

| Effect | Default n | Description |
|---|---|---|
| `diff` | — | Absolute difference between consecutive frames |
| `temporal-gradient` | 30 | Per-pixel temporal change mapped to TURBO colormap |
| `temporal-variance` | 30 | Per-pixel temporal std-dev mapped to TURBO colormap (motion heatmap) |

### Dense optical flow

All three flow effects visualize motion vectors as HSV color: hue encodes direction, brightness encodes speed.

| Effect | Default n | Description |
|---|---|---|
| `flow-dis` | — | DIS algorithm (fast, good quality) |
| `flow-farneback` | — | Farneback polynomial expansion (classic, use `-q` for quality) |
| `flow-raft` | — | RAFT neural network via torchvision (highest quality, slowest) |

#### Farneback quality presets (`-q`)

| Preset | pyr_scale | levels | winsize | iterations | poly_n | poly_sigma | flags |
|---|---|---|---|---|---|---|---|
| `low` | 0.5 | 3 | 15 | 3 | 5 | 1.2 | none |
| `medium` | 0.5 | 5 | 21 | 5 | 7 | 1.5 | Gaussian |
| `high` | 0.4 | 7 | 31 | 10 | 7 | 1.5 | Gaussian |

#### RAFT notes

- Requires `torchvision >= 0.22.0` (model weights download automatically on first run)
- Large frames are downscaled to 640px for inference to avoid GPU memory issues
- Uses MPS (Apple Silicon) when available, falls back to CPU on out-of-memory errors

## Examples

```bash
# Motion trails with 60-frame window
python3 temporal_fx.py video.mp4 -e echo -n 60

# Multicore echo, trails ahead of the subject
python3 temporal_fx.py video.mp4 -e echo -n 30 --anchor 0 --cores 4

# Gaussian-weighted blend with explicit sigma
python3 temporal_fx.py video.mp4 -e gaussian -n 30 --cores 4 --sigma 5.0

# Remove moving objects (median)
python3 temporal_fx.py video.mp4 -e median -n 30

# Heavy blend, then restore contrast and edges
python3 temporal_fx.py video.mp4 -e echo -n 60 --post-eq 2.0 --edge-preserve 0.5

# Force low-memory streaming for a long video
python3 temporal_fx.py long_video.mp4 -e echo -n 60 --memory streaming

# Dense optical flow comparison
python3 temporal_fx.py video.mp4 -e flow-dis
python3 temporal_fx.py video.mp4 -e flow-farneback -q high
python3 temporal_fx.py video.mp4 -e flow-raft

# Run all effects
python3 temporal_fx.py video.mp4 -e all
```

---

# retime.py

Slow motion (or speed-up) by frame interpolation: dense optical flow is computed between each pair of frames in both directions, each frame is warped along the scaled opposing flow, and the two warps are blended so each direction fills the other's occlusion holes. Output plays at the original fps with more (or fewer) frames.

```
python3 retime.py <input_video> [options]
```

## Options

| Flag | Description |
|---|---|
| `-o, --output` | Output path (default: `<input>_slow.mp4`) |
| `--factor F` | Retime factor, may be fractional. `>1` slows down (`2` = 50% speed, one synthesized frame between each pair; `4` = 25% speed), `<1` speeds up (`0.5` = 200% speed). Default: 2 |
| `--flow BACKEND` | Dense flow backend: `dis` (default — fast, CPU, no extra deps), `farneback` (classic, CPU), `raft` (AI model, best quality, needs PyTorch) |
| `--accel-frames N` | Ease in: ramp up from a freeze over the first N *source* frames |
| `--decel-frames N` | Ease out: ramp down to a freeze over the last N *source* frames |
| `--max-dim PX` | Max inference dimension for RAFT (default: 640); larger frames are downscaled for flow and the flow field upscaled back |
| `--limit N` | Only process the first N input frames (quick tests) |
| `--hflip` | Horizontally flip frames before flow/interpolation |

Notes:

- With `--accel-frames`/`--decel-frames` the retime is variable-speed, so a later constant speed-up will not restore the original timing — that asymmetry is usable as an effect in itself.
- RAFT uses MPS on Apple Silicon when available and falls back to CPU on out-of-memory.

## Examples

```bash
# 50% slow motion (factor 2), DIS flow
python3 retime.py video.mp4

# 25% speed with the RAFT AI model
python3 retime.py video.mp4 --factor 4 --flow raft

# Slow down 2.66x, easing in and out of freezes over 90 frames
python3 retime.py video.mp4 --factor 2.66 --accel-frames 90 --decel-frames 90

# Speed back up by the same factor
python3 retime.py video_slow.mp4 --factor 0.376 -o video_restored.mp4

# Round-trip with a temporal effect in the middle:
python3 retime.py video.mp4 --factor 2.66 -o slow.mp4
python3 temporal_fx.py slow.mp4 -e echo -n 30 --anchor "1@0L:0@300L" -o slow_echo.mp4
python3 retime.py slow_echo.mp4 --factor 0.376 -o final.mp4
```

---

## Output

Output files are saved alongside the input with the effect name appended, e.g.:

```
video.mp4 → video_echo.mp4, video_flow-dis.mp4, ...
```

Videos are re-encoded with H.264 (CRF 18) and audio is copied from the source when FFmpeg is available.
