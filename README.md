# Temporal Video Effects Suite

31 temporal effects for video, processing frames over time to create unique visual results.

## Requirements

- Python 3.10+
- OpenCV (`pip install opencv-python`)
- NumPy
- FFmpeg (on PATH, for H.264 re-encoding and audio muxing)
- PyTorch + torchvision >= 0.22.0 (only needed for `flow-raft`)

## Usage

```
python3 temporal_fx.py <input_video> -e <effect> [options]
```

### Options

| Flag | Description |
|---|---|
| `-e, --effect` | Effect name or `all` (required) |
| `-n, --frames` | Temporal window size (overrides effect default) |
| `-o, --output` | Output file path (auto-generated if omitted) |
| `--decay` | Decay factor for `decay` effect (default: 0.92) |
| `--step` | Step size for `strobe` effect (default: 4) |
| `-q, --quality` | Quality preset for `flow-farneback`: `low`, `medium`, `high` (default: low) |
| `--pre-eq` | Apply CLAHE histogram equalization to input frames before processing |

## Effects

### Temporal blending

| Effect | Default n | Description |
|---|---|---|
| `echo` | 30 | Blend previous N frames with equal weight (motion trails) |
| `gaussian` | 30 | Gaussian-weighted blend across N frames (bell curve falloff from center) |
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

## Pre-processing

The `--pre-eq` flag applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to all input frames before any effect is applied. This expands the dynamic range of the source material, which can reveal detail in dark or overexposed footage. Works with any effect.

## Examples

```bash
# Motion trails with 60-frame window
python3 temporal_fx.py video.mp4 -e echo -n 60

# Gaussian-weighted blend (smoother than echo)
python3 temporal_fx.py video.mp4 -e gaussian -n 40

# Remove moving objects (median)
python3 temporal_fx.py video.mp4 -e median -n 30

# Slow exponential persistence
python3 temporal_fx.py video.mp4 -e decay --decay 0.95

# Brightest with pre-equalized input
python3 temporal_fx.py video.mp4 -e brightest -n 30 --pre-eq

# Brightest with expanded dynamic range
python3 temporal_fx.py video.mp4 -e brightest-eq -n 60

# Rainbow motion trails
python3 temporal_fx.py video.mp4 -e hue-trails -n 30

# Bitwise XOR (highlights per-bit differences)
python3 temporal_fx.py video.mp4 -e bitwise-xor

# Video feedback loop
python3 temporal_fx.py video.mp4 -e feedback

# Dense optical flow comparison
python3 temporal_fx.py video.mp4 -e flow-dis
python3 temporal_fx.py video.mp4 -e flow-farneback -q high
python3 temporal_fx.py video.mp4 -e flow-raft

# Run all effects
python3 temporal_fx.py video.mp4 -e all
```

## Output

Output files are saved alongside the input with the effect name appended, e.g.:

```
video.mp4 → video_echo.mp4, video_flow-dis.mp4, ...
```

Videos are re-encoded with H.264 (CRF 18) and audio is copied from the source when FFmpeg is available.
