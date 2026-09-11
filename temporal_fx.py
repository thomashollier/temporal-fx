#!/usr/bin/env python3
"""
Temporal Video Effects Suite — 31 temporal effects for video.
Each effect processes frames over time to create unique visual results.
"""

import cv2
import numpy as np
import argparse
import subprocess
import sys
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kcurve"))
from kcurve import curve as kcurve_parse


def _is_animated(param):
    """Check if a parameter is an animated (per-frame) array."""
    return isinstance(param, np.ndarray)


def _at(param, i):
    """Resolve a parameter for frame i. Works with scalars and per-frame arrays."""
    if isinstance(param, np.ndarray):
        return param[i]
    return param


def _precompute(param, total):
    """Convert a kcurve callable to a per-frame numpy array, or return static value."""
    if callable(param) and not isinstance(param, (int, float, type(None))):
        return np.array([float(param(i)) for i in range(total)])
    return param


def parse_param(val, dtype=float):
    """Parse a CLI parameter as a static number or kcurve spec."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    try:
        return dtype(val)
    except (ValueError, TypeError):
        return kcurve_parse(val)


def progress_bar(current, total, elapsed, bar_width=40):
    frac = current / total
    filled = int(bar_width * frac)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = frac * 100

    if current > 0 and elapsed > 0:
        eta = elapsed / current * (total - current)
        mins, secs = divmod(int(eta), 60)
        eta_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
    else:
        eta_str = "..."

    elapsed_mins, elapsed_secs = divmod(int(elapsed), 60)
    elapsed_str = f"{elapsed_mins}m{elapsed_secs:02d}s" if elapsed_mins else f"{elapsed_secs}s"

    print(f"\r  {bar} {pct:5.1f}%  {current}/{total}  elapsed {elapsed_str}  eta {eta_str}   ", end="", flush=True)


# ---------------------------------------------------------------------------
# Generic shared-memory multicore processing (used by parallelizable effects)
# ---------------------------------------------------------------------------

_par = {}


def _init_parallel(in_name, out_name, shape, dtype_str, counter, effect, n, extra):
    _par['in_shm'] = shared_memory.SharedMemory(name=in_name)
    _par['out_shm'] = shared_memory.SharedMemory(name=out_name)
    dt = np.dtype(dtype_str)
    _par['frames'] = np.ndarray(shape, dtype=dt, buffer=_par['in_shm'].buf)
    _par['output'] = np.ndarray(shape, dtype=dt, buffer=_par['out_shm'].buf)
    _par['counter'] = counter
    _par['effect'] = effect
    _par['n'] = n
    _par['extra'] = extra if extra else {}


def _process_one(i):
    # Resolve animated parameters for this frame
    saved_n = _par['n']
    if isinstance(saved_n, np.ndarray):
        _par['n'] = max(1, int(round(saved_n[i])))
    saved_extra = {}
    extra = _par.get('extra') or {}
    for key in ('step', 'sigma', 'anchor'):
        if key in extra and isinstance(extra[key], np.ndarray):
            saved_extra[key] = extra[key]
            extra[key] = float(extra[key][i])

    _PARALLEL_WORKERS[_par['effect']](i)

    # Restore for next call in this process
    _par['n'] = saved_n
    for key, val in saved_extra.items():
        extra[key] = val

    with _par['counter'].get_lock():
        _par['counter'].value += 1


def _parallel_effect(frames, n, effect, cores, extra=None):
    """Run a parallelizable effect across multiple cores using shared memory."""
    total = len(frames)
    h, w, c = frames[0].shape
    shape = (total, h, w, c)
    dtype = np.uint8
    frame_bytes = int(np.prod(shape))

    in_shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
    out_shm = shared_memory.SharedMemory(create=True, size=frame_bytes)

    try:
        in_arr = np.ndarray(shape, dtype=dtype, buffer=in_shm.buf)
        for idx, f in enumerate(frames):
            in_arr[idx] = f

        counter = mp.Value('i', 0)
        use_cores = min(cores, total)

        t0 = time.time()
        pool = mp.Pool(
            use_cores,
            initializer=_init_parallel,
            initargs=(in_shm.name, out_shm.name, shape,
                      np.dtype(dtype).str, counter, effect, n, extra),
        )

        result = pool.map_async(_process_one, range(total))

        while not result.ready():
            progress_bar(counter.value, total, time.time() - t0)
            time.sleep(0.1)

        result.get()
        pool.close()
        pool.join()

        progress_bar(total, total, time.time() - t0)
        print()

        out_arr = np.ndarray(shape, dtype=dtype, buffer=out_shm.buf)
        out_frames = [out_arr[i].copy() for i in range(total)]

    finally:
        in_shm.close()
        in_shm.unlink()
        out_shm.close()
        out_shm.unlink()

    return out_frames


# --- Per-effect parallel worker functions ---

def _pw_echo(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    anchor = _par['extra'].get('anchor', 1.0)
    step = max(1, int(round(_par['extra'].get('step', 1))))
    idxs = _echo_indices(i, n, total, anchor, step)
    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    for fi in idxs:
        acc += frames[fi].astype(np.float64)
    acc /= len(idxs)
    output[i] = np.clip(acc, 0, 255).astype(np.uint8)


def _pw_gaussian(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    sigma = _par['extra'].get('sigma', n / 4.0)
    half = n // 2
    full_size = 2 * half + 1
    weights = np.exp(-0.5 * (np.arange(full_size) - half) ** 2 / (sigma ** 2))
    start = max(0, i - half)
    end = min(total, i + half + 1)
    w_start = start - (i - half)
    w = weights[w_start:w_start + (end - start)]
    w = w / w.sum()
    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    for j, fi in enumerate(range(start, end)):
        acc += frames[fi].astype(np.float64) * w[j]
    output[i] = np.clip(acc, 0, 255).astype(np.uint8)


def _pw_slit_scan(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    height = frames.shape[1]
    for row in range(height):
        offset = int((row / height - 0.5) * n)
        src_idx = min(max(i + offset, 0), total - 1)
        output[i, row] = frames[src_idx, row]


def _pw_median(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    stack = frames[start:end].astype(np.uint8)
    output[i] = np.median(stack, axis=0).astype(np.uint8)


def _pw_time_ramp(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    win = max(1, int(1 + (n - 1) * i / max(1, total - 1)))
    start = max(0, i - win + 1)
    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    for fi in range(start, i + 1):
        acc += frames[fi].astype(np.float64)
    acc /= (i + 1 - start)
    output[i] = np.clip(acc, 0, 255).astype(np.uint8)


def _pw_strobe(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    step = _par['extra']['step']
    indices = list(range(i, max(i - n * step, -1), -step))
    indices = [idx for idx in indices if 0 <= idx < total]
    if not indices:
        indices = [i]
    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    for idx in indices:
        acc += frames[idx].astype(np.float64)
    acc /= len(indices)
    output[i] = np.clip(acc, 0, 255).astype(np.uint8)


def _pw_ping_pong(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    count = 0
    for offset in range(-half, half + 1):
        fwd = i + offset
        rev = i - offset
        for idx in (fwd, rev):
            if 0 <= idx < total:
                acc += frames[idx].astype(np.float64)
                count += 1
    acc /= max(count, 1)
    output[i] = np.clip(acc, 0, 255).astype(np.uint8)


def _pw_rolling_shutter(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    height = frames.shape[1]
    for row in range(height):
        offset = int(row / height * n)
        src_idx = min(max(i + offset, 0), total - 1)
        output[i, row] = frames[src_idx, row]


def _pw_brightest(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    bright = np.max(frames[start:end], axis=0)
    output[i] = cv2.addWeighted(bright, 0.85, frames[i], 0.15, 0)


def _pw_darkest(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    dark = np.min(frames[start:end], axis=0)
    output[i] = cv2.addWeighted(dark, 0.85, frames[i], 0.15, 0)


def _pw_temporal_gradient(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    stack = frames[start:end].astype(np.float64)
    std = np.std(stack, axis=0)
    magnitude = np.max(std, axis=2).astype(np.float32)
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = (magnitude / mag_max * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)
    output[i] = cv2.applyColorMap(magnitude, cv2.COLORMAP_TURBO)


def _pw_brightest_eq(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    bright = np.max(frames[start:end], axis=0)
    output[i] = equalize_frame(bright)


def _pw_darkest_eq(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    dark = np.min(frames[start:end], axis=0)
    output[i] = equalize_frame(dark)


def _pw_brightest_edge(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    bright = np.max(frames[start:end], axis=0)
    eq = equalize_frame(bright)
    gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    eq[edges > 0] = 255
    output[i] = eq


def _pw_darkest_edge(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    dark = np.min(frames[start:end], axis=0)
    eq = equalize_frame(dark)
    gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    eq[edges > 0] = 255
    output[i] = eq


def _pw_bitwise_or(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    acc = frames[start].copy()
    for j in range(start + 1, end):
        acc = cv2.bitwise_or(acc, frames[j])
    output[i] = acc


def _pw_bitwise_and(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    acc = frames[start].copy()
    for j in range(start + 1, end):
        acc = cv2.bitwise_and(acc, frames[j])
    output[i] = acc


def _pw_bitwise_nor(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    acc = frames[start].copy()
    for j in range(start + 1, end):
        acc = cv2.bitwise_or(acc, frames[j])
    output[i] = cv2.bitwise_not(acc)


def _pw_bitwise_nor_eq(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    acc = frames[start].copy()
    for j in range(start + 1, end):
        acc = cv2.bitwise_or(acc, frames[j])
    output[i] = equalize_frame(cv2.bitwise_not(acc))


def _pw_screen(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    log_sum = np.zeros(frames.shape[1:], dtype=np.float32)
    for fi in range(start, end):
        log_sum += np.log((1.0 - frames[fi].astype(np.float32) / 255.0) + 1e-10)
    out = (1.0 - np.exp(log_sum)) * 255
    output[i] = out.clip(0, 255).astype(np.uint8)


def _pw_multiply(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    count = end - start
    log_sum = np.zeros(frames.shape[1:], dtype=np.float32)
    for fi in range(start, end):
        log_sum += np.log(frames[fi].astype(np.float32) / 255.0 + 1e-10)
    out = np.exp(log_sum / count) * 255
    output[i] = out.clip(0, 255).astype(np.uint8)


def _pw_temporal_variance(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    half = n // 2
    start = max(0, i - half)
    end = min(total, i + half + 1)
    grays = []
    for fi in range(start, end):
        grays.append(cv2.cvtColor(frames[fi], cv2.COLOR_BGR2GRAY).astype(np.float32))
    stack = np.stack(grays, axis=0)
    std = np.std(stack, axis=0)
    std_max = std.max()
    if std_max > 0:
        norm = (std / std_max * 255).astype(np.uint8)
    else:
        norm = np.zeros(std.shape, dtype=np.uint8)
    output[i] = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def _pw_hue_trails(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    max_samples = 12
    start = max(0, i - n + 1)
    count = i - start + 1
    if count > max_samples:
        indices = np.linspace(start, i, max_samples, dtype=int)
    else:
        indices = list(range(start, i + 1))
    acc = np.zeros(frames.shape[1:], dtype=np.float32)
    w_sum = 0.0
    num = len(indices)
    for k, idx in enumerate(indices):
        age = num - 1 - k
        hsv = cv2.cvtColor(frames[idx].copy(), cv2.COLOR_BGR2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + age * 8) % 180).astype(np.uint8)
        w = 1.0 / (1 + age * 0.4)
        acc += cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) * w
        w_sum += w
    acc /= w_sum
    output[i] = acc.clip(0, 255).astype(np.uint8)


def _pw_time_mosaic(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    h, w = frames.shape[1], frames.shape[2]
    grid = 8
    tile_h, tile_w = h // grid, w // grid
    start = max(0, i - n + 1)
    window_len = i - start + 1
    out = np.zeros(frames.shape[1:], dtype=np.uint8)
    for gy in range(grid):
        for gx in range(grid):
            idx = (gy * grid + gx) % window_len
            src = start + idx
            y0 = gy * tile_h
            y1 = (gy + 1) * tile_h if gy < grid - 1 else h
            x0 = gx * tile_w
            x1 = (gx + 1) * tile_w if gx < grid - 1 else w
            out[y0:y1, x0:x1] = frames[src][y0:y1, x0:x1]
    output[i] = out


_PARALLEL_WORKERS = {
    "echo": _pw_echo,
    "gaussian": _pw_gaussian,
    "slit-scan": _pw_slit_scan,
    "median": _pw_median,
    "time-ramp": _pw_time_ramp,
    "strobe": _pw_strobe,
    "ping-pong": _pw_ping_pong,
    "rolling-shutter": _pw_rolling_shutter,
    "brightest": _pw_brightest,
    "darkest": _pw_darkest,
    "temporal-gradient": _pw_temporal_gradient,
    "brightest-eq": _pw_brightest_eq,
    "darkest-eq": _pw_darkest_eq,
    "brightest-edge": _pw_brightest_edge,
    "darkest-edge": _pw_darkest_edge,
    "bitwise-or": _pw_bitwise_or,
    "bitwise-and": _pw_bitwise_and,
    "bitwise-nor": _pw_bitwise_nor,
    "bitwise-nor-eq": _pw_bitwise_nor_eq,
    "screen": _pw_screen,
    "multiply": _pw_multiply,
    "temporal-variance": _pw_temporal_variance,
    "hue-trails": _pw_hue_trails,
    "time-mosaic": _pw_time_mosaic,
}


def load_frames(input_path):
    """Load all frames from a video into memory."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    print("  Reading all frames into memory...")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  Loaded {len(frames)} frames")
    return frames, fps, width, height


# ---------------------------------------------------------------------------
# Streaming mode infrastructure — O(window) memory instead of O(total)
# ---------------------------------------------------------------------------

def probe_video(input_path):
    """Get video metadata without loading frames."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total, fps, width, height


def _fmt_bytes(b):
    """Format byte count as human-readable string."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def get_available_memory():
    """Get available system memory in bytes."""
    import re as _re
    # macOS
    try:
        r = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        total_mem = int(r.stdout.strip())
        r = subprocess.run(['vm_stat'], capture_output=True, text=True)
        ps_match = _re.search(r'page size of (\d+)', r.stdout)
        page_size = int(ps_match.group(1)) if ps_match else 16384
        free = 0
        for label in ('Pages free', 'Pages inactive', 'Pages purgeable'):
            m = _re.search(rf'{label}:\s+(\d+)', r.stdout)
            if m:
                free += int(m.group(1)) * page_size
        return int(free * 0.8) if free > 0 else int(total_mem * 0.5)
    except Exception:
        pass
    # Linux
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 8 * 1024**3  # fallback 8 GB


def estimate_memory_bytes(total, width, height, n, cores, effect_name):
    """Estimate peak RAM usage for loading all frames."""
    frame_bytes = width * height * 3
    mem = frame_bytes * total * 2  # input + output lists
    if cores > 1 and effect_name in _PARALLEL_WORKERS:
        mem += frame_bytes * total * 2  # shared memory copies
    if effect_name in ('screen', 'multiply'):
        mem += width * height * 3 * 4 * total  # float32 precomputed
    elif effect_name == 'temporal-variance':
        mem += width * height * 4 * total
    return mem


class FrameBuffer:
    """Lazy-loading video frame buffer with eviction for streaming mode."""

    def __init__(self, video_path, total, transform=None):
        self.cap = cv2.VideoCapture(video_path)
        self.total = total
        self._cache = {}
        self._next_read = 0
        self._transform = transform
        self._read_to(0)
        self.frame_shape = self._cache[0].shape

    def __len__(self):
        return self.total

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total)
            return [self[i] for i in range(start, stop, step or 1)]
        if key < 0:
            key += self.total
        if key not in self._cache:
            self._read_to(key)
        return self._cache[key]

    def _read_to(self, target):
        while self._next_read <= target:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self._transform:
                frame = self._transform(self._next_read, frame)
            self._cache[self._next_read] = frame
            self._next_read += 1

    def evict_before(self, idx):
        for k in [k for k in self._cache if k < idx]:
            del self._cache[k]

    def cached_count(self):
        return len(self._cache)

    def close(self):
        self.cap.release()


def write_and_mux(out_frames, fps, width, height, output_path, input_path, no_audio=False):
    """Write frames to mp4v then re-encode with H.264 and mux audio."""
    tmp_path = str(Path(output_path).with_suffix("")) + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    for f in out_frames:
        out.write(f)
    out.release()

    try:
        final_path = str(Path(output_path).with_suffix("")) + "_final.mp4"
        if no_audio:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-an",
                final_path,
            ]
            print("  Re-encoding with H.264 (no audio)...")
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-i", input_path,
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-c:a", "copy",
                "-shortest",
                final_path,
            ]
            print("  Re-encoding with H.264 and muxing audio...")
        subprocess.run(cmd, check=True, capture_output=True)
        Path(tmp_path).unlink()
        Path(final_path).rename(output_path)
        print(f"  Final output: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Note: ffmpeg step skipped ({e}). Output is raw mp4v.")
        Path(tmp_path).rename(output_path)


# ---------------------------------------------------------------------------
# Effect implementations
# ---------------------------------------------------------------------------

def _echo_window(i, n_i, total, anchor):
    """Compute start/end indices for echo window given anchor position.
    anchor=0: current frame is start of window, anchor=1: current frame is end."""
    back = int(round(anchor * (n_i - 1)))
    fwd = n_i - 1 - back
    start = max(0, i - back)
    end = min(total, i + fwd + 1)
    return start, end


def _echo_indices(i, n_i, total, anchor, step):
    """Frame indices for an echo blend: the anchor-positioned n-frame window,
    sampling every `step`-th frame within it. Frame i is always included, so
    n=60 step=2 spans a 60-frame window but blends 30 frames (every other)."""
    start, end = _echo_window(i, n_i, total, anchor)
    if step <= 1:
        return list(range(start, end))
    return [idx for idx in range(start, end) if (i - idx) % step == 0]


def fx_echo(frames, n, cores=1, **kw):
    """Blend only previous N frames (motion trails)."""
    anchor = kw.get('anchor', 1.0)
    step = kw.get('step', 1)
    if cores > 1:
        return _parallel_effect(frames, n, "echo", cores,
                                extra={'anchor': anchor, 'step': step})
    total = len(frames)
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        a_i = float(_at(anchor, i)) if _is_animated(anchor) else float(anchor)
        step_i = max(1, int(round(_at(step, i))))
        idxs = _echo_indices(i, n_i, total, a_i, step_i)
        acc = np.zeros_like(frames[0], dtype=np.float64)
        for idx in idxs:
            acc += frames[idx].astype(np.float64)
        acc /= len(idxs)
        result.append(np.clip(acc, 0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_gaussian(frames, n, cores=1, sigma=None, **kw):
    """Gaussian-weighted blend across N frames — bell curve falloff from center."""
    if sigma is None:
        sigma = n / 4.0 if not _is_animated(n) else n.astype(float) / 4.0
    if cores > 1:
        return _parallel_effect(frames, n, "gaussian", cores,
                                extra={'sigma': sigma})
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        sigma_i = float(_at(sigma, i)) if _is_animated(sigma) else (sigma if not _is_animated(n) else n_i / 4.0)
        full_size = 2 * half + 1
        kernel = np.exp(-0.5 * (np.arange(full_size) - half) ** 2 / (sigma_i ** 2))
        start = max(0, i - half)
        end = min(total, i + half + 1)
        window = frames[start:end]
        # Align kernel to actual window
        k_start = half - (i - start)
        w = kernel[k_start:k_start + len(window)]
        w = w / w.sum()
        acc = np.zeros_like(frames[0], dtype=np.float64)
        for j, f in enumerate(window):
            acc += f.astype(np.float64) * w[j]
        result.append(acc.clip(0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_slit_scan(frames, n, cores=1, **kw):
    """Each row of pixels comes from a different frame in time."""
    if cores > 1:
        return _parallel_effect(frames, n, "slit-scan", cores)
    total = len(frames)
    height = frames[0].shape[0]
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        out = np.empty_like(frames[0])
        for row in range(height):
            # Map each row to a frame offset within [-n//2, n//2]
            offset = int((row / height - 0.5) * n_i)
            src_idx = np.clip(i + offset, 0, total - 1)
            out[row] = frames[src_idx][row]
        result.append(out)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_diff(frames, n=None, **kw):
    """Absolute difference between consecutive frames."""
    total = len(frames)
    result = []
    t0 = time.time()
    result.append(np.zeros_like(frames[0]))
    progress_bar(1, total, time.time() - t0)
    for i in range(1, total):
        d = cv2.absdiff(frames[i], frames[i - 1])
        result.append(d)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_median(frames, n, cores=1, **kw):
    """Median pixel across N frames (removes moving objects)."""
    if cores > 1:
        return _parallel_effect(frames, n, "median", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        med = np.median(stack, axis=0).astype(np.uint8)
        result.append(med)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_decay(frames, n=None, decay=0.92, **kw):
    """Exponential persistence: output = decay*prev + (1-decay)*current."""
    total = len(frames)
    result = []
    t0 = time.time()
    acc = frames[0].astype(np.float64)
    result.append(frames[0].copy())
    progress_bar(1, total, time.time() - t0)
    for i in range(1, total):
        decay_i = float(_at(decay, i))
        acc = decay_i * acc + (1.0 - decay_i) * frames[i].astype(np.float64)
        result.append(np.clip(acc, 0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_time_ramp(frames, n, cores=1, **kw):
    """Blend window grows from 1 to N over the clip duration."""
    if cores > 1:
        return _parallel_effect(frames, n, "time-ramp", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        # Window size ramps from 1 at frame 0 to n_i at last frame
        win = max(1, int(1 + (n_i - 1) * i / max(1, total - 1)))
        start = max(0, i - win + 1)
        window = frames[start:i + 1]
        acc = np.zeros_like(frames[0], dtype=np.float64)
        for f in window:
            acc += f.astype(np.float64)
        acc /= len(window)
        result.append(np.clip(acc, 0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_strobe(frames, n, step=4, cores=1, **kw):
    """Blend every Kth frame across a wider time span."""
    if cores > 1:
        return _parallel_effect(frames, n, "strobe", cores, extra={'step': step})
    total = len(frames)
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        step_i = max(1, int(round(_at(step, i))))
        indices = list(range(i, max(i - n_i * step_i, -1), -step_i))
        indices = [idx for idx in indices if 0 <= idx < total]
        if not indices:
            indices = [i]
        acc = np.zeros_like(frames[0], dtype=np.float64)
        for idx in indices:
            acc += frames[idx].astype(np.float64)
        acc /= len(indices)
        result.append(np.clip(acc, 0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_ping_pong(frames, n, cores=1, **kw):
    """Average forward + time-reversed frames together."""
    if cores > 1:
        return _parallel_effect(frames, n, "ping-pong", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        acc = np.zeros_like(frames[0], dtype=np.float64)
        count = 0
        for offset in range(-half, half + 1):
            # Forward index
            fwd = i + offset
            # Reverse index (ping-pong mirror)
            rev = i - offset
            for idx in (fwd, rev):
                if 0 <= idx < total:
                    acc += frames[idx].astype(np.float64)
                    count += 1
        acc /= max(count, 1)
        result.append(np.clip(acc, 0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_rolling_shutter(frames, n, cores=1, **kw):
    """Each scanline offset by 1 frame in time (rolling shutter simulation)."""
    if cores > 1:
        return _parallel_effect(frames, n, "rolling-shutter", cores)
    total = len(frames)
    height = frames[0].shape[0]
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        out = np.empty_like(frames[0])
        for row in range(height):
            # Spread n_i frames across the image height
            offset = int(row / height * n_i)
            src_idx = np.clip(i + offset, 0, total - 1)
            out[row] = frames[src_idx][row]
        result.append(out)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_brightest(frames, n, cores=1, **kw):
    """Keep brightest pixel across N frames (light trails)."""
    if cores > 1:
        return _parallel_effect(frames, n, "brightest", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        bright = np.max(stack, axis=0)
        result.append(cv2.addWeighted(bright, 0.85, frames[i], 0.15, 0))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_darkest(frames, n, cores=1, **kw):
    """Keep darkest pixel across N frames."""
    if cores > 1:
        return _parallel_effect(frames, n, "darkest", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        dark = np.min(stack, axis=0)
        result.append(cv2.addWeighted(dark, 0.85, frames[i], 0.15, 0))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_temporal_gradient(frames, n, cores=1, **kw):
    """Map per-pixel temporal change to a color gradient."""
    if cores > 1:
        return _parallel_effect(frames, n, "temporal-gradient", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0).astype(np.float64)
        # Compute temporal standard deviation per pixel
        std = np.std(stack, axis=0)
        # Collapse to single channel (max across BGR)
        magnitude = np.max(std, axis=2).astype(np.float32)
        # Normalize to 0-255
        mag_max = magnitude.max()
        if mag_max > 0:
            magnitude = (magnitude / mag_max * 255).astype(np.uint8)
        else:
            magnitude = magnitude.astype(np.uint8)
        # Apply colormap (TURBO gives a nice gradient)
        colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_TURBO)
        result.append(colored)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def equalize_frame(frame, clip_limit=3.0):
    """Apply CLAHE histogram equalization on the L channel in LAB space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fx_brightest_eq(frames, n, cores=1, **kw):
    """Keep brightest pixel across N frames + CLAHE histogram equalization."""
    if cores > 1:
        return _parallel_effect(frames, n, "brightest-eq", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        bright = np.max(stack, axis=0)
        result.append(equalize_frame(bright))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_darkest_eq(frames, n, cores=1, **kw):
    """Keep darkest pixel across N frames + CLAHE histogram equalization."""
    if cores > 1:
        return _parallel_effect(frames, n, "darkest-eq", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        dark = np.min(stack, axis=0)
        result.append(equalize_frame(dark))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_brightest_edge(frames, n, cores=1, **kw):
    """Brightest pixel + CLAHE equalization with Canny edge overlay."""
    if cores > 1:
        return _parallel_effect(frames, n, "brightest-edge", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        bright = np.max(stack, axis=0)
        eq = equalize_frame(bright)
        gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        eq[edges > 0] = 255
        result.append(eq)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_darkest_edge(frames, n, cores=1, **kw):
    """Darkest pixel + CLAHE equalization with Canny edge overlay."""
    if cores > 1:
        return _parallel_effect(frames, n, "darkest-edge", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(frames[start:end], axis=0)
        dark = np.min(stack, axis=0)
        eq = equalize_frame(dark)
        gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        eq[edges > 0] = 255
        result.append(eq)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_bitwise_or(frames, n, cores=1, **kw):
    """Bitwise OR across N consecutive frames — accumulates all lit pixels."""
    if cores > 1:
        return _parallel_effect(frames, n, "bitwise-or", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        acc = frames[start].copy()
        for j in range(start + 1, end):
            acc = cv2.bitwise_or(acc, frames[j])
        result.append(acc)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_bitwise_and(frames, n, cores=1, **kw):
    """Bitwise AND across N consecutive frames — keeps only persistent pixels."""
    if cores > 1:
        return _parallel_effect(frames, n, "bitwise-and", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        acc = frames[start].copy()
        for j in range(start + 1, end):
            acc = cv2.bitwise_and(acc, frames[j])
        result.append(acc)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_bitwise_nor(frames, n, cores=1, **kw):
    """Bitwise NOR across N consecutive frames — inverse of OR, keeps unlit pixels."""
    if cores > 1:
        return _parallel_effect(frames, n, "bitwise-nor", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        acc = frames[start].copy()
        for j in range(start + 1, end):
            acc = cv2.bitwise_or(acc, frames[j])
        result.append(cv2.bitwise_not(acc))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_bitwise_nor_eq(frames, n, cores=1, **kw):
    """Bitwise NOR across N frames + CLAHE histogram equalization."""
    if cores > 1:
        return _parallel_effect(frames, n, "bitwise-nor-eq", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        acc = frames[start].copy()
        for j in range(start + 1, end):
            acc = cv2.bitwise_or(acc, frames[j])
        result.append(equalize_frame(cv2.bitwise_not(acc)))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_bitwise_xor(frames, n=None, **kw):
    """Bitwise XOR between consecutive frames — highlights per-bit changes."""
    total = len(frames)
    result = []
    t0 = time.time()
    result.append(np.zeros_like(frames[0]))
    progress_bar(1, total, time.time() - t0)
    for i in range(1, total):
        result.append(cv2.bitwise_xor(frames[i], frames[i - 1]))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_screen(frames, n, cores=1, **kw):
    """Screen blend across N frames — combines light, like double-exposure film.
    Screen: 1 - product(1 - f_i/255). Uses rolling log-sum for speed."""
    if cores > 1:
        return _parallel_effect(frames, n, "screen", cores)
    total = len(frames)
    # Precompute log(1 - f/255) for each frame
    log_inv = [np.log((1.0 - frames[i].astype(np.float32) / 255.0) + 1e-10) for i in range(total)]
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    running_sum = None
    prev_start, prev_end = 0, 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        if running_sum is None:
            running_sum = np.sum(np.stack(log_inv[start:end]), axis=0)
        else:
            # Remove frames that left the window
            for j in range(prev_start, start):
                running_sum -= log_inv[j]
            # Add frames that entered the window
            for j in range(prev_end, end):
                running_sum += log_inv[j]
        prev_start, prev_end = start, end
        out = (1.0 - np.exp(running_sum)) * 255
        result.append(out.clip(0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_multiply(frames, n, cores=1, **kw):
    """Multiply blend across N frames — combines shadows, darkens overlaps.
    Normalized: product^(1/count). Uses rolling log-sum for speed."""
    if cores > 1:
        return _parallel_effect(frames, n, "multiply", cores)
    total = len(frames)
    # Precompute log(f/255) for each frame
    log_f = [np.log(frames[i].astype(np.float32) / 255.0 + 1e-10) for i in range(total)]
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    running_sum = None
    prev_start, prev_end = 0, 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        count = end - start
        if running_sum is None:
            running_sum = np.sum(np.stack(log_f[start:end]), axis=0)
        else:
            for j in range(prev_start, start):
                running_sum -= log_f[j]
            for j in range(prev_end, end):
                running_sum += log_f[j]
        prev_start, prev_end = start, end
        out = np.exp(running_sum / count) * 255
        result.append(out.clip(0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_temporal_variance(frames, n, cores=1, **kw):
    """Per-pixel temporal standard deviation mapped to TURBO colormap."""
    if cores > 1:
        return _parallel_effect(frames, n, "temporal-variance", cores)
    total = len(frames)
    # Pre-compute grayscale frames
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    result = []
    t0 = time.time()
    half = n // 2 if not _is_animated(n) else 0
    for i in range(total):
        n_i = max(1, int(round(_at(n, i)))); half = n_i // 2
        start = max(0, i - half)
        end = min(total, i + half + 1)
        stack = np.stack(grays[start:end], axis=0)
        std = np.std(stack, axis=0)
        std_max = std.max()
        if std_max > 0:
            norm = (std / std_max * 255).astype(np.uint8)
        else:
            norm = np.zeros(std.shape, dtype=np.uint8)
        result.append(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_hue_trails(frames, n, cores=1, **kw):
    """Echo with progressive hue shift — rainbow-colored motion trails.
    Samples up to 12 frames from the window for speed."""
    if cores > 1:
        return _parallel_effect(frames, n, "hue-trails", cores)
    total = len(frames)
    max_samples = 12
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        start = max(0, i - n_i + 1)
        count = i - start + 1
        # Sample evenly if window is larger than max_samples
        if count > max_samples:
            indices = np.linspace(start, i, max_samples, dtype=int)
        else:
            indices = list(range(start, i + 1))
        acc = np.zeros(frames[0].shape, dtype=np.float32)
        w_sum = 0.0
        num = len(indices)
        for k, idx in enumerate(indices):
            age = num - 1 - k
            hsv = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2HSV)
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + age * 8) % 180).astype(np.uint8)
            w = 1.0 / (1 + age * 0.4)
            acc += cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) * w
            w_sum += w
        acc /= w_sum
        result.append(acc.clip(0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_time_mosaic(frames, n, cores=1, **kw):
    """Grid of tiles, each from a different moment in the N-frame window."""
    if cores > 1:
        return _parallel_effect(frames, n, "time-mosaic", cores)
    total = len(frames)
    h, w = frames[0].shape[:2]
    grid = 8
    tile_h, tile_w = h // grid, w // grid
    result = []
    t0 = time.time()
    for i in range(total):
        n_i = max(1, int(round(_at(n, i))))
        start = max(0, i - n_i + 1)
        window = frames[start:i + 1]
        count = len(window)
        out = np.zeros_like(frames[0])
        for gy in range(grid):
            for gx in range(grid):
                idx = (gy * grid + gx) % count
                y0, y1 = gy * tile_h, (gy + 1) * tile_h if gy < grid - 1 else h
                x0, x1 = gx * tile_w, (gx + 1) * tile_w if gx < grid - 1 else w
                out[y0:y1, x0:x1] = window[idx][y0:y1, x0:x1]
        result.append(out)
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_feedback(frames, n=None, **kw):
    """Recursive frame blending with slight zoom — video feedback loop effect."""
    total = len(frames)
    result = []
    t0 = time.time()
    h, w = frames[0].shape[:2]
    blend_alpha = 0.7  # how much feedback vs original
    zoom = 1.03  # subtle zoom per frame
    # zoom matrix — scale from center
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    feedback = frames[0].copy()
    result.append(feedback.copy())
    progress_bar(1, total, time.time() - t0)
    for i in range(1, total):
        zoomed = cv2.warpAffine(feedback, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        feedback = cv2.addWeighted(zoomed, blend_alpha, frames[i], 1.0 - blend_alpha, 0)
        result.append(feedback.copy())
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_motion_streak(frames, n=None, **kw):
    """Directional motion blur along optical flow vectors."""
    total = len(frames)
    result = []
    t0 = time.time()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    result.append(frames[0].copy())
    progress_bar(1, total, time.time() - t0)
    streak_len = 12  # pixel length of streak
    for i in range(1, total):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_gray, curr_gray, None)
        h, w = frames[i].shape[:2]
        # Warp frame along flow to create streak
        acc = frames[i].astype(np.float64)
        steps = 6
        for s in range(1, steps + 1):
            frac = s / steps * streak_len
            map_x = np.arange(w, dtype=np.float32)[None, :] - flow[..., 0] * frac / streak_len
            map_y = np.arange(h, dtype=np.float32)[:, None] - flow[..., 1] * frac / streak_len
            warped = cv2.remap(frames[i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acc += warped.astype(np.float64)
        acc /= (steps + 1)
        result.append(acc.clip(0, 255).astype(np.uint8))
        prev_gray = curr_gray
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def flow_to_bgr(flow):
    """Convert 2-channel optical flow (dx, dy) to BGR via HSV visualization.
    Hue = direction of motion, Value = magnitude (speed), Saturation = 255.
    """
    dx, dy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue 0-179
    hsv[..., 1] = 255  # Saturation
    mag_max = mag.max()
    if mag_max > 0:
        hsv[..., 2] = (np.clip(mag / mag_max, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def fx_flow_dis(frames, n=None, **kw):
    """Dense optical flow using DIS (fast)."""
    total = len(frames)
    result = []
    t0 = time.time()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    result.append(np.zeros_like(frames[0]))
    progress_bar(1, total, time.time() - t0)
    for i in range(1, total):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_gray, curr_gray, None)
        result.append(flow_to_bgr(flow))
        prev_gray = curr_gray
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


FARNEBACK_PRESETS = {
    "low":    (0.5, 3, 15, 3, 5, 1.2, 0),
    "medium": (0.5, 5, 21, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN),
    "high":   (0.4, 7, 31, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN),
}


def fx_flow_farneback(frames, n=None, quality="low", **kw):
    """Dense optical flow using Farneback (classic)."""
    params = FARNEBACK_PRESETS.get(quality, FARNEBACK_PRESETS["low"])
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = params
    print(f"  quality={quality} (pyr_scale={pyr_scale}, levels={levels}, winsize={winsize}, "
          f"iterations={iterations}, poly_n={poly_n}, poly_sigma={poly_sigma})")
    total = len(frames)
    result = []
    t0 = time.time()
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    result.append(np.zeros_like(frames[0]))
    progress_bar(1, total, time.time() - t0)
    for i in range(1, total):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
        )
        result.append(flow_to_bgr(flow))
        prev_gray = curr_gray
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_flow_raft(frames, n=None, **kw):
    """Dense optical flow using RAFT AI model (highest quality, slowest)."""
    try:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    except ImportError:
        print("  Error: RAFT requires torch and torchvision >= 0.22.0")
        print("  Install with: pip install torch torchvision>=0.22.0")
        return [np.zeros_like(frames[0])] * len(frames)

    # Determine inference resolution — RAFT correlation volume is O(H*W*H*W/64)
    # so large frames need downscaling for inference
    h_orig, w_orig = frames[0].shape[:2]
    max_dim = 640  # safe for MPS/CPU memory
    scale = min(max_dim / h_orig, max_dim / w_orig, 1.0)
    inf_h = (int(h_orig * scale) // 8) * 8
    inf_w = (int(w_orig * scale) // 8) * 8
    if scale < 1.0:
        print(f"  Downscaling {w_orig}x{h_orig} -> {inf_w}x{inf_h} for RAFT inference")

    # Pick device — try MPS, fall back to CPU
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        try:
            t = torch.randn(1, 3, 8, 8, device="mps")
            _ = t + t
            device = torch.device("mps")
            print(f"  Using MPS device")
        except Exception:
            print(f"  MPS unavailable, using CPU")
    else:
        print(f"  Using CPU device")

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()
    transforms = weights.transforms()

    total = len(frames)
    result = []
    t0 = time.time()
    result.append(np.zeros_like(frames[0]))
    progress_bar(1, total, time.time() - t0)

    def _run_raft(prev_frame, curr_frame, dev):
        """Run RAFT on a frame pair, return flow at original resolution."""
        # BGR -> RGB, HWC -> CHW, float [0,1]
        prev_t = torch.from_numpy(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        curr_t = torch.from_numpy(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        # Downscale for inference
        prev_t = torch.nn.functional.interpolate(
            prev_t.unsqueeze(0), size=(inf_h, inf_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        curr_t = torch.nn.functional.interpolate(
            curr_t.unsqueeze(0), size=(inf_h, inf_w), mode="bilinear", align_corners=False
        ).squeeze(0)

        prev_t, curr_t = transforms(prev_t, curr_t)
        batch_prev = prev_t.unsqueeze(0).to(dev)
        batch_curr = curr_t.unsqueeze(0).to(dev)

        with torch.no_grad():
            flow_preds = model(batch_prev, batch_curr)

        flow = flow_preds[-1].squeeze(0).cpu().numpy()  # (2, inf_h, inf_w)
        flow = flow.transpose(1, 2, 0)  # (inf_h, inf_w, 2)

        # Upscale flow back to original resolution and rescale vectors
        if inf_h != h_orig or inf_w != w_orig:
            flow = cv2.resize(flow, (w_orig, h_orig))
            flow[..., 0] *= w_orig / inf_w
            flow[..., 1] *= h_orig / inf_h

        return flow

    for i in range(1, total):
        try:
            flow = _run_raft(frames[i - 1], frames[i], device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "MPS" in str(e):
                if device.type != "cpu":
                    print(f"\n  MPS OOM at frame {i}, falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.to(device)
                    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    flow = _run_raft(frames[i - 1], frames[i], device)
                else:
                    raise
            else:
                raise

        result.append(flow_to_bgr(flow))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


# ---------------------------------------------------------------------------
# Effect registry
# ---------------------------------------------------------------------------

EFFECTS = {
    "echo":              (fx_echo,              30,  "Blend previous N frames (motion trails)"),
    "gaussian":          (fx_gaussian,           30,  "Gaussian-weighted blend (bell curve falloff)"),
    "slit-scan":         (fx_slit_scan,         120, "Each row from a different frame in time"),
    "diff":              (fx_diff,              None, "Absolute difference between consecutive frames"),
    "median":            (fx_median,             15,  "Median pixel across N frames"),
    "decay":             (fx_decay,             None, "Exponential persistence"),
    "time-ramp":         (fx_time_ramp,          60,  "Blend window grows from 1 to N"),
    "strobe":            (fx_strobe,             30,  "Blend every Kth frame across wider span"),
    "ping-pong":         (fx_ping_pong,          30,  "Average forward + reversed frames"),
    "rolling-shutter":   (fx_rolling_shutter,    30,  "Each scanline offset in time"),
    "brightest":         (fx_brightest,           45, "Keep brightest pixel (light trails)"),
    "darkest":           (fx_darkest,             45, "Keep darkest pixel"),
    "temporal-gradient": (fx_temporal_gradient,   30, "Temporal change mapped to color gradient"),
    "brightest-eq":      (fx_brightest_eq,        45, "Brightest pixel + CLAHE equalization"),
    "darkest-eq":        (fx_darkest_eq,          45, "Darkest pixel + CLAHE equalization"),
    "brightest-edge":    (fx_brightest_edge,      45, "Brightest pixel + CLAHE + Canny edge overlay"),
    "darkest-edge":      (fx_darkest_edge,        45, "Darkest pixel + CLAHE + Canny edge overlay"),
    "bitwise-or":        (fx_bitwise_or,          15, "Bitwise OR across N frames (accumulate lit pixels)"),
    "bitwise-and":       (fx_bitwise_and,         15, "Bitwise AND across N frames (keep persistent pixels)"),
    "bitwise-nor":       (fx_bitwise_nor,         15, "Bitwise NOR across N frames (inverse OR, keeps unlit pixels)"),
    "bitwise-nor-eq":    (fx_bitwise_nor_eq,      15, "Bitwise NOR + CLAHE equalization"),
    "bitwise-xor":       (fx_bitwise_xor,        None, "Bitwise XOR between consecutive frames"),
    "screen":            (fx_screen,              30, "Screen blend across N frames (double-exposure)"),
    "multiply":          (fx_multiply,            30, "Multiply blend across N frames (shadow combine)"),
    "temporal-variance": (fx_temporal_variance,   30, "Per-pixel temporal std-dev mapped to TURBO colormap"),
    "hue-trails":        (fx_hue_trails,          30, "Echo with progressive hue shift (rainbow trails)"),
    "time-mosaic":       (fx_time_mosaic,         60, "Grid of tiles from different moments in time"),
    "feedback":          (fx_feedback,           None, "Recursive blend with slight zoom (feedback loop)"),
    "motion-streak":     (fx_motion_streak,      None, "Directional blur along optical flow vectors"),
    "flow-dis":          (fx_flow_dis,           None, "Dense optical flow (DIS, fast)"),
    "flow-farneback":    (fx_flow_farneback,     None, "Dense optical flow (Farneback, classic)"),
    "flow-raft":         (fx_flow_raft,          None, "Dense optical flow (RAFT AI model)"),
}


def edge_enhance(frame, strength=0.5, thickness=3):
    """Multiply frame by an edge mask (black edges on white background).
    Computes Sobel edge magnitude, dilates with Gaussian blur for thickness, then:
      mask = 1 - strength * edges
      output = frame * mask
    Dark edge lines survive temporal averaging."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mag_max = mag.max()
    if mag_max > 0:
        edges = (mag / mag_max).astype(np.float32)
    else:
        edges = np.zeros_like(gray, dtype=np.float32)
    # Thicken edges with Gaussian blur, then re-normalize
    ksize = thickness * 2 + 1
    edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
    e_max = edges.max()
    if e_max > 0:
        edges = edges / e_max
    # White background, black edges — strength controls how dark the lines are
    mask = 1.0 - strength * edges
    out = frame.astype(np.float32) * mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_post_eq(frame, clip_limit):
    """Apply CLAHE histogram equalization to restore contrast after processing."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_gamma(frame, gamma):
    """Apply gamma correction via LUT. gamma>1 brightens midtones, <1 darkens."""
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.clip((np.linspace(0.0, 1.0, 256) ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def apply_sharpen(frame, amount, radius=3.0):
    """Unsharp mask. amount scales the high-frequency detail added back
    (0 = no-op, ~0.5-1.5 typical); radius is the Gaussian blur sigma in px."""
    if amount <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), max(radius, 1e-6))
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def apply_orig_mix(frame, original, amount):
    """Blend a fraction of the original frame back over the result.
    amount 0 = pure result, 1 = pure original."""
    if amount <= 0:
        return frame
    a = min(amount, 1.0)
    return cv2.addWeighted(frame, 1.0 - a, original, a, 0)


def _print_param(label, val, fmt=".2f"):
    """Print a parameter value — static or animated."""
    if _is_animated(val):
        print(f"  {label} = animated ({val.min():{fmt}} .. {val.max():{fmt}})")
    elif callable(val) and not isinstance(val, (int, float)):
        print(f"  {label} = animated (kcurve)")
    else:
        print(f"  {label} = {val:{fmt}}")


# ---------------------------------------------------------------------------
# Streaming mode — per-frame processing and pipeline
# ---------------------------------------------------------------------------

def _max_window_radius(effect_name, n, anchor=1.0, step=4):
    """Max lookback and lookahead frames needed by an effect."""
    max_n = int(n.max()) if _is_animated(n) else (int(n) if n is not None else 1)

    if effect_name == 'echo':
        max_a = float(anchor.max()) if _is_animated(anchor) else float(anchor)
        min_a = float(anchor.min()) if _is_animated(anchor) else float(anchor)
        back = int(round(max_a * (max_n - 1)))
        fwd = max_n - 1 - int(round(min_a * (max_n - 1)))
        return back, max(fwd, 0)

    if effect_name in ('slit-scan', 'rolling-shutter'):
        return max_n, max_n

    if effect_name == 'strobe':
        max_step = int(step.max()) if _is_animated(step) else int(step)
        return max_n * max_step, 0

    if effect_name in ('time-ramp', 'hue-trails', 'time-mosaic'):
        return max_n, 0

    if effect_name in ('diff', 'bitwise-xor', 'decay', 'feedback',
                        'motion-streak', 'flow-dis', 'flow-farneback', 'flow-raft'):
        return 1, 0

    # Centered-window effects (gaussian, median, brightest, darkest, etc.)
    half = max_n // 2
    return half, half


def _stream_frame(effect, buf, i, total, n, params, state):
    """Compute one output frame in streaming mode."""
    n_i = max(1, int(round(_at(n, i)))) if n is not None else None
    shape = buf.frame_shape

    def _centered():
        half = n_i // 2
        return max(0, i - half), min(total, i + half + 1)

    if effect == 'echo':
        anchor = params.get('anchor', 1.0)
        a_i = float(_at(anchor, i)) if _is_animated(anchor) else float(anchor)
        step_i = max(1, int(round(_at(params.get('step', 1), i))))
        idxs = _echo_indices(i, n_i, total, a_i, step_i)
        acc = np.zeros(shape, dtype=np.float64)
        for idx in idxs:
            acc += buf[idx].astype(np.float64)
        return np.clip(acc / len(idxs), 0, 255).astype(np.uint8)

    if effect == 'gaussian':
        sigma = params.get('sigma')
        if sigma is not None and _is_animated(sigma):
            sigma_i = float(_at(sigma, i))
        elif sigma is not None:
            sigma_i = float(sigma)
        else:
            sigma_i = n_i / 4.0
        half = n_i // 2
        full = 2 * half + 1
        kernel = np.exp(-0.5 * (np.arange(full) - half) ** 2 / (sigma_i ** 2))
        s, e = max(0, i - half), min(total, i + half + 1)
        window = buf[s:e]
        k_start = half - (i - s)
        w = kernel[k_start:k_start + len(window)]
        w = w / w.sum()
        acc = np.zeros(shape, dtype=np.float64)
        for j, f in enumerate(window):
            acc += f.astype(np.float64) * w[j]
        return acc.clip(0, 255).astype(np.uint8)

    if effect == 'slit-scan':
        h = shape[0]
        out = np.empty(shape, dtype=np.uint8)
        for row in range(h):
            offset = int((row / h - 0.5) * n_i)
            src = np.clip(i + offset, 0, total - 1)
            out[row] = buf[src][row]
        return out

    if effect == 'diff':
        return np.zeros(shape, dtype=np.uint8) if i == 0 else cv2.absdiff(buf[i], buf[i - 1])

    if effect == 'median':
        s, e = _centered()
        return np.median(np.stack(buf[s:e], axis=0), axis=0).astype(np.uint8)

    if effect == 'decay':
        d = float(_at(params.get('decay', 0.92), i))
        if 'acc' not in state:
            state['acc'] = buf[0].astype(np.float64)
            return buf[0].copy()
        state['acc'] = d * state['acc'] + (1.0 - d) * buf[i].astype(np.float64)
        return np.clip(state['acc'], 0, 255).astype(np.uint8)

    if effect == 'time-ramp':
        win = max(1, int(1 + (n_i - 1) * i / max(1, total - 1)))
        s = max(0, i - win + 1)
        window = buf[s:i + 1]
        acc = np.zeros(shape, dtype=np.float64)
        for f in window:
            acc += f.astype(np.float64)
        return np.clip(acc / len(window), 0, 255).astype(np.uint8)

    if effect == 'strobe':
        step_i = max(1, int(round(_at(params.get('step', 4), i))))
        indices = [idx for idx in range(i, max(i - n_i * step_i, -1), -step_i)
                   if 0 <= idx < total] or [i]
        acc = np.zeros(shape, dtype=np.float64)
        for idx in indices:
            acc += buf[idx].astype(np.float64)
        return np.clip(acc / len(indices), 0, 255).astype(np.uint8)

    if effect == 'ping-pong':
        half = n_i // 2
        acc = np.zeros(shape, dtype=np.float64)
        count = 0
        for offset in range(-half, half + 1):
            for idx in (i + offset, i - offset):
                if 0 <= idx < total:
                    acc += buf[idx].astype(np.float64)
                    count += 1
        return np.clip(acc / max(count, 1), 0, 255).astype(np.uint8)

    if effect == 'rolling-shutter':
        h = shape[0]
        out = np.empty(shape, dtype=np.uint8)
        for row in range(h):
            offset = int(row / h * n_i)
            src = np.clip(i + offset, 0, total - 1)
            out[row] = buf[src][row]
        return out

    if effect == 'brightest':
        s, e = _centered()
        bright = np.max(np.stack(buf[s:e], axis=0), axis=0)
        return cv2.addWeighted(bright, 0.85, buf[i], 0.15, 0)

    if effect == 'darkest':
        s, e = _centered()
        dark = np.min(np.stack(buf[s:e], axis=0), axis=0)
        return cv2.addWeighted(dark, 0.85, buf[i], 0.15, 0)

    if effect == 'temporal-gradient':
        s, e = _centered()
        stack = np.stack(buf[s:e], axis=0).astype(np.float64)
        std = np.std(stack, axis=0)
        mag = np.max(std, axis=2).astype(np.float32)
        mx = mag.max()
        mag = (mag / mx * 255).astype(np.uint8) if mx > 0 else mag.astype(np.uint8)
        return cv2.applyColorMap(mag, cv2.COLORMAP_TURBO)

    if effect == 'brightest-eq':
        s, e = _centered()
        return equalize_frame(np.max(np.stack(buf[s:e], axis=0), axis=0))

    if effect == 'darkest-eq':
        s, e = _centered()
        return equalize_frame(np.min(np.stack(buf[s:e], axis=0), axis=0))

    if effect == 'brightest-edge':
        s, e = _centered()
        eq = equalize_frame(np.max(np.stack(buf[s:e], axis=0), axis=0))
        gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        eq[edges > 0] = 255
        return eq

    if effect == 'darkest-edge':
        s, e = _centered()
        eq = equalize_frame(np.min(np.stack(buf[s:e], axis=0), axis=0))
        gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        eq[edges > 0] = 255
        return eq

    if effect in ('bitwise-or', 'bitwise-nor', 'bitwise-nor-eq'):
        s, e = _centered()
        acc = buf[s].copy()
        for j in range(s + 1, e):
            acc = cv2.bitwise_or(acc, buf[j])
        if effect == 'bitwise-or':
            return acc
        inv = cv2.bitwise_not(acc)
        return equalize_frame(inv) if effect == 'bitwise-nor-eq' else inv

    if effect == 'bitwise-and':
        s, e = _centered()
        acc = buf[s].copy()
        for j in range(s + 1, e):
            acc = cv2.bitwise_and(acc, buf[j])
        return acc

    if effect == 'bitwise-xor':
        return np.zeros(shape, dtype=np.uint8) if i == 0 else cv2.bitwise_xor(buf[i], buf[i - 1])

    if effect == 'screen':
        s, e = _centered()
        log_sum = np.zeros(shape, dtype=np.float32)
        for f in buf[s:e]:
            log_sum += np.log((1.0 - f.astype(np.float32) / 255.0) + 1e-10)
        return ((1.0 - np.exp(log_sum)) * 255).clip(0, 255).astype(np.uint8)

    if effect == 'multiply':
        s, e = _centered()
        count = e - s
        log_sum = np.zeros(shape, dtype=np.float32)
        for f in buf[s:e]:
            log_sum += np.log(f.astype(np.float32) / 255.0 + 1e-10)
        return (np.exp(log_sum / count) * 255).clip(0, 255).astype(np.uint8)

    if effect == 'temporal-variance':
        s, e = _centered()
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in buf[s:e]]
        std = np.std(np.stack(grays, axis=0), axis=0)
        mx = std.max()
        norm = (std / mx * 255).astype(np.uint8) if mx > 0 else np.zeros(std.shape, dtype=np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    if effect == 'hue-trails':
        max_samples = 12
        s = max(0, i - n_i + 1)
        count = i - s + 1
        indices = np.linspace(s, i, max_samples, dtype=int) if count > max_samples else list(range(s, i + 1))
        acc = np.zeros(shape, dtype=np.float32)
        w_sum = 0.0
        num = len(indices)
        for k, idx in enumerate(indices):
            age = num - 1 - k
            hsv = cv2.cvtColor(buf[idx], cv2.COLOR_BGR2HSV)
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + age * 8) % 180).astype(np.uint8)
            w = 1.0 / (1 + age * 0.4)
            acc += cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) * w
            w_sum += w
        return (acc / w_sum).clip(0, 255).astype(np.uint8)

    if effect == 'time-mosaic':
        h, w = shape[:2]
        grid = 8
        th, tw = h // grid, w // grid
        s = max(0, i - n_i + 1)
        window = buf[s:i + 1]
        cnt = len(window)
        out = np.zeros(shape, dtype=np.uint8)
        for gy in range(grid):
            for gx in range(grid):
                idx = (gy * grid + gx) % cnt
                y0 = gy * th; y1 = (gy + 1) * th if gy < grid - 1 else h
                x0 = gx * tw; x1 = (gx + 1) * tw if gx < grid - 1 else w
                out[y0:y1, x0:x1] = window[idx][y0:y1, x0:x1]
        return out

    if effect == 'feedback':
        if 'fb' not in state:
            h, w = shape[:2]
            state['fb'] = buf[0].copy()
            state['M'] = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.03)
            state['wh'] = (w, h)
            return buf[0].copy()
        zoomed = cv2.warpAffine(state['fb'], state['M'], state['wh'],
                                borderMode=cv2.BORDER_REFLECT)
        state['fb'] = cv2.addWeighted(zoomed, 0.7, buf[i], 0.3, 0)
        return state['fb'].copy()

    if effect == 'motion-streak':
        if 'dis' not in state:
            state['dis'] = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            state['prev_gray'] = cv2.cvtColor(buf[0], cv2.COLOR_BGR2GRAY)
            return buf[0].copy()
        curr_gray = cv2.cvtColor(buf[i], cv2.COLOR_BGR2GRAY)
        flow = state['dis'].calc(state['prev_gray'], curr_gray, None)
        h, w = shape[:2]
        acc = buf[i].astype(np.float64)
        streak_len, steps = 12, 6
        for si in range(1, steps + 1):
            frac = si / steps * streak_len
            map_x = np.arange(w, dtype=np.float32)[None, :] - flow[..., 0] * frac / streak_len
            map_y = np.arange(h, dtype=np.float32)[:, None] - flow[..., 1] * frac / streak_len
            acc += cv2.remap(buf[i], map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT).astype(np.float64)
        state['prev_gray'] = curr_gray
        return (acc / (steps + 1)).clip(0, 255).astype(np.uint8)

    if effect == 'flow-dis':
        if 'dis' not in state:
            state['dis'] = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            state['prev_gray'] = cv2.cvtColor(buf[0], cv2.COLOR_BGR2GRAY)
            return np.zeros(shape, dtype=np.uint8)
        curr_gray = cv2.cvtColor(buf[i], cv2.COLOR_BGR2GRAY)
        flow = state['dis'].calc(state['prev_gray'], curr_gray, None)
        state['prev_gray'] = curr_gray
        return flow_to_bgr(flow)

    if effect == 'flow-farneback':
        preset = FARNEBACK_PRESETS.get(params.get('quality', 'low'), FARNEBACK_PRESETS['low'])
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = preset
        if 'prev_gray' not in state:
            state['prev_gray'] = cv2.cvtColor(buf[0], cv2.COLOR_BGR2GRAY)
            return np.zeros(shape, dtype=np.uint8)
        curr_gray = cv2.cvtColor(buf[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(state['prev_gray'], curr_gray, None,
                                             pyr_scale, levels, winsize, iterations,
                                             poly_n, poly_sigma, flags)
        state['prev_gray'] = curr_gray
        return flow_to_bgr(flow)

    if effect == 'flow-raft':
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        if 'raft' not in state:
            h_o, w_o = shape[:2]
            max_dim = 640
            sc = min(max_dim / h_o, max_dim / w_o, 1.0)
            inf_h = (int(h_o * sc) // 8) * 8
            inf_w = (int(w_o * sc) // 8) * 8
            device = torch.device("cpu")
            if torch.backends.mps.is_available():
                try:
                    _ = torch.randn(1, 3, 8, 8, device="mps") + torch.randn(1, 3, 8, 8, device="mps")
                    device = torch.device("mps")
                    print(f"  Using MPS device")
                except Exception:
                    print(f"  MPS unavailable, using CPU")
            else:
                print(f"  Using CPU device")
            weights = Raft_Large_Weights.DEFAULT
            state['raft'] = {
                'model': raft_large(weights=weights).to(device).eval(),
                'transforms': weights.transforms(),
                'device': device,
                'inf_h': inf_h, 'inf_w': inf_w, 'h_o': h_o, 'w_o': w_o,
            }
            if sc < 1.0:
                print(f"  Downscaling {w_o}x{h_o} -> {inf_w}x{inf_h} for RAFT inference")
            return np.zeros(shape, dtype=np.uint8)
        r = state['raft']

        def _to_tensor(frame):
            t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            return torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(r['inf_h'], r['inf_w']),
                mode="bilinear", align_corners=False).squeeze(0)

        prev_t, curr_t = r['transforms'](_to_tensor(buf[i - 1]), _to_tensor(buf[i]))
        try:
            with torch.no_grad():
                flow_preds = r['model'](prev_t.unsqueeze(0).to(r['device']),
                                         curr_t.unsqueeze(0).to(r['device']))
        except RuntimeError:
            if r['device'].type != 'cpu':
                r['device'] = torch.device('cpu')
                r['model'] = r['model'].to(r['device'])
                with torch.no_grad():
                    flow_preds = r['model'](prev_t.unsqueeze(0).to(r['device']),
                                             curr_t.unsqueeze(0).to(r['device']))
            else:
                raise
        flow = flow_preds[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        if r['inf_h'] != r['h_o'] or r['inf_w'] != r['w_o']:
            flow = cv2.resize(flow, (r['w_o'], r['h_o']))
            flow[..., 0] *= r['w_o'] / r['inf_w']
            flow[..., 1] *= r['h_o'] / r['inf_h']
        return flow_to_bgr(flow)

    raise ValueError(f"Effect '{effect}' not supported in streaming mode")


def _run_streaming(input_path, effect_name, n, output_path, fps, width, height,
                   total, decay, step, quality, pre_eq, post_eq,
                   sigma, edge_str, edge_thick, no_audio, anchor, gamma=None, edge_gamma=None,
                   sharpen=None, sharpen_radius=3.0, orig_mix=None, hflip=False):
    """Run effect in streaming mode — O(window) memory."""
    base_transform = None
    if edge_str is not None or pre_eq is not None:
        def base_transform(idx, frame):
            if edge_str is not None:
                s = float(_at(edge_str, idx))
                t = int(_at(edge_thick, idx))
                frame = edge_enhance(frame, s, t)
                if edge_gamma is not None:
                    frame = apply_gamma(frame, float(_at(edge_gamma, idx)))
            if pre_eq is not None:
                frame = apply_post_eq(frame, float(_at(pre_eq, idx)))
            return frame

    # Stash untouched originals (keyed by index) so orig-mix can blend them
    # back at output time; evicted right after use to stay O(window).
    originals_cache = {}
    if orig_mix is not None:
        def transform(idx, frame):
            originals_cache[idx] = frame.copy()
            return base_transform(idx, frame) if base_transform else frame
    else:
        transform = base_transform

    lookback, lookahead = _max_window_radius(effect_name, n, anchor=anchor, step=step)
    buf = FrameBuffer(input_path, total, transform=transform)

    peak_frames = lookback + lookahead + 1
    peak_mem = width * height * 3 * peak_frames
    print(f"  Peak buffer: {peak_frames} frames (~{_fmt_bytes(peak_mem)})")

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + f"_{effect_name}"))

    tmp_path = str(Path(output_path).with_suffix("")) + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    params = {'anchor': anchor, 'decay': decay, 'step': step,
              'sigma': sigma, 'quality': quality}
    state = {}

    print(f"  Processing {effect_name} (streaming)...")
    t0 = time.time()
    for i in range(total):
        out_frame = _stream_frame(effect_name, buf, i, total, n, params, state)
        if post_eq is not None:
            out_frame = apply_post_eq(out_frame, float(_at(post_eq, i)))
        if gamma is not None:
            out_frame = apply_gamma(out_frame, float(_at(gamma, i)))
        if sharpen is not None:
            out_frame = apply_sharpen(out_frame, float(_at(sharpen, i)), float(_at(sharpen_radius, i)))
        if orig_mix is not None:
            orig = originals_cache.pop(i, None)
            if orig is not None:
                out_frame = apply_orig_mix(out_frame, orig, float(_at(orig_mix, i)))
        if hflip:
            out_frame = cv2.flip(out_frame, 1)
        writer.write(out_frame)
        min_needed = max(0, i + 1 - lookback)
        buf.evict_before(min_needed)
        progress_bar(i + 1, total, time.time() - t0)

    print()
    writer.release()
    buf.close()

    # FFmpeg re-encode + audio mux
    try:
        final_path = str(Path(output_path).with_suffix("")) + "_final.mp4"
        if no_audio:
            cmd = ["ffmpeg", "-y", "-i", tmp_path,
                   "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                   "-an", final_path]
            print("  Re-encoding with H.264 (no audio)...")
        else:
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-i", input_path,
                   "-map", "0:v", "-map", "1:a?",
                   "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                   "-c:a", "copy", "-shortest", final_path]
            print("  Re-encoding with H.264 and muxing audio...")
        subprocess.run(cmd, check=True, capture_output=True)
        Path(tmp_path).unlink()
        Path(final_path).rename(output_path)
        print(f"  Final output: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Note: ffmpeg step skipped ({e}). Output is raw mp4v.")
        Path(tmp_path).rename(output_path)

    print(f"{'='*60}\n")
    return output_path


def run_effect(input_path, effect_name, n_override, output_path, decay, step, quality="low", pre_eq=None, post_eq=None, cores=1, sigma=None, edge_str=None, edge_thick=3, no_audio=False, reverse=False, anchor=1.0, memory_mode="auto", gamma=None, edge_gamma=None, sharpen=None, sharpen_radius=3.0, orig_mix=None, hflip=False):
    fn, default_n, desc = EFFECTS[effect_name]
    n = n_override if n_override is not None else default_n

    # --step is shared; unset (None) means each effect's natural default:
    # strobe strides by 4, echo strides by 1 (no skip).
    if step is None:
        step = 4 if effect_name == 'strobe' else 1

    print(f"\n{'='*60}")
    print(f"Effect: {effect_name} — {desc}")
    if n is not None:
        if _is_animated(n):
            print(f"  n = animated ({int(n.min())} .. {int(n.max())})")
        else:
            print(f"  n = {n}")
    if pre_eq is not None:
        _print_param("pre-eq", pre_eq, ".1f")
    if post_eq is not None:
        _print_param("post-eq", post_eq, ".1f")
    if gamma is not None:
        _print_param("gamma", gamma)
    if sharpen is not None:
        _print_param("sharpen", sharpen)
        _print_param("sharpen-radius", sharpen_radius, ".1f")
    if orig_mix is not None:
        _print_param("orig-mix", orig_mix)
    if edge_str is not None:
        _print_param("edge-preserve", edge_str)
    if edge_gamma is not None:
        _print_param("edge-gamma", edge_gamma)
    if effect_name == "decay":
        _print_param("decay", decay)
    if effect_name == "strobe":
        _print_param("step", step, "d" if not _is_animated(step) else ".1f")
    if effect_name == "echo" and (_is_animated(step) or step != 1):
        _print_param("step", step, "d" if not _is_animated(step) else ".1f")
    if sigma is not None and effect_name == "gaussian":
        _print_param("sigma", sigma)
    if effect_name == "echo" and not (_is_animated(anchor) or anchor == 1.0):
        _print_param("anchor", anchor)
    elif effect_name == "echo" and _is_animated(anchor):
        _print_param("anchor", anchor)
    if cores > 1:
        print(f"  cores = {cores}")
    if reverse:
        print(f"  reverse = on (blend with future frames)")
    if hflip:
        print(f"  hflip = on")
    if no_audio:
        print(f"  no-audio = on")
    print(f"Input: {input_path}")

    # Probe video metadata
    probe_total, fps, width, height = probe_video(input_path)
    print(f"  {width}x{height}, {fps:.2f} fps, {probe_total} frames")

    # Pre-resolve animated parameters to per-frame arrays
    n = _precompute(n, probe_total)
    decay = _precompute(decay, probe_total)
    step = _precompute(step, probe_total)
    sigma = _precompute(sigma, probe_total) if sigma is not None else None
    pre_eq = _precompute(pre_eq, probe_total) if pre_eq is not None else None
    post_eq = _precompute(post_eq, probe_total) if post_eq is not None else None
    gamma = _precompute(gamma, probe_total) if gamma is not None else None
    anchor = _precompute(anchor, probe_total)
    edge_str = _precompute(edge_str, probe_total) if edge_str is not None else None
    edge_thick = _precompute(edge_thick, probe_total) if edge_thick is not None else None
    edge_gamma = _precompute(edge_gamma, probe_total) if edge_gamma is not None else None
    sharpen = _precompute(sharpen, probe_total) if sharpen is not None else None
    sharpen_radius = _precompute(sharpen_radius, probe_total) if sharpen_radius is not None else None
    orig_mix = _precompute(orig_mix, probe_total) if orig_mix is not None else None

    # Decide memory mode
    estimated = estimate_memory_bytes(probe_total, width, height, n, cores, effect_name)
    available = get_available_memory()
    if memory_mode == "auto":
        use_streaming = estimated > available
    elif memory_mode == "streaming":
        use_streaming = True
    else:
        use_streaming = False

    if use_streaming and reverse:
        print(f"  Note: --reverse requires RAM mode, falling back")
        use_streaming = False

    mode_str = "streaming" if use_streaming else "ram"
    print(f"  memory = {mode_str} (need ~{_fmt_bytes(estimated)}, avail ~{_fmt_bytes(available)})")

    if use_streaming:
        if cores > 1:
            print(f"  Note: --cores ignored in streaming mode")
        return _run_streaming(input_path, effect_name, n, output_path, fps, width, height,
                              probe_total, decay, step, quality, pre_eq, post_eq,
                              sigma, edge_str, edge_thick, no_audio, anchor, gamma, edge_gamma,
                              sharpen, sharpen_radius, orig_mix, hflip)

    # --- RAM mode ---
    print("  Reading all frames into memory...")
    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    total = len(frames)
    print(f"  Loaded {total} frames")

    # Capture untouched originals before pre-processing rebinds frames[i].
    # Pre-eq/edge-enhance reassign list elements (not in-place), so a shallow
    # list copy preserves the original arrays cheaply.
    originals = list(frames) if orig_mix is not None else None

    if edge_str is not None:
        anim_edge = _is_animated(edge_str) or _is_animated(edge_thick)
        print(f"  Enhancing edges on {total} input frames...")
        t0 = time.time()
        for i in range(total):
            s = float(_at(edge_str, i))
            t = int(_at(edge_thick, i))
            frames[i] = edge_enhance(frames[i], s, t)
            if edge_gamma is not None:
                frames[i] = apply_gamma(frames[i], float(_at(edge_gamma, i)))
            progress_bar(i + 1, total, time.time() - t0)
        print()

    if pre_eq is not None:
        anim_eq = _is_animated(pre_eq)
        if not anim_eq:
            print(f"  Equalizing {total} input frames (clip_limit={pre_eq:.1f})...")
            if cores > 1:
                with mp.Pool(min(cores, total)) as pool:
                    frames = pool.starmap(apply_post_eq, [(f, pre_eq) for f in frames])
            else:
                frames = [apply_post_eq(f, pre_eq) for f in frames]
        else:
            print(f"  Equalizing {total} input frames (animated clip_limit)...")
            t0 = time.time()
            for i in range(total):
                frames[i] = apply_post_eq(frames[i], float(pre_eq[i]))
                progress_bar(i + 1, total, time.time() - t0)
            print()

    if reverse:
        print(f"  Reversing frames (blend with future frames)...")
        frames = frames[::-1]

    print(f"  Processing {effect_name}...")
    out_frames = fn(frames, n=n, decay=decay, step=step, quality=quality, cores=cores, sigma=sigma, anchor=anchor)

    if reverse:
        out_frames = out_frames[::-1]

    if post_eq is not None:
        anim_eq = _is_animated(post_eq)
        if not anim_eq:
            print(f"  Applying post-EQ (CLAHE clip_limit={post_eq:.1f})...")
            if cores > 1:
                with mp.Pool(min(cores, len(out_frames))) as pool:
                    out_frames = pool.starmap(apply_post_eq, [(f, post_eq) for f in out_frames])
            else:
                out_frames = [apply_post_eq(f, post_eq) for f in out_frames]
        else:
            print(f"  Applying post-EQ (animated clip_limit)...")
            t0 = time.time()
            total_out = len(out_frames)
            for i in range(total_out):
                out_frames[i] = apply_post_eq(out_frames[i], float(post_eq[i]))
                progress_bar(i + 1, total_out, time.time() - t0)
            print()

    if gamma is not None:
        anim_gamma = _is_animated(gamma)
        if not anim_gamma:
            print(f"  Applying gamma ({gamma:.2f})...")
            if cores > 1:
                with mp.Pool(min(cores, len(out_frames))) as pool:
                    out_frames = pool.starmap(apply_gamma, [(f, gamma) for f in out_frames])
            else:
                out_frames = [apply_gamma(f, gamma) for f in out_frames]
        else:
            print(f"  Applying gamma (animated)...")
            t0 = time.time()
            total_out = len(out_frames)
            for i in range(total_out):
                out_frames[i] = apply_gamma(out_frames[i], float(gamma[i]))
                progress_bar(i + 1, total_out, time.time() - t0)
            print()

    if sharpen is not None:
        anim_sharpen = _is_animated(sharpen) or _is_animated(sharpen_radius)
        if not anim_sharpen:
            print(f"  Applying sharpen (amount={float(sharpen):.2f}, radius={float(sharpen_radius):.1f})...")
            if cores > 1:
                with mp.Pool(min(cores, len(out_frames))) as pool:
                    out_frames = pool.starmap(apply_sharpen, [(f, sharpen, sharpen_radius) for f in out_frames])
            else:
                out_frames = [apply_sharpen(f, sharpen, sharpen_radius) for f in out_frames]
        else:
            print(f"  Applying sharpen (animated)...")
            t0 = time.time()
            total_out = len(out_frames)
            for i in range(total_out):
                out_frames[i] = apply_sharpen(out_frames[i], float(_at(sharpen, i)), float(_at(sharpen_radius, i)))
                progress_bar(i + 1, total_out, time.time() - t0)
            print()

    if orig_mix is not None and originals is not None:
        print(f"  Mixing original back over result...")
        for i in range(len(out_frames)):
            if i < len(originals):
                out_frames[i] = apply_orig_mix(out_frames[i], originals[i], float(_at(orig_mix, i)))

    if hflip:
        print(f"  Flipping horizontally...")
        out_frames = [cv2.flip(f, 1) for f in out_frames]

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + f"_{effect_name}"))

    write_and_mux(out_frames, fps, width, height, output_path, input_path, no_audio=no_audio)
    print(f"{'='*60}\n")
    return output_path


if __name__ == "__main__":
    effects_help = "\n".join(
        f"  {name:20s} {desc}" + (f"  (default n={default_n})" if default_n else "")
        for name, (_, default_n, desc) in EFFECTS.items()
    )
    parser = argparse.ArgumentParser(
        description="Temporal Video Effects Suite — 31 temporal effects for video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Effects:
{effects_help}

Effect-specific options:
  --decay FLOAT       Decay factor for 'decay' effect (default: 0.92)
  --step  INT         Frame stride for 'strobe' (default 4) and 'echo' (default 1).
                      For echo, -n is the window span; n/step frames are blended.
  --sigma FLOAT       Gaussian sigma for 'gaussian' effect (default: n/4)
                        Smaller = sharper peak, larger = flatter
  -q, --quality LEVEL Quality preset for 'flow-farneback' (default: low)
                        low    — fast (pyr_scale=0.5, levels=3, winsize=15, iter=3)
                        medium — balanced (pyr_scale=0.5, levels=5, winsize=21, iter=5, Gaussian)
                        high   — best (pyr_scale=0.4, levels=7, winsize=31, iter=10, Gaussian)

Pre/post-processing:
  --pre-eq CLIP       Apply CLAHE histogram equalization to all input frames
                      before any effect processing (clip limit, e.g. 2.0)
  --post-eq CLIP      Apply CLAHE equalization after processing to restore
                      contrast (clip limit, e.g. 2.0; higher = stronger)
  --gamma G           Gamma correction after processing (>1 brightens midtones,
                      <1 darkens; e.g. 1.5). Applied after post-eq.
  --sharpen AMOUNT    Unsharp-mask sharpen after processing (~0.5-1.5 typical).
  --sharpen-radius PX Gaussian blur radius for --sharpen (default: 3.0).
  --orig-mix FRAC     Blend a fraction of the original frame back over the
                      final result (0.0-1.0). Applied last, on top of everything.

Parallelism:
  --cores N           Number of CPU cores for parallel processing (default: 4)
                      Most effects support shared-memory multicore processing.
                      Not parallelized: decay, feedback (sequential accumulation),
                      flow-dis, flow-farneback, flow-raft, motion-streak (optical flow),
                      diff, bitwise-xor (trivially fast consecutive-frame ops).
                      pre-eq and post-eq are also parallelized.

Animated parameters (kcurve):
  Any numeric parameter (-n, --decay, --step, --sigma, --pre-eq, --post-eq,
  --gamma, --sharpen, --sharpen-radius, --orig-mix, --edge-preserve, --edge-thickness,
  --edge-gamma) accepts animated values via kcurve syntax:

  Keyframes:   value@frame[L|S|B] separated by ":"
               L=linear, S=spline (Catmull-Rom), B=bezier (ease in/out)
  Expressions: math with 'f' as frame number
               sin, cos, lerp, clamp, smoothstep, noise, fbm, pi, etc.

  Preview curves: python -m kcurve "10@0L:60@100S:10@200L"

Examples:
  %(prog)s video.mp4 -e echo
  %(prog)s video.mp4 -e echo -n 60
  %(prog)s video.mp4 -e echo -n 30 --cores 4
  %(prog)s video.mp4 -e gaussian -n 40
  %(prog)s video.mp4 -e gaussian -n 30 --cores 4 --sigma 5.0
  %(prog)s video.mp4 -e decay --decay 0.85
  %(prog)s video.mp4 -e strobe --step 8
  %(prog)s video.mp4 -e brightest -n 30 --pre-eq 2.0
  %(prog)s video.mp4 -e echo -n 60 --post-eq 2.0
  %(prog)s video.mp4 -e flow-farneback -q high
  %(prog)s video.mp4 -e flow-raft
  %(prog)s video.mp4 -e all

  Animated parameters:
  %(prog)s video.mp4 -e echo -n "10@0L:60@200L"
  %(prog)s video.mp4 -e echo -n "sin(f*0.05)*30+30"
  %(prog)s video.mp4 -e decay --decay "0.8@0L:0.99@200S"
  %(prog)s video.mp4 -e gaussian --sigma "1@0L:10@200S"
""",
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument(
        "--effect", "-e", required=True,
        choices=list(EFFECTS.keys()) + ["all"],
        help="Effect to apply (or 'all' to run every effect)",
    )
    parser.add_argument("-n", "--frames", type=str, default=None,
                        help="Number of frames for temporal window (or kcurve spec)")
    parser.add_argument("--decay", type=str, default=0.92,
                        help="Decay factor for 'decay' effect (default: 0.92, or kcurve spec)")
    parser.add_argument("--step", type=str, default=None,
                        help="Frame stride for 'strobe' and 'echo' (blend every Kth frame). "
                             "Default: strobe=4, echo=1 (no skip). kcurve spec allowed. "
                             "For echo, n is the window span and n/step frames are blended.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output video file path")
    parser.add_argument("--sigma", type=str, default=None,
                        help="Gaussian sigma for 'gaussian' effect (default: n/4, or kcurve spec). "
                             "Smaller = sharper peak, larger = flatter")
    parser.add_argument("--quality", "-q", default="low",
                        choices=["low", "medium", "high"],
                        help="Quality preset for flow-farneback (default: low)")
    parser.add_argument("--pre-eq", type=str, default=None, metavar="CLIP",
                        help="Apply CLAHE equalization to input frames before processing "
                             "(clip limit, e.g. 2.0, or kcurve spec)")
    parser.add_argument("--post-eq", type=str, default=None, metavar="CLIP",
                        help="Apply CLAHE equalization after processing to restore contrast "
                             "(clip limit, e.g. 2.0, or kcurve spec)")
    parser.add_argument("--edge-preserve", type=str, default=None, metavar="STRENGTH",
                        help="Preserve edges from source frames via Sobel edge detection. "
                             "Strength 0.0-1.0 controls blend (e.g. 0.5, or kcurve spec). "
                             "Multiplies edge mask with source and blends into processed output.")
    parser.add_argument("--edge-thickness", type=str, default=3,
                        help="Edge line thickness for --edge-preserve (default: 3, or kcurve spec). "
                             "Higher = thicker, more visible edge lines.")
    parser.add_argument("--edge-gamma", type=str, default=None, metavar="G",
                        help="Gamma applied to each frame after the --edge-preserve pass "
                             "(>1 brightens midtones, <1 darkens; e.g. 2.2, or kcurve spec). "
                             "Only active when --edge-preserve is set.")
    parser.add_argument("--gamma", type=str, default=None, metavar="G",
                        help="Gamma correction after processing (>1 brightens midtones, "
                             "<1 darkens; e.g. 1.5, or kcurve spec)")
    parser.add_argument("--sharpen", type=str, default=None, metavar="AMOUNT",
                        help="Unsharp-mask sharpen after processing "
                             "(0 = none, ~0.5-1.5 typical; e.g. 1.0, or kcurve spec)")
    parser.add_argument("--sharpen-radius", type=str, default=3.0, metavar="PX",
                        help="Gaussian blur radius (sigma, px) for --sharpen (default: 3.0, or kcurve spec)")
    parser.add_argument("--orig-mix", type=str, default=None, metavar="FRAC",
                        help="Blend a fraction of the original frame back over the final result "
                             "(0.0 = none, 1.0 = pure original; e.g. 0.3, or kcurve spec). "
                             "Applied last, after all other processing.")
    parser.add_argument("--anchor", type=str, default=1.0, metavar="POS",
                        help="Window anchor position for echo effect. "
                             "0.0 = current frame is start of window, "
                             "1.0 = current frame is end (default: 1.0, or kcurve spec)")
    parser.add_argument("--reverse", action="store_true",
                        help="Blend with following frames instead of previous "
                             "(reverses input, processes, then reverses output)")
    parser.add_argument("--hflip", action="store_true",
                        help="Mirror the output horizontally (left-right), applied last")
    parser.add_argument("--no-audio", action="store_true",
                        help="Strip audio from output (no audio muxing)")
    parser.add_argument("--cores", type=int, default=4,
                        help="Number of CPU cores for parallel processing (default: 4)")
    parser.add_argument("--memory", default="auto", choices=["auto", "ram", "streaming"],
                        help="Memory mode: auto (pick based on available RAM, default), "
                             "ram (load all frames), streaming (sliding window, low memory)")
    args = parser.parse_args()

    # Parse animatable parameters (static number or kcurve spec)
    args.frames = parse_param(args.frames, int)
    args.decay = parse_param(args.decay, float)
    args.step = parse_param(args.step, int) if args.step is not None else None
    args.sigma = parse_param(args.sigma, float)
    args.pre_eq = parse_param(args.pre_eq, float)
    args.post_eq = parse_param(args.post_eq, float)
    args.gamma = parse_param(args.gamma, float)
    args.sharpen = parse_param(args.sharpen, float)
    args.sharpen_radius = parse_param(args.sharpen_radius, float)
    args.orig_mix = parse_param(args.orig_mix, float)
    args.anchor = parse_param(args.anchor, float)
    args.edge_preserve = parse_param(args.edge_preserve, float)
    args.edge_thickness = parse_param(args.edge_thickness, int)
    args.edge_gamma = parse_param(args.edge_gamma, float)

    if args.effect == "all":
        for name in EFFECTS:
            run_effect(args.input, name, args.frames, None, args.decay, args.step, args.quality, args.pre_eq, args.post_eq, args.cores, args.sigma, args.edge_preserve, args.edge_thickness, args.no_audio, args.reverse, args.anchor, args.memory, gamma=args.gamma, edge_gamma=args.edge_gamma, sharpen=args.sharpen, sharpen_radius=args.sharpen_radius, orig_mix=args.orig_mix, hflip=args.hflip)
    else:
        run_effect(args.input, args.effect, args.frames, args.output, args.decay, args.step, args.quality, args.pre_eq, args.post_eq, args.cores, args.sigma, args.edge_preserve, args.edge_thickness, args.no_audio, args.reverse, args.anchor, args.memory, gamma=args.gamma, edge_gamma=args.edge_gamma, sharpen=args.sharpen, sharpen_radius=args.sharpen_radius, orig_mix=args.orig_mix, hflip=args.hflip)
