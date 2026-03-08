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
    _PARALLEL_WORKERS[_par['effect']](i)
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
    start = max(0, i - n + 1)
    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    for fi in range(start, i + 1):
        acc += frames[fi].astype(np.float64)
    acc /= (i + 1 - start)
    output[i] = np.clip(acc, 0, 255).astype(np.uint8)


def _pw_gaussian(i):
    frames, output, n = _par['frames'], _par['output'], _par['n']
    total = frames.shape[0]
    weights = np.array(_par['extra']['weights'])
    half = n // 2
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


def write_and_mux(out_frames, fps, width, height, output_path, input_path):
    """Write frames to mp4v then re-encode with H.264 and mux audio."""
    tmp_path = str(Path(output_path).with_suffix("")) + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    for f in out_frames:
        out.write(f)
    out.release()

    try:
        final_path = str(Path(output_path).with_suffix("")) + "_final.mp4"
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

def fx_echo(frames, n, cores=1, **kw):
    """Blend only previous N frames (motion trails)."""
    if cores > 1:
        return _parallel_effect(frames, n, "echo", cores)
    total = len(frames)
    result = []
    t0 = time.time()
    for i in range(total):
        start = max(0, i - n + 1)
        window = frames[start:i + 1]
        acc = np.zeros_like(frames[0], dtype=np.float64)
        for f in window:
            acc += f.astype(np.float64)
        acc /= len(window)
        result.append(np.clip(acc, 0, 255).astype(np.uint8))
        progress_bar(i + 1, total, time.time() - t0)
    print()
    return result


def fx_gaussian(frames, n, cores=1, sigma=None, **kw):
    """Gaussian-weighted blend across N frames — bell curve falloff from center."""
    if sigma is None:
        sigma = n / 4.0
    if cores > 1:
        half = n // 2
        full_size = 2 * half + 1
        kernel = np.exp(-0.5 * (np.arange(full_size) - half) ** 2 / (sigma ** 2))
        return _parallel_effect(frames, n, "gaussian", cores,
                                extra={'weights': kernel.tolist()})
    total = len(frames)
    result = []
    t0 = time.time()
    half = n // 2
    full_size = 2 * half + 1
    kernel = np.exp(-0.5 * (np.arange(full_size) - half) ** 2 / (sigma ** 2))
    for i in range(total):
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
        out = np.empty_like(frames[0])
        for row in range(height):
            # Map each row to a frame offset within [-n//2, n//2]
            offset = int((row / height - 0.5) * n)
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
    half = n // 2
    for i in range(total):
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
        acc = decay * acc + (1.0 - decay) * frames[i].astype(np.float64)
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
        # Window size ramps from 1 at frame 0 to n at last frame
        win = max(1, int(1 + (n - 1) * i / max(1, total - 1)))
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
        indices = list(range(i, max(i - n * step, -1), -step))
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
    half = n // 2
    for i in range(total):
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
        out = np.empty_like(frames[0])
        for row in range(height):
            # Spread n frames across the image height
            offset = int(row / height * n)
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
    half = n // 2
    running_sum = None
    prev_start, prev_end = 0, 0
    for i in range(total):
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
    half = n // 2
    running_sum = None
    prev_start, prev_end = 0, 0
    for i in range(total):
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
    half = n // 2
    for i in range(total):
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
        start = max(0, i - n + 1)
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
        start = max(0, i - n + 1)
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


def apply_post_eq(frame, clip_limit):
    """Apply CLAHE histogram equalization to restore contrast after processing."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_effect(input_path, effect_name, n_override, output_path, decay, step, quality="low", pre_eq=None, post_eq=None, cores=1, sigma=None):
    fn, default_n, desc = EFFECTS[effect_name]
    n = n_override if n_override is not None else default_n

    print(f"\n{'='*60}")
    print(f"Effect: {effect_name} — {desc}")
    if n is not None:
        print(f"  n = {n}")
    if pre_eq is not None:
        print(f"  pre-eq = on (CLAHE clip_limit={pre_eq:.1f})")
    if post_eq is not None:
        print(f"  post-eq = on (CLAHE clip_limit={post_eq:.1f})")
    if effect_name == "decay":
        print(f"  decay = {decay}")
    if effect_name == "strobe":
        print(f"  step = {step}")
    if sigma is not None and effect_name == "gaussian":
        print(f"  sigma = {sigma}")
    if cores > 1:
        print(f"  cores = {cores}")
    print(f"Input: {input_path}")

    frames, fps, width, height = load_frames(input_path)

    if pre_eq is not None:
        print(f"  Equalizing {len(frames)} input frames (clip_limit={pre_eq:.1f})...")
        if cores > 1:
            with mp.Pool(min(cores, len(frames))) as pool:
                frames = pool.starmap(apply_post_eq, [(f, pre_eq) for f in frames])
        else:
            frames = [apply_post_eq(f, pre_eq) for f in frames]

    print(f"  Processing {effect_name}...")
    out_frames = fn(frames, n=n, decay=decay, step=step, quality=quality, cores=cores, sigma=sigma)

    if post_eq is not None:
        print(f"  Applying post-EQ (CLAHE clip_limit={post_eq:.1f})...")
        if cores > 1:
            with mp.Pool(min(cores, len(out_frames))) as pool:
                out_frames = pool.starmap(apply_post_eq, [(f, post_eq) for f in out_frames])
        else:
            out_frames = [apply_post_eq(f, post_eq) for f in out_frames]

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + f"_{effect_name}"))

    write_and_mux(out_frames, fps, width, height, output_path, input_path)
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
  --step  INT         Step size for 'strobe' effect (default: 4)
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

Parallelism:
  --cores N           Number of CPU cores for parallel processing (default: 4)
                      Most effects support shared-memory multicore processing.
                      Not parallelized: decay, feedback (sequential accumulation),
                      flow-dis, flow-farneback, flow-raft, motion-streak (optical flow),
                      diff, bitwise-xor (trivially fast consecutive-frame ops).
                      pre-eq and post-eq are also parallelized.

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
""",
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument(
        "--effect", "-e", required=True,
        choices=list(EFFECTS.keys()) + ["all"],
        help="Effect to apply (or 'all' to run every effect)",
    )
    parser.add_argument("-n", "--frames", type=int, default=None,
                        help="Number of frames for temporal window")
    parser.add_argument("--decay", type=float, default=0.92,
                        help="Decay factor for 'decay' effect (default: 0.92)")
    parser.add_argument("--step", type=int, default=4,
                        help="Step size for 'strobe' effect (default: 4)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output video file path")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian sigma for 'gaussian' effect (default: n/4). "
                             "Smaller = sharper peak, larger = flatter")
    parser.add_argument("--quality", "-q", default="low",
                        choices=["low", "medium", "high"],
                        help="Quality preset for flow-farneback (default: low)")
    parser.add_argument("--pre-eq", type=float, default=None, metavar="CLIP",
                        help="Apply CLAHE equalization to input frames before processing "
                             "(clip limit, e.g. 2.0; higher = stronger)")
    parser.add_argument("--post-eq", type=float, default=None, metavar="CLIP",
                        help="Apply CLAHE equalization after processing to restore contrast "
                             "(clip limit, e.g. 2.0; higher = stronger)")
    parser.add_argument("--cores", type=int, default=4,
                        help="Number of CPU cores for parallel processing (default: 4)")
    args = parser.parse_args()

    if args.effect == "all":
        for name in EFFECTS:
            run_effect(args.input, name, args.frames, None, args.decay, args.step, args.quality, args.pre_eq, args.post_eq, args.cores, args.sigma)
    else:
        run_effect(args.input, args.effect, args.frames, args.output, args.decay, args.step, args.quality, args.pre_eq, args.post_eq, args.cores, args.sigma)
