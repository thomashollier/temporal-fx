#!/usr/bin/env python3
"""
Frame blending script: combines N surrounding frames into each output frame.
Each source frame contributes 1/N to the blended result, creating a temporal
averaging / long-exposure effect.
"""

import cv2
import numpy as np
import argparse
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory
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


def make_weights(n, mode="uniform", sigma=None):
    """Generate blending weights for n frames.

    mode='uniform': equal weight 1/n for each frame
    mode='gaussian': gaussian bell curve centered on the middle frame
        sigma controls the spread (default: n/6, so ~3 sigma fits the window)
    """
    if mode == "uniform":
        return np.ones(n) / n

    if sigma is None:
        sigma = n / 6.0
    center = (n - 1) / 2.0
    x = np.arange(n)
    weights = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    weights /= weights.sum()
    return weights


def print_weights_preview(weights, n, mode, sigma):
    """Print an ASCII visualization of the weight distribution."""
    print(f"  Blend mode: {mode}", end="")
    if mode == "gaussian":
        print(f" (sigma={sigma:.1f})")
    else:
        print()

    # Show a compact ASCII chart
    bar_height = 8
    max_w = weights.max()
    cols = min(n, 70)
    if n > cols:
        # Downsample for display
        indices = np.linspace(0, n - 1, cols).astype(int)
        display_w = weights[indices]
    else:
        display_w = weights

    for row in range(bar_height, 0, -1):
        threshold = max_w * row / bar_height
        line = "  │"
        for w in display_w:
            line += "█" if w >= threshold else " "
        print(line + "│")
    print(f"  └{'─' * len(display_w)}┘")
    print(f"  center weight: {weights[len(weights)//2]:.4f}  "
          f"edge weight: {weights[0]:.4f}  "
          f"ratio: {weights[len(weights)//2] / weights[0]:.1f}x")


def apply_post_eq(frame, clip_limit):
    """Apply CLAHE histogram equalization to restore contrast after blending."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------- multiprocessing worker ---------------

_worker = {}


def _init_worker(in_name, out_name, shape, dtype_str, counter, n, weights_list, post_eq):
    _worker['in_shm'] = shared_memory.SharedMemory(name=in_name)
    _worker['out_shm'] = shared_memory.SharedMemory(name=out_name)
    dt = np.dtype(dtype_str)
    _worker['frames'] = np.ndarray(shape, dtype=dt, buffer=_worker['in_shm'].buf)
    _worker['output'] = np.ndarray(shape, dtype=dt, buffer=_worker['out_shm'].buf)
    _worker['counter'] = counter
    _worker['n'] = n
    _worker['weights'] = np.array(weights_list)
    _worker['post_eq'] = post_eq


def _process_frame(i):
    frames = _worker['frames']
    output = _worker['output']
    n = _worker['n']
    full_weights = _worker['weights']
    post_eq = _worker['post_eq']
    counter = _worker['counter']
    total = frames.shape[0]
    half = n // 2

    start = max(0, i - half)
    end = min(total, i - half + n)
    w_start = start - (i - half)
    w_end = w_start + (end - start)
    w = full_weights[w_start:w_end]
    w = w / w.sum()

    acc = np.zeros(frames.shape[1:], dtype=np.float64)
    for j, fi in enumerate(range(start, end)):
        acc += frames[fi].astype(np.float64) * w[j]
    blended = np.clip(acc, 0, 255).astype(np.uint8)

    if post_eq is not None:
        blended = apply_post_eq(blended, post_eq)

    output[i] = blended

    with counter.get_lock():
        counter.value += 1


# --------------- main blend function ---------------

def blend_frames(input_path, output_path, n, mode="uniform", sigma=None,
                 post_eq=None, cores=4):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if mode == "gaussian" and sigma is None:
        sigma = n / 6.0

    full_weights = make_weights(n, mode, sigma)

    print(f"Input: {input_path}")
    print(f"  {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    print(f"  Blend window: {n} frames")
    print_weights_preview(full_weights, n, mode, sigma)
    if post_eq is not None:
        print(f"  Post-EQ: CLAHE (clip_limit={post_eq:.1f})")
    print(f"  Cores: {cores}")
    print(f"Output: {output_path}")

    # Pre-read all frames into memory
    print("Reading all frames into memory...")
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)
    cap.release()
    total = len(frames_list)
    print(f"  Loaded {total} frames")

    if total == 0:
        print("Error: no frames read from input.")
        sys.exit(1)

    # Allocate shared memory for input and output frames
    shape = (total, height, width, 3)
    dtype = np.uint8
    frame_bytes = int(np.prod(shape))

    in_shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
    out_shm = shared_memory.SharedMemory(create=True, size=frame_bytes)

    try:
        # Copy frames into shared input buffer
        in_arr = np.ndarray(shape, dtype=dtype, buffer=in_shm.buf)
        for idx, f in enumerate(frames_list):
            in_arr[idx] = f
        del frames_list  # free original list

        counter = mp.Value('i', 0)
        use_cores = min(cores, total)

        print(f"Blending with {use_cores} processes...")
        t_start = time.time()

        pool = mp.Pool(
            use_cores,
            initializer=_init_worker,
            initargs=(in_shm.name, out_shm.name, shape,
                      np.dtype(dtype).str, counter, n,
                      full_weights.tolist(), post_eq),
        )

        result = pool.map_async(_process_frame, range(total))

        # Poll progress while workers run
        while not result.ready():
            progress_bar(counter.value, total, time.time() - t_start)
            time.sleep(0.1)

        result.get()  # propagate any worker exceptions
        pool.close()
        pool.join()

        progress_bar(total, total, time.time() - t_start)

        # Write output video from shared output buffer
        print("\nWriting output video...")
        out_arr = np.ndarray(shape, dtype=dtype, buffer=out_shm.buf)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for i in range(total):
            out.write(out_arr[i])
        out.release()

    finally:
        in_shm.close()
        in_shm.unlink()
        out_shm.close()
        out_shm.unlink()

    print("Done writing video.")

    # Mux audio from original if ffmpeg is available
    try:
        import subprocess
        final_path = str(Path(output_path).with_suffix("")) + "_final.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
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
        print("Re-encoding with H.264 and muxing audio...")
        subprocess.run(cmd, check=True, capture_output=True)
        Path(output_path).unlink()
        Path(final_path).rename(output_path)
        print(f"Final output: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Note: ffmpeg step skipped ({e}). Output is raw mp4v.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal frame blending")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video file")
    parser.add_argument(
        "-n", "--frames", type=int, default=30,
        help="Number of frames to blend together (default: 30)",
    )
    parser.add_argument(
        "-m", "--mode", choices=["uniform", "gaussian"], default="uniform",
        help="Blending mode: uniform (equal weight) or gaussian (bell curve)",
    )
    parser.add_argument(
        "-s", "--sigma", type=float, default=None,
        help="Gaussian sigma (default: n/6). Smaller = sharper peak, larger = flatter",
    )
    parser.add_argument(
        "--post-eq", type=float, default=None, metavar="CLIP",
        help="Apply CLAHE equalization after blending to restore contrast "
             "(clip limit, e.g. 2.0; higher = stronger)",
    )
    parser.add_argument(
        "--cores", type=int, default=4,
        help="Number of CPU cores for parallel processing (default: 4)",
    )
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        suffix = f"_blend{args.frames}"
        if args.mode == "gaussian":
            s = args.sigma if args.sigma else args.frames / 6.0
            suffix += f"_gauss{s:.0f}"
        if args.post_eq is not None:
            suffix += f"_eq{args.post_eq:.0f}"
        args.output = str(p.with_stem(p.stem + suffix))

    blend_frames(args.input, args.output, args.frames, args.mode, args.sigma,
                 args.post_eq, args.cores)
