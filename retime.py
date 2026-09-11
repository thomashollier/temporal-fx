#!/usr/bin/env python3
"""Optical-flow frame interpolation for slow motion.

Synthesizes in-between frames using dense optical flow (DIS by default;
RAFT or Farneback via --flow), then writes them out at the original fps
so playback is slowed down.

A --factor of 2 (default) inserts one midpoint frame between every pair,
doubling the frame count and halving the speed (50% slow motion).

Usage:
    python3 retime.py video.mp4                  # 50% speed (factor 2)
    python3 retime.py video.mp4 --factor 4       # 25% speed
    python3 retime.py video.mp4 -o slow.mp4
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def progress_bar(current, total, elapsed, bar_width=40):
    frac = current / total if total else 1.0
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


def load_frames(input_path, limit=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if limit is not None and len(frames) >= limit:
            break
    cap.release()
    return frames, fps, width, height


class DisFlow:
    """Dense Inverse Search optical flow (fast, CPU, no extra deps)."""

    def __init__(self, h_orig, w_orig, max_dim=640):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        print("  Using DIS optical flow")

    def flow(self, a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return self.dis.calc(ga, gb, None)


# Farneback params mirror temporal_fx.py's "high" preset.
_FARNEBACK = (0.4, 7, 31, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)


class FarnebackFlow:
    """Classic Farneback dense optical flow (CPU, no extra deps)."""

    def __init__(self, h_orig, w_orig, max_dim=640):
        print("  Using Farneback optical flow")

    def flow(self, a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(ga, gb, None, *_FARNEBACK)


class RaftFlow:
    """Wraps RAFT for dense flow on frame pairs, matching temporal_fx.py setup."""

    def __init__(self, h_orig, w_orig, max_dim=640):
        try:
            import torch
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        except ImportError:
            raise SystemExit(
                "RAFT requires torch and torchvision >= 0.22.0\n"
                "  Install with: pip install torch torchvision>=0.22.0"
            )
        self.torch = torch
        self.h_orig, self.w_orig = h_orig, w_orig

        # RAFT correlation volume is O(H*W*H*W/64); downscale large frames.
        scale = min(max_dim / h_orig, max_dim / w_orig, 1.0)
        self.inf_h = (int(h_orig * scale) // 8) * 8
        self.inf_w = (int(w_orig * scale) // 8) * 8
        if scale < 1.0:
            print(f"  Downscaling {w_orig}x{h_orig} -> {self.inf_w}x{self.inf_h} for RAFT inference")

        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            try:
                t = torch.randn(1, 3, 8, 8, device="mps")
                _ = t + t
                self.device = torch.device("mps")
                print("  Using MPS device")
            except Exception:
                print("  MPS unavailable, using CPU")
        else:
            print("  Using CPU device")

        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(self.device).eval()
        self.transforms = weights.transforms()

    def _to_tensor(self, frame):
        torch = self.torch
        t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(self.inf_h, self.inf_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        return t

    def _calc(self, a, b, device):
        torch = self.torch
        a_t, b_t = self.transforms(self._to_tensor(a), self._to_tensor(b))
        with torch.no_grad():
            flow_preds = self.model(a_t.unsqueeze(0).to(device), b_t.unsqueeze(0).to(device))
        flow = flow_preds[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (inf_h, inf_w, 2)
        if self.inf_h != self.h_orig or self.inf_w != self.w_orig:
            flow = cv2.resize(flow, (self.w_orig, self.h_orig))
            flow[..., 0] *= self.w_orig / self.inf_w
            flow[..., 1] *= self.h_orig / self.inf_h
        return flow

    def flow(self, a, b):
        """Dense flow a->b, with MPS-OOM fallback to CPU."""
        torch = self.torch
        try:
            return self._calc(a, b, self.device)
        except RuntimeError as e:
            if self.device.type != "cpu" and ("out of memory" in str(e).lower() or "MPS" in str(e)):
                print("\n  MPS OOM, falling back to CPU...")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                return self._calc(a, b, self.device)
            raise


def interpolate(a, b, f_fwd, f_bwd, t):
    """Synthesize the intermediate frame at time t in (0,1) between a and b.

    Backward-warp each source toward the midpoint with cv2.remap, scaling
    the opposing flow by the temporal distance, then blend. Using both
    directions lets each source fill the other's occlusion/disocclusion holes.
    """
    h, w = a.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Sample a along backward flow scaled by t, b along forward flow scaled by (1-t).
    map_ax = xx + t * f_bwd[..., 0]
    map_ay = yy + t * f_bwd[..., 1]
    map_bx = xx + (1.0 - t) * f_fwd[..., 0]
    map_by = yy + (1.0 - t) * f_fwd[..., 1]

    warp_a = cv2.remap(a, map_ax, map_ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp_b = cv2.remap(b, map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return cv2.addWeighted(warp_a, 1.0 - t, warp_b, t, 0.0)


def source_positions(last, factor, accel=0, decel=0):
    """List of source sample positions (in [0, last]), one per output frame.

    factor>1 slows, <1 speeds up. The cruise speed is v = 1/factor source
    frames per output frame.

    With accel/decel > 0 the clip eases in from a freeze and out to a freeze.
    The velocity follows a smoothstep over the OUTPUT timeline (not source
    position) — parametrizing over source would make the zero-velocity freeze
    tails infinitely long. accel/decel are the *source* frame spans consumed
    during the ramps, so `accel=30` means "the first 30 source frames play out
    while accelerating from a standstill". Mean of smoothstep over [0,1] is
    0.5, so a ramp covering S source frames lasts 2*S/v output frames.
    """
    v = 1.0 / factor
    if accel <= 0 and decel <= 0:
        out_frames = int(round(last * factor)) + 1
        return [min(j * v, last) for j in range(out_frames)]

    cruise = last - accel - decel
    if cruise < 0:
        raise SystemExit(f"accel ({accel}) + decel ({decel}) exceed source span ({last})")

    t_a = 2.0 * accel / v       # output duration of the accel ramp
    t_b = cruise / v            # output duration of the cruise
    t_c = 2.0 * decel / v       # output duration of the decel ramp
    total = t_a + t_b + t_c
    out_frames = int(round(total)) + 1

    def antideriv(u):  # integral of smoothstep(x)=3x^2-2x^3 from 0..u
        return u ** 3 - 0.5 * u ** 4

    pos = []
    for j in range(out_frames):
        tau = j
        if tau <= t_a and t_a > 0:                       # accelerating
            s = v * t_a * antideriv(tau / t_a)
        elif tau <= t_a + t_b:                           # cruising
            s = accel + v * (tau - t_a)
        else:                                            # decelerating
            u = min((tau - t_a - t_b) / t_c, 1.0) if t_c > 0 else 1.0
            w = 1.0 - u
            s = (last - decel) + v * t_c * (0.5 - antideriv(w))
        pos.append(min(max(s, 0.0), last))
    return pos


def main():
    parser = argparse.ArgumentParser(
        description="Optical-flow slow motion via frame interpolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input video")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: <input>_slow.mp4)")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="Retime factor (may be fractional): >1 slows (2=50%% speed), <1 speeds up (0.5=200%%)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N input frames")
    parser.add_argument("--max-dim", type=int, default=640, help="Max RAFT inference dimension")
    parser.add_argument("--flow", choices=["raft", "dis", "farneback"], default="dis",
                        help="Dense flow backend (default: dis)")
    parser.add_argument("--accel-frames", type=int, default=0, metavar="N",
                        help="Ease in from a freeze over the first N source frames")
    parser.add_argument("--decel-frames", type=int, default=0, metavar="N",
                        help="Ease out to a freeze over the last N source frames")
    parser.add_argument("--hflip", action="store_true",
                        help="Horizontally flip frames before flow/interpolation")
    args = parser.parse_args()

    if args.factor <= 0:
        raise SystemExit("--factor must be > 0")

    print(f"\n{'='*60}")
    print(f"Retime — {args.flow.upper()} flow interpolation ({100 / args.factor:.0f}% speed, factor {args.factor:g})")
    print(f"Input: {args.input}")

    frames, fps, width, height = load_frames(args.input, args.limit)
    if len(frames) < 2:
        raise SystemExit("Need at least 2 frames to interpolate")
    print(f"  {width}x{height}, {fps:.2f} fps, {len(frames)} frames")

    if args.hflip:
        print("  Flipping horizontally...")
        frames = [cv2.flip(f, 1) for f in frames]

    backends = {"raft": RaftFlow, "dis": DisFlow, "farneback": FarnebackFlow}
    raft = backends[args.flow](height, width, max_dim=args.max_dim)

    output_path = args.output
    if output_path is None:
        p = Path(args.input)
        output_path = str(p.with_stem(p.stem + "_slow"))

    tmp_path = str(Path(output_path).with_suffix("")) + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    # Resample the timeline: each output frame samples a source position.
    # The integer part picks the source pair, the fraction is t. Flow is
    # computed once per pair and reused for every output frame landing in it.
    n_pairs = len(frames) - 1
    positions = source_positions(n_pairs, args.factor, args.accel_frames, args.decel_frames)
    out_frames = len(positions)
    if args.accel_frames or args.decel_frames:
        print(f"  ease-in {args.accel_frames}f / ease-out {args.decel_frames}f (source)")
    print(f"  {len(frames)} -> {out_frames} frames via {n_pairs} flow pairs...")
    t0 = time.time()

    cached_pair = -1
    f_fwd = f_bwd = None
    for j, src_t in enumerate(positions):
        i = min(int(src_t), n_pairs - 1)  # source pair index, clamped at the tail
        t = src_t - i
        if t <= 1e-6:
            writer.write(frames[i])
        else:
            if cached_pair != i:
                a, b = frames[i], frames[i + 1]
                f_fwd = raft.flow(a, b)
                f_bwd = raft.flow(b, a)
                cached_pair = i
            writer.write(interpolate(frames[i], frames[i + 1], f_fwd, f_bwd, t))
        progress_bar(j + 1, out_frames, time.time() - t0)
    print()
    writer.release()

    print(f"  Wrote {out_frames} frames at {fps:.2f} fps")

    # Re-encode H.264. Audio is dropped — retiming desyncs the original track.
    try:
        final_path = str(Path(output_path).with_suffix("")) + "_final.mp4"
        cmd = ["ffmpeg", "-y", "-i", tmp_path,
               "-c:v", "libx264", "-crf", "18", "-preset", "medium",
               "-an", final_path]
        print("  Re-encoding with H.264 (no audio)...")
        subprocess.run(cmd, check=True, capture_output=True)
        Path(tmp_path).unlink()
        Path(final_path).rename(output_path)
        print(f"  Final output: {output_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Note: ffmpeg step skipped ({e}). Output is raw mp4v.")
        Path(tmp_path).rename(output_path)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
