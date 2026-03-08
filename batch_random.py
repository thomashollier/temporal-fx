#!/usr/bin/env python3
"""Generate 50 random effect videos from source files."""
import random
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SOURCE_DIR = Path("source")
OUTPUT_DIR = Path("output")

SOURCES = [str(p) for p in sorted(SOURCE_DIR.glob("*.mp4"))
           if "_flow-" not in p.name and "_echo" not in p.name
           and "_diff" not in p.name and "_decay" not in p.name]

# Effects with their parameter ranges (effect, n_range, extra_args)
EFFECT_CONFIGS = [
    ("echo",            (10, 90),   {}),
    ("slit-scan",       (40, 200),  {}),
    ("diff",            None,       {}),
    ("median",          (5, 30),    {}),
    ("decay",           None,       {"--decay": (0.80, 0.97)}),
    ("time-ramp",       (20, 120),  {}),
    ("strobe",          (10, 60),   {"--step": (2, 12)}),
    ("ping-pong",       (10, 60),   {}),
    ("rolling-shutter", (10, 60),   {}),
    ("brightest",       (15, 90),   {}),
    ("darkest",         (15, 90),   {}),
    ("temporal-gradient", (10, 60), {}),
    ("flow-dis",        None,       {}),
    ("flow-farneback",  None,       {"--quality": ["low", "medium", "high"]}),
    ("flow-raft",       None,       {}),
]

def make_job():
    """Generate one random job."""
    src = random.choice(SOURCES)
    effect, n_range, extras = random.choice(EFFECT_CONFIGS)

    args = ["python3", "temporal_fx.py", src, "-e", effect]

    n_val = None
    if n_range:
        n_val = random.randint(*n_range)
        args += ["-n", str(n_val)]

    extra_strs = []
    for flag, val_spec in extras.items():
        if isinstance(val_spec, list):
            v = random.choice(val_spec)
            args += [flag, str(v)]
            extra_strs.append(f"{flag}={v}")
        elif isinstance(val_spec, tuple):
            if isinstance(val_spec[0], float):
                v = round(random.uniform(*val_spec), 2)
            else:
                v = random.randint(*val_spec)
            args += [flag, str(v)]
            extra_strs.append(f"{flag}={v}")

    # Build output filename
    src_stem = Path(src).stem
    n_str = f"_n{n_val}" if n_val else ""
    extra_suffix = "_".join(str(v).replace("--","") for v in extra_strs).replace("=","")
    if extra_suffix:
        extra_suffix = "_" + extra_suffix
    out_name = f"{src_stem}_{effect}{n_str}{extra_suffix}.mp4"
    out_path = str(OUTPUT_DIR / out_name)
    args += ["-o", out_path]

    return args, out_name

def run_job(job_id, args, name):
    """Run a single job, return result."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            return job_id, name, "OK"
        else:
            err = r.stderr.strip().split("\n")[-1] if r.stderr else "unknown"
            return job_id, name, f"FAIL: {err}"
    except subprocess.TimeoutExpired:
        return job_id, name, "TIMEOUT"
    except Exception as e:
        return job_id, name, f"ERROR: {e}"

if __name__ == "__main__":
    random.seed(42)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate 50 unique jobs
    jobs = []
    seen_names = set()
    while len(jobs) < 50:
        args, name = make_job()
        if name not in seen_names:
            seen_names.add(name)
            jobs.append((args, name))

    print(f"Generated {len(jobs)} jobs")
    print(f"{'='*70}")

    # Separate RAFT jobs (heavy, run 1 at a time) from others (run 4 at a time)
    raft_jobs = [(i, a, n) for i, (a, n) in enumerate(jobs) if "flow-raft" in a]
    other_jobs = [(i, a, n) for i, (a, n) in enumerate(jobs) if "flow-raft" not in a]

    completed = 0
    total = len(jobs)

    # Run non-RAFT jobs with parallelism
    print(f"\nRunning {len(other_jobs)} standard jobs (4 parallel)...")
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_job, i, a, n): n for i, a, n in other_jobs}
        for fut in as_completed(futures):
            completed += 1
            job_id, name, status = fut.result()
            print(f"  [{completed:2d}/{total}] {status:6s} {name}")

    # Run RAFT jobs sequentially
    if raft_jobs:
        print(f"\nRunning {len(raft_jobs)} RAFT jobs (sequential)...")
        for i, args, name in raft_jobs:
            completed += 1
            _, _, status = run_job(i, args, name)
            print(f"  [{completed:2d}/{total}] {status:6s} {name}")

    print(f"\n{'='*70}")
    print(f"Done. {completed}/{total} jobs processed.")
    print(f"Output: {OUTPUT_DIR}/")
