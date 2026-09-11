# Example results

Rendered outputs preserved from the shell-script pipelines, one movie per
technique. Each entry lists the command (or pipeline) that produced it.
Sources: `selects2/` (run_tests.sh outputs), Google Drive
`RP_Projects/DST/2026-06-14/Softened_Kling4_Kling6/`,
`RP_Projects/DST/breathing01/`, and
`RP_Projects/DST/2026_04_02/DancingIllustration/`.

All clips are H.265 (hvc1) at 1080p.

## 01_trails-ahead_anchor0.mp4

Echo trails leading the subject instead of dragging behind it
(`run_breathing.sh`):

```bash
python3 temporal_fx.py in.mp4 -e echo -n 30 --anchor 0 --cores 4
```

## 02_anchor-swing_slowed.mp4

The "breathing echo" anchor swing, shown at the slowed stage (before the
final speed-up) so the swing reads clearly. A 20-second highlight from the
middle of the clip, where the trails cross over from behind to ahead. Anchor animates 1 → 0 across the
clip: trails start fully behind the dancer and end fully ahead:

```bash
python3 retime.py src.mp4 --factor 3.4 --flow dis --accel-frames 90 --decel-frames 90 -o slow.mp4
python3 temporal_fx.py slow.mp4 -e echo -n 30 --anchor "1@0L:0@<last-frame>L" --no-audio
```

## 03_eased-breathing-echo_roundtrip.mp4

The full eased pipeline (`eased_echo_pipeline.sh`): 3.4x eased slow-down
(freeze in/out over 90 source frames) → anchor-swing echo → constant 3.4x
speed-up. The eased slow-down is variable-speed, so the constant speed-up
deliberately leaves the timing breathing:

```bash
./eased_echo_pipeline.sh src.mp4
```

## 04_look-stack_echo.mp4

The standard post-processing "look" from `tests.conf` (test `echo`,
effect-only, no retiming): edge re-injection, edge gamma, unsharp mask,
10% original wash, CLAHE restore, plus the anchor swing:

```bash
python3 temporal_fx.py in.mp4 -e echo -n 30 --hflip \
  --edge-preserve 0.75 --edge-gamma 2.2 \
  --sharpen 1.5 --sharpen-radius 3 \
  --orig-mix 0.1 --post-eq 1 \
  --anchor "1@0L:0@<last-frame>L"
```

## 05_look-stack_gaussian_2x-sandwich.mp4

Same look stack on `-e gaussian` inside a 2x eased slow/speed-up sandwich
(test `gaussian` in `tests.conf`) — the direct echo-vs-gaussian A/B against
04:

```bash
python3 retime.py src.mp4 --factor 2 --flow dis --accel-frames 60 --decel-frames 60 --hflip -o slow.mp4
python3 temporal_fx.py slow.mp4 -e gaussian -n 30 \
  --edge-preserve 0.75 --edge-gamma 2.2 --sharpen 1.5 --sharpen-radius 3 \
  --orig-mix 0.1 --anchor "1@0L:0@<last-frame>L" --no-audio -o fx.mp4
python3 retime.py fx.mp4 --factor 0.5 --flow dis
```

## 06_slowecho_3.4x-eased-roundtrip.mp4

Test `slowEcho` from `tests.conf`: the deepest retime (3.4x, 90-frame eased
freezes) round-tripped, with the full look stack on the echo:

```bash
./run_tests.sh slowEcho
```

## 07_multi-param-choreography.mp4

`run_animated.sh` — four parameters keyframed against one dance clip, with a
synchronized 10-frame "hit" at frames 290–300 (pre-eq spikes to 4.0 and snaps
back while edges drop) and a collapse at 1200–1300 (window shrinks 45 → 6,
edges fade to zero):

```bash
python3 temporal_fx.py in.mov -e echo \
  -n "60@1L:24@340L:48@720L:45@1200L:6@1300L" \
  --anchor 0 --cores 4 \
  --pre-eq "1@1L:4.0@290S:1.0@300S:1@600S:5@700L" \
  --edge-preserve "1@1L:1@290S:0.3@300S:0.3@1200L:0.0@1300L" \
  --edge-thickness "1@1L:2@290S:0.3@300S:0.3@1200L:0.0@1300L"
```
