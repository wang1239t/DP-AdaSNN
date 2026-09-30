# DP-AdaSNN

Core implementation of DP-AdaSNN with adaptive thresholds, the original time-surface-driven (TSD) loss, and the SpikeYOLO-P2 detector. Training, held-out testing and event-file detection are supported for GEN1 and DSEC.

[中文说明](README.zh-CN.md) · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Installation

Use Python 3.11 and a matching PyTorch/torchvision build for your CPU or CUDA platform. The release was checked with PyTorch 2.7.0, torchvision 0.22.0 and SpikingJelly 0.0.0.0.14.

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. The modified `ultralytics/` package is included here; do not install the PyPI Ultralytics package as a replacement. Dependencies are not installed automatically by the entrypoints. Full-model `.pt` files use Python deserialization: use checkpoints you trust, such as those you trained locally.

## Data

Obtain datasets separately: [GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) and [DSEC](https://dsec.ifi.uzh.ch/).
This release consumes already prepared per-window event files and YOLO detection labels; it does not redistribute raw datasets or perform raw-dataset conversion.

```text
data/gen1/                   # or data/dsec/
  train/events/*.npy
  train/labels/*.txt
  val/events/*.npy
  val/labels/*.txt
  test/events/*.npy
  test/labels/*.txt
```

Each event file is an `N×4` NumPy array with `[x, y, timestamp, polarity]` columns, increasing timestamps in microseconds and polarity 0/1. The configured column order converts this to `[x,y,polarity,timestamp]`. Labels contain `class_id center_x center_y width height` with normalized coordinates. GEN1 classes are car/pedestrian; DSEC has the seven classes listed in its dataset YAML. Keep the original train/validation/test split.

Set an **absolute** dataset root in `ultralytics/cfg/datasets/GEN1.yaml` or `DSEC.yaml`. The supplied `data/...` value is a placeholder: resolve it to your actual prepared dataset. Provide locally trained checkpoints through the `model` field of test/detect YAMLs. Set `source` in detection YAMLs to a `.npy` file or event directory; files in a directory must have unique filename stems.

## Run

```bash
python train.py --cfg ultralytics/cfg/experiments/gen1_train.yaml
python test.py --cfg ultralytics/cfg/experiments/gen1_test.yaml
python detect.py --cfg ultralytics/cfg/experiments/gen1_detect.yaml
```

Replace `gen1` with `dsec` for DSEC. Test defaults to `split: test`; training still uses validation. Detection streams files and saves ordinary prediction images and optional YOLO text labels. Rendering uses input event counts only and does not affect the detector.

Training templates preserve the two-GPU batch configurations (GEN1 16, DSEC 8). Adapt `cuda_visible_devices`, `device`, paths and output names to your machine; keep the scientific settings fixed when reproducing the supplied protocol. Explicit positive batch sizes are required. Set `device: cpu` and `cuda_visible_devices: null` for CPU checks. Existing output directories are incremented by Ultralytics rather than overwritten.

## Frozen method settings

| Setting | GEN1 | DSEC |
|---|---|---|
| Detector scale | s, P2/P3/P4/P5 | s, P2/P3/P4/P5 |
| Event size | 240×304, padded/resized to 320 | 480×640, no padding |
| Input steps / frontend slots | 6 / 4 | 4 / 3 |
| Count cap | 512 | None |
| Time-surface tau | 50,000 µs | 18,000 µs |
| Reset / output clamp | 0 / 24 | 0 / 24 |
| SAT / log compression | Off / Off | Off / Off |
| AdaVth / original TSD | On / On | On / On |

The frontend, original TSD formulas and configurations were extracted from frozen research snapshots. Loss gains, temporal reversal and event resizing remain part of the configuration. No benchmark claims are made from synthetic checks.

## Release boundary

Included: the core method, required detection framework, GEN1/DSEC event representation and loading, six entrypoint configuration templates, standard losses and P/R/mAP evaluation.

Excluded: internal-state/firing-rate hooks, physics and energy reports, efficiency profilers, audit capture, frontend feature visualizations, external logging callbacks, research variants, other datasets, experiment management, trained weights and historical results.

The configuration and class names needed to use the core remain available. Private monitoring APIs and diagnostic recording arguments are not part of this release. Historical full-object research checkpoints that contain removed classes are not promised to load; train within this release or transfer compatible state dictionaries explicitly.

## Small regression checks

```bash
python -m unittest discover -s tests -v
```

Windows entrypoints retain the original process-level OpenMP compatibility setting; this does not modify the installed environment. GPU warmup uses five-dimensional event tensors, and AMP checks use the local model with small synthetic inputs.

Acceptance includes CPU synthetic train/test/detect, exact comparisons with frozen sources, and single-GPU inference/AMP interfaces at 64×96. CUDA training, multi-GPU training and full real-dataset results remain unverified.
