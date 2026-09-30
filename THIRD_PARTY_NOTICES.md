# Third-party notices

This distribution uses AGPL-3.0; see [LICENSE](LICENSE). Existing source notices are retained.

- **Ultralytics YOLO 8.0.197**: the included `ultralytics/` framework is a modified copy. Source: https://github.com/ultralytics/ultralytics. Its AGPL-3.0 license is included at the repository root.
- **SpikeYOLO**: the spiking backbone and head are retained from the source project used by this implementation. Source: https://github.com/BICLab/SpikeYOLO. Existing module implementation and attribution are preserved.
- **ASGL-SNN**: `ultralytics/nn/modules/activation.py` retains its original attribution to Ziming Wang and https://github.com/Windere/ASGL-SNN.
- **SpikingJelly** is an external dependency, not bundled source. Source: https://github.com/fangwei123456/spikingjelly. Its installed package supplies its own license notices.
- PyTorch, torchvision, NumPy, OpenCV, Matplotlib and other external dependencies remain governed by their respective licenses; this repository does not relicense them.

Release modifications remove research monitoring, auditing, extra profilers, unrelated dataset extensions and external logging integrations. Event inference uses a small adapter that loads configured event tensors and renders ordinary bounding boxes. Model construction honors the caller's device, and local full-model checkpoint loading explicitly supports PyTorch's `weights_only` option.
