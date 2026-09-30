"""DP-AdaSNN detect entrypoint (AGPL-3.0)."""
import argparse
from pathlib import Path
from experiment_config import apply_runtime_env, load_experiment_cfg, yolo_kwargs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, help='Experiment YAML path')
    args = parser.parse_args()
    cfg = load_experiment_cfg(args.cfg)
    apply_runtime_env(cfg)
    from ultralytics import YOLO
    from event_predictor import EventDetectionPredictor
    model = YOLO(cfg['model'], task='detect')
    kwargs = yolo_kwargs(cfg, exclude=('cuda_visible_devices',))
    kwargs.pop('data', None)
    source = Path(kwargs.pop('source'))
    paths = [source] if source.is_file() else sorted(source.rglob('*.npy'))
    if not paths or any(p.suffix != '.npy' for p in paths):
        raise ValueError(f'Expected .npy event files at {source}')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Event filenames must have unique stems to avoid output collisions.')
    for result in model.predict(source=[str(path) for path in paths], predictor=EventDetectionPredictor,
                                stream=True, **kwargs):
        pass

if __name__ == '__main__':
    main()
