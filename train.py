"""DP-AdaSNN train entrypoint (AGPL-3.0)."""
import argparse
from experiment_config import apply_runtime_env, load_experiment_cfg, yolo_kwargs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, help='Experiment YAML path')
    args = parser.parse_args()
    cfg = load_experiment_cfg(args.cfg)
    apply_runtime_env(cfg)
    from ultralytics import YOLO
    model = YOLO(cfg['model'], task='detect')
    kwargs = yolo_kwargs(cfg, exclude=('cuda_visible_devices',))
    kwargs.setdefault('pretrained', False)
    model.train(**kwargs)

if __name__ == '__main__':
    main()
