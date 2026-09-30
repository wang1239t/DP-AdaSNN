# Ultralytics YOLO, AGPL-3.0
"""PyTorch-only backend for the public event detector."""
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):

        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(f'{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices '
                           f'{min(names.keys())}-{max(names.keys())} defined in your dataset YAML.')
    return names

class AutoBackend(nn.Module):
    def __init__(self, weights, device=torch.device('cpu'), dnn=False, data=None,
                 fp16=False, fuse=True, verbose=True):
        super().__init__()
        from ultralytics.nn.tasks import attempt_load_weights
        self.device = device
        self.nn_module = isinstance(weights, nn.Module)
        if self.nn_module:
            model = weights.to(device)
            model = model.fuse(verbose=verbose) if fuse else model
        else:
            path = Path(weights)
            if path.suffix != '.pt' or not path.is_file():
                raise FileNotFoundError(f'Expected a local PyTorch .pt checkpoint: {path}')
            model = attempt_load_weights(str(path), device=device, inplace=True, fuse=fuse)
        self.fp16 = bool(fp16 and device.type == 'cuda')
        self.model = model.half() if self.fp16 else model.float()
        self.model.eval()
        self.stride = max(int(model.stride.max()), 32)
        self.names = check_class_names(model.names)
        self.pt = True
        self.jit = self.onnx = self.xml = self.engine = self.coreml = False
        self.saved_model = self.pb = self.tflite = self.edgetpu = self.paddle = self.triton = False
        self.nhwc = False
        self.batch_size = 1
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, im, augment=False, visualize=False):
        im = im.half() if self.fp16 else im.float()
        return self.model(im, augment=augment, visualize=visualize)

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type == 'cuda':
            if len(imgsz) == 4:
                batch, _, height, width = imgsz
                core = self.model.model[0]
                imgsz = (batch, core.nb_steps, 4, height, width)
            image = torch.zeros(*imgsz, device=self.device,
                                dtype=torch.float16 if self.fp16 else torch.float32)
            with torch.no_grad():
                self.forward(image)
