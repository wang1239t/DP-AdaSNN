# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import os
import platform
import random
import time
from contextlib import contextmanager
from copy import deepcopy,copy
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, __version__
from ultralytics.utils.checks import check_version

TORCH_1_9 = check_version(torch.__version__, '1.9.0')
TORCH_2_0 = check_version(torch.__version__, '2.0.0')

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])

def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate

def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    import cpuinfo

    k = 'brand_raw', 'hardware_raw', 'arch_string_raw'
    info = cpuinfo.get_cpu_info()
    string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], 'unknown')
    return string.replace('(R)', '').replace('CPU ', '').replace('@ ', '')

def select_device(device='', batch=0, newline=False, verbose=True):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional): If True, logs the device information. Defaults to True.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> select_device('cuda:0')
        device(type='cuda', index=0)

        >>> select_device('cpu')
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """

    if isinstance(device, torch.device):
        return device

    s = f'Ultralytics YOLOv{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).lower()
    for remove in 'cuda:', 'none', '(', ')', '[', ']', "'", ' ':
        device = device.replace(remove, '')
    cpu = device == 'cpu'
    mps = device in ('mps', 'mps:0')
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        if device == 'cuda':
            device = '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))):
            LOGGER.info(s)
            install = 'See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no '\
                      'CUDA devices are seen by torch.\n' if torch.cuda.device_count() == 0 else ''
            raise ValueError(f"Invalid CUDA 'device={device}' requested."
                             f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                             f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                             f'\ntorch.cuda.is_available(): {torch.cuda.is_available()}'
                             f'\ntorch.cuda.device_count(): {torch.cuda.device_count()}'
                             f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                             f'{install}')

    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch > 0 and batch % n != 0:
            raise ValueError(f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}.")
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = 'cuda:0'
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():

        s += f'MPS ({get_cpu_info()})\n'
        arg = 'mps'
    else:
        s += f'CPU ({get_cpu_info()})\n'
        arg = 'cpu'

    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)

def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = nn.ConvTranspose2d(deconv.in_channels,
                                    deconv.out_channels,
                                    kernel_size=deconv.kernel_size,
                                    stride=deconv.stride,
                                    padding=deconv.padding,
                                    output_padding=deconv.output_padding,
                                    dilation=deconv.dilation,
                                    groups=deconv.groups,
                                    bias=True).requires_grad_(False).to(deconv.weight.device)

    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    b_conv = torch.zeros(deconv.weight.size(1), device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv

def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Model information.

    imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320].
    """
    if not verbose:
        return
    n_p = get_num_params(model)
    n_g = get_num_gradients(model)
    n_l = len(list(model.modules()))
    if detailed:
        LOGGER.info(
            f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOGGER.info('%5g %40s %9s %12g %20s %10.3g %10.3g %10s' %
                        (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std(), p.dtype))

    fused = ' (fused)' if getattr(model, 'is_fused', lambda: False)() else ''
    fs = ''
    yaml_file = getattr(model, 'yaml_file', '') or getattr(model, 'yaml', {}).get('yaml_file', '')
    model_name = Path(yaml_file).stem.replace('yolo', 'YOLO') or 'Model'
    LOGGER.info(f'{model_name} summary{fused}: {n_l} layers, {n_p} parameters, {n_g} gradients{fs}')
    return n_l, n_p, n_g, None

def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())

def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def model_info_for_loggers(trainer):
    """
    Return model info dict with useful model information.

    Example:
        YOLOv8n info for loggers
        ```python
        results = {'model/parameters': 3151904,
                   'model/GFLOPs': 8.746,
                   'model/speed_ONNX(ms)': 41.244,
                   'model/speed_TensorRT(ms)': 3.211,
                   'model/speed_PyTorch(ms)': 18.755}
        ```
    """
    return {'model/parameters': get_num_params(trainer.model)}

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales and pads an image tensor of shape img(bs,3,y,x) based on given ratio and grid size gs, optionally
    retaining the original shape.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor

def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch."""
    return max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k) - 1

def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Returns a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            LOGGER.warning('WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.')
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False

class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""

        self.ema = deepcopy(de_parallel(model)).eval()

        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)

def strip_optimizer(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in Path('path/to/weights').rglob('*.pt'):
            strip_optimizer(f)
        ```
    """
    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    if 'model' not in x:
        LOGGER.info(f'Skipping {f}, not a valid Ultralytics model.')
        return

    if hasattr(x['model'], 'args'):
        x['model'].args = dict(x['model'].args)
    args = {**DEFAULT_CFG_DICT, **x['train_args']} if 'train_args' in x else None
    if x.get('ema'):
        x['model'] = x['ema']
    for k in 'optimizer', 'scaler', 'best_fitness', 'ema', 'updates':
        x[k] = None
    x['epoch'] = -1
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}

    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")

class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement."""

    def __init__(self, patience=50):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float('inf')
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:
            return False

        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `patience=300` or use `patience=0` to disable EarlyStopping.')
        return stop
