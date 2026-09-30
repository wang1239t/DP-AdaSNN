# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from matplotlib import font_manager

from ultralytics.utils import (ASSETS, AUTOINSTALL, LINUX, LOGGER, ONLINE, ROOT, USER_CONFIG_DIR, SimpleNamespace,
                               ThreadingLocked, TryExcept, clean_url, colorstr, downloads, emojis, is_colab, is_docker,
                               is_jupyter, is_kaggle, is_online, is_pip_package, url2file)

def parse_requirements(file_path=ROOT.parent / 'requirements.txt', package=''):
    """
    Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file, i.e. package='ultralytics'.

    Returns:
        (List[Dict[str, str]]): List of parsed requirements as dictionaries with `name` and `specifier` keys.

    Example:
        ```python
        from ultralytics.utils.checks import parse_requirements

        parse_requirements(package='ultralytics')
        ```
    """

    if package:
        requires = [x for x in metadata.distribution(package).requires if 'extra == ' not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith('#'):
            line = line.split('#')[0].strip()
            match = re.match(r'([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?', line)
            if match:
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ''))

    return requirements

def parse_version(version='0.0.0') -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r'\d+', version)[:3]))
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}')
        return 0, 0, 0

def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        bool: True if the string is composed only of ASCII characters, False otherwise.
    """

    s = str(s)

    return all(ord(c) < 128 for c in s)

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int]): Updated image size.
    """

    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    else:
        raise TypeError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
                        f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'")

    if len(imgsz) > max_dim:
        msg = "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "\
              "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        if max_dim != 1:
            raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]

    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    if sz != imgsz:
        LOGGER.warning(f'WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}')

    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz

def check_version(current: str = '0.0.0',
                  required: str = '0.0.0',
                  name: str = 'version',
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
        ```
    """
    if not current:
        LOGGER.warning(f'WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.')
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError:
            if hard:
                raise ModuleNotFoundError(emojis(f'WARNING ⚠️ {current} package is required but not installed'))
            else:
                return False

    if not required:
        return True

    result = True
    c = parse_version(current)
    for r in required.strip(',').split(','):
        op, v = re.match(r'([^0-9]*)([\d.]+)', r).groups()
        v = parse_version(v)
        if op == '==' and c != v:
            result = False
        elif op == '!=' and c == v:
            result = False
        elif op in ('>=', '') and not (c >= v):
            result = False
        elif op == '<=' and not (c <= v):
            result = False
        elif op == '>' and not (c > v):
            result = False
        elif op == '<' and not (c < v):
            result = False
    if not result:
        warning_message = f'WARNING ⚠️ {name}{op}{required} is required, but {name}=={current} is currently installed'
        if hard:
            raise ModuleNotFoundError(emojis(warning_message))
        if verbose:
            LOGGER.warning(warning_message)
    return result

def check_latest_pypi_version(package_name='ultralytics'):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str): The latest version of the package.
    """
    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings()
        response = requests.get(f'https://pypi.org/pypi/{package_name}/json', timeout=3)
        if response.status_code == 200:
            return response.json()['info']['version']

def check_pip_update_available():
    """
    Checks if a new version of the ultralytics package is available on PyPI.

    Returns:
        (bool): True if an update is available, False otherwise.
    """
    if ONLINE and is_pip_package():
        with contextlib.suppress(Exception):
            from ultralytics import __version__
            latest = check_latest_pypi_version()
            if check_version(__version__, f'<{latest}'):
                LOGGER.info(f'New https://pypi.org/project/ultralytics/{latest} available 😃 '
                            f"Update with 'pip install -U ultralytics'")
                return True
    return False

@ThreadingLocked()
def check_font(font='Arial.ttf'):
    """
    Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        file (Path): Resolved font file path.
    """
    name = Path(font).name

    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    url = f'https://ultralytics.com/assets/{name}'
    if downloads.is_url(url):
        downloads.safe_download(url=url, file=file)
        return file

def check_python(minimum: str = '3.8.0') -> bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.

    Returns:
        None
    """
    return check_version(platform.python_version(), minimum, name='Python ', hard=True)

@TryExcept()
def check_requirements(requirements=ROOT.parent / 'requirements.txt', exclude=(), install=True, cmds=''):
    """
    Check if installed dependencies meet YOLOv8 requirements and attempt to auto-update if needed.

    Args:
        requirements (Union[Path, str, List[str]]): Path to a requirements.txt file, a single package requirement as a
            string, or a list of package requirements as strings.
        exclude (Tuple[str]): Tuple of package names to exclude from checking.
        install (bool): If True, attempt to auto-update packages that don't meet requirements.
        cmds (str): Additional commands to pass to the pip install command when auto-updating.

    Example:
        ```python
        from ultralytics.utils.checks import check_requirements

        # Check a requirements.txt file
        check_requirements('path/to/requirements.txt')

        # Check a single package
        check_requirements('ultralytics>=8.0.0')

        # Check multiple packages
        check_requirements(['numpy', 'ultralytics>=8.0.0'])
        ```
    """

    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()
    check_torchvision()
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f'{prefix} {file} not found, check failed.'
        requirements = [f'{x.name}{x.specifier}' for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split('/')[-1].replace('.git', '')
        match = re.match(r'([a-zA-Z0-9-_]+)([<>!=~]+.*)?', r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ''
        try:
            assert check_version(metadata.version(name), required)
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    s = ' '.join(f'"{x}"' for x in pkgs)
    if s:
        if install and AUTOINSTALL:
            n = len(pkgs)
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert is_online(), 'AutoUpdate skipped (offline)'
                LOGGER.info(subprocess.check_output(f'pip install --no-cache {s} {cmds}', shell=True).decode())
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n")
            except Exception as e:
                LOGGER.warning(f'{prefix} ❌ {e}')
                return False
        else:
            return False

    return True

def check_torchvision():
    """
    Checks the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the provided compatibility table based on:
    https://github.com/pytorch/vision#installation.

    The compatibility table is a dictionary where the keys are PyTorch versions and the values are lists of compatible
    Torchvision versions.
    """

    import torchvision

    compatibility_table = {'2.0': ['0.15'], '1.13': ['0.14'], '1.12': ['0.13']}

    v_torch = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
    v_torchvision = '.'.join(torchvision.__version__.split('+')[0].split('.')[:2])

    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        if all(v_torchvision != v for v in compatible_versions):
            print(f'WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n'
                  f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                  "'pip install -U torch torchvision' to update both.\n"
                  'For a full compatibility table see https://github.com/pytorch/vision#installation')

def check_suffix(file='yolov8n.pt', suffix='.pt', msg=''):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'

def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if 'yolov3' in file or 'yolov5' in file:
        if 'u.yaml' in file:
            file = file.replace('u.yaml', '.yaml')
        elif '.pt' in file and 'u' not in file:
            original_file = file
            file = re.sub(r'(.*yolov5([nsmlx]))\.pt', '\\1u.pt', file)
            file = re.sub(r'(.*yolov5([nsmlx])6)\.pt', '\\1u.pt', file)
            file = re.sub(r'(.*yolov3(|-tiny|-spp))\.pt', '\\1u.pt', file)
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f'trained with https://github.com/ultralytics/ultralytics and feature improved performance vs '
                    f'standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n')
    return file

def check_file(file, suffix='', download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)
    file = str(file).strip()
    file = check_yolov5u_filename(file)
    if not file or ('://' not in file and Path(file).exists()):
        return file
    elif download and file.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://', 'tcp://')):
        url = file
        file = url2file(file)
        if Path(file).exists():
            LOGGER.info(f'Found {clean_url(url)} locally at {file}')
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:
        files = glob.glob(str(ROOT / 'cfg' / '**' / file), recursive=True)
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []

def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)

def check_imshow(warn=False):
    """Check if environment supports image displays."""
    try:
        if LINUX:
            assert 'DISPLAY' in os.environ and not is_docker() and not is_colab() and not is_kaggle()
        cv2.imshow('test', np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False

def check_yolo(verbose=True, device=''):
    """Return a human-readable YOLO software and hardware summary."""
    import psutil

    from ultralytics.utils.torch_utils import select_device

    if is_jupyter():
        if check_requirements('wandb', install=False):
            os.system('pip uninstall -y wandb')
        if is_colab():
            shutil.rmtree('sample_data', ignore_errors=True)

    if verbose:

        gib = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        s = f'({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)'
        with contextlib.suppress(Exception):
            from IPython import display
            display.clear_output()
    else:
        s = ''

    select_device(device=device, newline=False)
    LOGGER.info(f'Setup complete ✅ {s}')

def collect_system_info():
    """Collect and print relevant system information including OS, Python, RAM, CPU, and CUDA."""

    import psutil

    from ultralytics.utils import ENVIRONMENT, is_git_dir
    from ultralytics.utils.torch_utils import get_cpu_info

    ram_info = psutil.virtual_memory().total / (1024 ** 3)
    check_yolo()
    LOGGER.info(f"\n{'OS':<20}{platform.platform()}\n"
                f"{'Environment':<20}{ENVIRONMENT}\n"
                f"{'Python':<20}{sys.version.split()[0]}\n"
                f"{'Install':<20}{'git' if is_git_dir() else 'pip' if is_pip_package() else 'other'}\n"
                f"{'RAM':<20}{ram_info:.2f} GB\n"
                f"{'CPU':<20}{get_cpu_info()}\n"
                f"{'CUDA':<20}{torch.version.cuda if torch and torch.cuda.is_available() else None}\n")

    for r in parse_requirements(package='ultralytics'):
        try:
            current = metadata.version(r.name)
            is_met = '✅ ' if check_version(current, str(r.specifier), hard=True) else '❌ '
        except metadata.PackageNotFoundError:
            current = '(not installed)'
            is_met = '❌ '
        LOGGER.info(f'{r.name:<20}{is_met}{current}{r.specifier}')

def git_describe(path=ROOT):
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    with contextlib.suppress(Exception):
        return subprocess.check_output(f'git -C {path} describe --tags --long --always', shell=True).decode()[:-1]
    return ''

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""

    def strip_auth(v):
        """Clean longer Ultralytics HUB URLs by stripping potential authentication information."""
        return clean_url(v) if (isinstance(v, str) and v.startswith('http') and len(v) > 100) else v

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={strip_auth(v)}' for k, v in args.items()))

def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    try:

        output = subprocess.check_output(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                                         encoding='utf-8')

        first_line = output.strip().split('\n')[0]

        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):

        return 0

def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    return cuda_device_count() > 0

def check_amp(model):
    """Check finite FP32/AMP event outputs using the configured local detector."""
    device = next(model.parameters()).device
    if device.type != 'cuda':
        return False
    training = model.training
    # This bounded probe does not download another model or consume training RNG.
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(0)
            core = model.model[0]
            events = torch.rand(1, core.nb_steps, 4, 64, 96, device=device)
            events[:, :, :2] *= 12
            model.eval()
            with torch.no_grad():
                full = model(events)[0]
                with torch.amp.autocast('cuda'):
                    mixed = model(events)[0]
            passed = (torch.isfinite(full).all() and torch.isfinite(mixed).all()
                      and full.shape == mixed.shape
                      and torch.allclose(full, mixed.float(), atol=0.5, rtol=1e-3))
            LOGGER.info(f'AMP event check: {"passed" if passed else "failed; using FP32"}')
            return bool(passed)
    finally:
        model.train(training)
