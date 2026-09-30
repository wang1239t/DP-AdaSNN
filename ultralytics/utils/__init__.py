# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import urllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

from ultralytics import __version__

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
ASSETS = ROOT / 'assets'
DEFAULT_CFG_PATH = ROOT / 'cfg/default.yaml'
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' if VERBOSE else None
LOGGING_NAME = 'ultralytics'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])
ARM64 = platform.machine() in ('arm64', 'aarch64')
HELP_MSG = 'DP-AdaSNN: run python train.py, test.py, or detect.py --cfg <GEN1/DSEC YAML>. See README.md.'

torch.set_printoptions(linewidth=320, precision=4, profile='default')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TQDM(tqdm_original):
    """
    Custom Ultralytics tqdm class with different default arguments.

    Args:
        *args (list): Positional arguments passed to original tqdm.
        **kwargs (dict): Keyword arguments, with custom defaults applied.
    """

    def __init__(self, *args, **kwargs):
        """Initialize custom Ultralytics tqdm class with different default arguments."""

        kwargs['disable'] = not VERBOSE or kwargs.get('disable', False)
        kwargs.setdefault('bar_format', TQDM_BAR_FORMAT)
        super().__init__(*args, **kwargs)

class SimpleClass:
    """Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, SimpleClass):

                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

class IterableSimpleNamespace(SimpleNamespace):
    """Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """)

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)

def plt_settings(rcparams=None, backend='Agg'):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {'font.size': 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            if backend != original_backend:
                plt.close('all')
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close('all')
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator

def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})

def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string

class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.

    This filter is particularly useful for ensuring compatibility with Windows terminals that may not support the
    display of emojis in log messages.
    """

    def filter(self, record):
        """Filter logs by emoji unicode characters on windows."""
        record.msg = emojis(record.msg)
        return super().filter(record)

set_logging(LOGGING_NAME, verbose=VERBOSE)
LOGGER = logging.getLogger(LOGGING_NAME)
if WINDOWS:
    LOGGER.addFilter(EmojiFilter())

class ThreadingLocked:
    """
    A decorator class for ensuring thread-safe execution of a function or method. This class can be used as a decorator
    to make sure that if the decorated function is called from multiple threads, only one thread at a time will be able
    to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Example:
        ```python
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # Your code here
            pass
        ```
    """

    def __init__(self):
        """Initializes the decorator class for thread-safe execution of a function or method."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Applies thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)

        return decorated

def yaml_save(file='data.yaml', data=None, header=''):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():

        file.parent.mkdir(parents=True, exist_ok=True)

    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    with open(file, 'w', errors='ignore', encoding='utf-8') as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()

        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        data = yaml.safe_load(s) or {}
        if append_filename:
            data['yaml_file'] = str(file)
        return data

def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a YAML file or a YAML-formatted dictionary.

    Args:
        yaml_file: The file path of the YAML file or a YAML-formatted dictionary.

    Returns:
        None
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")

DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)

def is_ubuntu() -> bool:
    """
    Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    with contextlib.suppress(FileNotFoundError):
        with open('/etc/os-release') as f:
            return 'ID=ubuntu' in f.read()
    return False

def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ

def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'

def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook. Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False

def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    file = Path('/proc/self/cgroup')
    if file.exists():
        with open(file) as f:
            return 'docker' in f.read()
    else:
        return False

def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    import socket

    for host in '1.1.1.1', '8.8.8.8', '223.5.5.5':
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:

            test_connection.close()
            return True
    return False

ONLINE = is_online()

def is_pip_package(filepath: str = __name__) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    spec = importlib.util.find_spec(filepath)

    return spec is not None and spec.origin is not None

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)

def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ('PYTEST_CURRENT_TEST' in os.environ) or ('pytest' in sys.modules) or ('pytest' in Path(sys.argv[0]).stem)

def is_github_actions_ci() -> bool:
    """
    Determine if the current environment is a GitHub Actions CI Python runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions CI Python runner, False otherwise.
    """
    return 'GITHUB_ACTIONS' in os.environ and 'RUNNER_OS' in os.environ and 'RUNNER_TOOL_CACHE' in os.environ

def is_git_dir():
    """
    Determines whether the current file is part of a git repository. If the current file is not part of a git
    repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return get_git_dir() is not None

def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d

def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str | None): The origin URL of the git repository or None if not git directory.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'])
            return origin.decode().strip()

def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name or None if not a git directory.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            return origin.decode().strip()

def get_default_args(func):
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

def get_ubuntu_version():
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        with contextlib.suppress(FileNotFoundError, AttributeError):
            with open('/etc/os-release') as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]

def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """

    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')

    if not is_dir_writeable(path.parent):
        LOGGER.warning(f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
                       'Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path.')
        path = Path('/tmp') / sub_dir if is_dir_writeable('/tmp') else Path().cwd() / sub_dir

    path.mkdir(parents=True, exist_ok=True)

    return path

USER_CONFIG_DIR = Path(os.getenv('YOLO_CONFIG_DIR') or get_user_config_dir())
SETTINGS_YAML = USER_CONFIG_DIR / 'settings.yaml'

def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr('blue', 'bold', 'hello world')
        >>> '\033[34m\033[1mhello world\033[0m'
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def remove_colorstr(input_string):
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr('blue', 'bold', 'hello world'))
        >>> 'hello world'
    """
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\\-_]|\[[0-9]*[ -/]*[@-~])')
    return ansi_escape.sub('', input_string)

class TryExcept(contextlib.ContextDecorator):
    """
    YOLOv8 TryExcept class.

    Use as @TryExcept() decorator or 'with TryExcept():' context manager.
    """

    def __init__(self, msg='', verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

def threaded(func):
    """
    Multi-threads a target function and returns thread.

    Use as @threaded decorator.
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function and returns the thread."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper

class SettingsManager(dict):
    """
    Manages Ultralytics settings stored in a YAML file.

    Args:
        file (str | Path): Path to the Ultralytics settings YAML file. Default is USER_CONFIG_DIR / 'settings.yaml'.
        version (str): Settings version. In case of local version mismatch, new default settings will be saved.
    """

    def __init__(self, file=SETTINGS_YAML, version='0.0.4'):
        """Initialize the SettingsManager with default settings, load and validate current settings from the YAML
        file.
        """
        import copy
        import hashlib

        from ultralytics.utils.checks import check_version
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        git_dir = get_git_dir()
        root = git_dir or Path()
        datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()

        self.file = Path(file)
        self.version = version
        self.defaults = {
            'settings_version': version,
            'datasets_dir': str(datasets_root / 'datasets'),
            'weights_dir': str(root / 'weights'),
            'runs_dir': str(root / 'runs'),
            'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),
            'sync': True,
            'api_key': '',
            'clearml': True,
            'comet': True,
            'dvc': True,
            'hub': True,
            'mlflow': True,
            'neptune': True,
            'raytune': True,
            'tensorboard': True,
            'wandb': True}

        super().__init__(copy.deepcopy(self.defaults))

        with torch_distributed_zero_first(RANK):
            if not self.file.exists():
                self.save()

            self.load()
            correct_keys = self.keys() == self.defaults.keys()
            correct_types = all(type(a) is type(b) for a, b in zip(self.values(), self.defaults.values()))
            correct_version = check_version(self['settings_version'], self.version)
            if not (correct_keys and correct_types and correct_version):
                LOGGER.warning(
                    'WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem '
                    'with your settings or a recent ultralytics package update. '
                    f"\nView settings with 'yolo settings' or at '{self.file}'"
                    "\nUpdate settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'.")
                self.reset()

    def load(self):
        """Loads settings from the YAML file."""
        super().update(yaml_load(self.file))

    def save(self):
        """Saves the current settings to the YAML file."""
        yaml_save(self.file, dict(self))

    def update(self, *args, **kwargs):
        """Updates a setting value in the current settings."""
        super().update(*args, **kwargs)
        self.save()

    def reset(self):
        """Resets the settings to default and saves them."""
        self.clear()
        self.update(self.defaults)
        self.save()

def deprecation_warn(arg, new_arg, version=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    if not version:
        version = float(__version__[:3]) + 0.2
    LOGGER.warning(f"WARNING ⚠️ '{arg}' is deprecated and will be removed in 'ultralytics {version}' in the future. "
                   f"Please use '{new_arg}' instead.")

def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(':/', '://')
    return urllib.parse.unquote(url).split('?')[0]

def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name

PREFIX = colorstr('Ultralytics: ')
SETTINGS = SettingsManager()
DATASETS_DIR = Path(SETTINGS['datasets_dir'])
WEIGHTS_DIR = Path(SETTINGS['weights_dir'])
ENVIRONMENT = 'Colab' if is_colab() else 'Kaggle' if is_kaggle() else 'Jupyter' if is_jupyter() else\
    'Docker' if is_docker() else platform.system()
TESTS_RUNNING = is_pytest_running() or is_github_actions_ci()

from .patches import imread, imshow, imwrite, torch_save

torch.save = torch_save
if WINDOWS:

    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
