# Ultralytics YOLO, AGPL-3.0
import contextlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union
from ultralytics.utils import ASSETS, DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_PATH, LOGGER, RANK, ROOT, SETTINGS, SETTINGS_YAML, TESTS_RUNNING, IterableSimpleNamespace, __version__, checks, colorstr, deprecation_warn, yaml_load, yaml_print
MODES = ('train', 'val', 'predict')
TASKS = ('detect',)
TASK2DATA = {'detect': 'GEN1.yaml'}
TASK2MODEL = {'detect': 'dpadasnn_gen1_yolov8s.yaml'}
TASK2METRIC = {'detect': 'metrics/mAP50-95(B)'}
CLI_HELP_MSG = 'DP-AdaSNN: run python train.py, test.py, or detect.py --cfg <GEN1/DSEC YAML>. See README.md.'
CFG_FLOAT_KEYS = ('warmup_epochs', 'box', 'cls', 'dfl', 'degrees', 'shear', 'event_tau', 'spike_balance')
CFG_FRACTION_KEYS = ('iou', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum', 'warmup_bias_lr', 'label_smoothing', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'conf', 'iou', 'fraction')
CFG_INT_KEYS = ('epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic', 'mask_ratio', 'max_det', 'vid_stride', 'line_width', 'nbs', 'save_period', 'event_micro_slice')
CFG_BOOL_KEYS = ('save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'overlap_mask', 'val', 'save_json', 'save_hybrid', 'half', 'dnn', 'plots', 'show', 'save_txt', 'save_conf', 'save_crop', 'show_labels', 'show_conf', 'augment', 'agnostic_nms', 'retina_masks', 'boxes', 'event_time_reverse', 'event_no_letterbox_pad', 'event_train_shuffle')

def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace]=DEFAULT_CFG_DICT, overrides: Dict=None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)
    if overrides:
        overrides = cfg2dict(overrides)
        if 'save_dir' not in cfg:
            overrides.pop('save_dir', None)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}
    for k in ('project', 'name'):
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")
    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and (not isinstance(v, (int, float))):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not 0.0 <= v <= 1.0:
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and (not isinstance(v, int)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and (not isinstance(v, bool)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")
    return IterableSimpleNamespace(**cfg)

def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""
    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path
        project = args.project or (ROOT / '../tests/tmp/runs' if TESTS_RUNNING else Path(SETTINGS['runs_dir'])) / args.task
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)
    return Path(save_dir)

def _handle_deprecation(custom):
    """Hardcoded function to handle deprecated config keys."""
    for key in custom.copy().keys():
        if key == 'hide_labels':
            deprecation_warn(key, 'show_labels')
            custom['show_labels'] = custom.pop('hide_labels') == 'False'
        if key == 'hide_conf':
            deprecation_warn(key, 'show_conf')
            custom['show_conf'] = custom.pop('hide_conf') == 'False'
        if key == 'line_thickness':
            deprecation_warn(key, 'line_width')
            custom['line_width'] = custom.pop('line_thickness')
    return custom

def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list. If
    any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (dict): a dictionary of custom configuration options
        base (dict): a dictionary of base configuration options
        e (Error, optional): An optional error that is passed by the calling function.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f'{k}={base[k]}' if base.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e

def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings. The function considers cases where the first
    argument ends with '=' or the second starts with '=', as well as when the middle one is an equals sign.

    Args:
        args (List[str]): A list of strings where each element is an argument.

    Returns:
        List[str]: A list of strings where the arguments around isolated '=' are merged.
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == '=' and 0 < i < len(args) - 1:
            new_args[-1] += f'={args[i + 1]}'
            del args[i + 1]
        elif arg.endswith('=') and i < len(args) - 1 and ('=' not in args[i + 1]):
            new_args.append(f'{arg}{args[i + 1]}')
            del args[i + 1]
        elif arg.startswith('=') and i > 0:
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args

def parse_key_value_pair(pair):
    """Parse one 'key=value' pair and return key and value."""
    k, v = pair.split('=', 1)
    k, v = (k.strip(), v.strip())
    assert v, f"missing '{k}' value"
    return (k, smart_value(v))

def smart_value(v):
    """Convert a string to an underlying type such as int, float, bool, etc."""
    v_lower = v.lower()
    if v_lower == 'none':
        return None
    elif v_lower == 'true':
        return True
    elif v_lower == 'false':
        return False
    else:
        with contextlib.suppress(Exception):
            return eval(v)
        return v
