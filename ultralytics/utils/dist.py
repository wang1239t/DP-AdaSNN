# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9

def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f'{trainer.__class__.__module__}.{trainer.__class__.__name__}'.rsplit('.', 1)

    content = f'''overrides = {vars(trainer.args)} \nif __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.train()'''
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='_temp_',
                                     suffix=f'{id(trainer)}.py',
                                     mode='w+',
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)
    return file.name

def _ddp_relaunch_args(entry_file, launch_file):
    """Preserve script arguments only when DDP relaunches the original entrypoint."""
    return list(sys.argv[1:]) if launch_file == entry_file else []

def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)
    entry_file = str(Path(sys.argv[0]).resolve())
    file = entry_file
    safe_pattern = re.compile(r'^[a-zA-Z0-9_. /\\-]{1,128}$')
    if not (safe_pattern.match(file) and Path(file).exists() and file.endswith('.py')):
        file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{world_size}', '--master_port', f'{port}', file,
           *_ddp_relaunch_args(entry_file, file)]
    return cmd, file

def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f'{id(trainer)}.py' in file:
        os.remove(file)
