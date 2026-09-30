# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (DATASETS_DIR, LOGGER, NUM_THREADS, ROOT, SETTINGS_YAML, TQDM, clean_url, colorstr,
                               emojis, yaml_load)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes

HELP_URL = 'See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance.'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'npy'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'

def img2label_paths(img_paths):
    """将events目录下的.npy路径转换为labels目录下的.txt路径"""

    event_dir = f'{os.sep}events{os.sep}'
    label_dir = f'{os.sep}labels{os.sep}'

    return [

        x.replace(event_dir, label_dir)

        .replace('.npy', '.txt')

        .replace(os.sep * 2, os.sep)
        for x in img_paths
    ]

def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()

def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size
    if img.format == 'JPEG':
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)
                if rotation in [6, 8]:
                    s = s[1], s[0]
    return s

def verify_image(args):
    """Verify one image."""
    (im_file, cls), prefix = args

    nf, nc, msg = 0, 0, ''
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = (shape[1], shape[0])
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'
        nf = 1
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
    return (im_file, cls), nf, nc, msg

def verify_image_label(args):
    """Verify one image-label pair (仅保留普通目标检测标签验证)"""
    im_file, lb_file, prefix, _, num_cls, *_ = args

    nm, nf, ne, nc, msg = 0, 0, 0, 0, ''
    lb = np.zeros((0, 5), dtype=np.float32)

    try:

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lines = f.read().strip().splitlines()
                lb = [x.split() for x in lines if len(x)]
                lb = np.array(lb, dtype=np.float32)

            nl = len(lb)
            if nl:

                assert lb.shape[1] == 5, f'标签需5列，当前检测到{lb.shape[1]}列'

                points = lb[:, 1:]
                assert points.max() <= 1, f'坐标值超过1.0: {points[points > 1]}'
                assert lb.min() >= 0, f'负值标签: {lb[lb < 0]}'

                max_cls = lb[:, 0].max()
                assert max_cls < num_cls, f'类别ID {int(max_cls)} 超出最大允许值 {num_cls - 1}'

                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    msg = f'{prefix}WARNING ⚠️ {lb_file}: {nl - len(i)} 个重复标签已移除'
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)

    except Exception as e:

        error_msg = f"验证标签文件时出错: {lb_file}\n错误类型: {type(e).__name__}\n错误信息: {str(e)}"
        print(error_msg)
        nc = 1
        msg = f'{prefix}ERROR ❌ {lb_file}: {e}'

    return im_file, lb, nm, nf, ne, nc, msg

def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)

    return cv2.resize(mask, (nw, nh))

def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])

def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index

def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the samestem as the provided path. An AssertionError
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob('*.yaml')) or list(path.rglob('*.yaml'))
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]

def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    data = check_file(dataset)

    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=False)
        data = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = data.parent, False

    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)

    for k in 'train', 'val':
        if k not in data:
            if k == 'val' and 'validation' in data:
                LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
                data['val'] = data.pop('validation')
            else:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data['path'] = path
    for k in 'train', 'val', 'test':
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
        if not all(x.exists() for x in val):
            name = clean_url(dataset)
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None
            if s.startswith('http') and s.endswith('.zip'):
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith('bash '):
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:
                exec(s, {'yaml': data})
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')

    return data
