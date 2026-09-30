# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from .augment import custom_event_resize
from .scripts.generate_slices import to_voxel_cube_numpy, to_timesurface_numpy, to_voxel_grid_numpy,\
    to_fused_count_surface_numpy
from skimage import exposure

from .utils import HELP_URL, IMG_FORMATS

def cfg_get(cfg, key, default=None):
    """Read a config value from a namespace or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def slice_events(events, num_slice, overlap=0):

    times = events[:, 3]
    if len(times) <= 0:
        return [None for i in range(num_slice)], 0

    time_window = (times[-1] - times[0]) // (
            num_slice * (
            1 - overlap) + overlap)
    stride = (1 - overlap) * time_window
    window_start_times = np.arange(num_slice) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)
    indices_end = np.searchsorted(times, window_end_times)
    slices = [events[start:end] for start, end in list(zip(indices_start, indices_end))]

    return slices, stride

def voxel_deal(voxel_grid):
    output_img = []
    for i in range(voxel_grid.shape[0]):
        img = voxel_grid[i, 0]

        mean_pos = np.mean(img[img > 0])
        mean_neg = np.mean(img[img < 0])
        var_pos = np.var(img[img > 0])
        var_neg = np.var(img[img < 0])

        img = np.clip(img, a_min=3 * mean_neg, a_max=3 * mean_pos)

        mean_pos = np.mean(img[img > 0])
        mean_neg = np.mean(img[img < 0])
        var_pos = np.var(img[img > 0])
        var_neg = np.var(img[img < 0])
        img = np.clip(img, a_min=mean_neg - 3 * var_neg, a_max=mean_pos + 3 * var_pos)

        max_val = np.max(img)
        min_val = np.min(img)

        img[img > 0] /= max_val
        img[img < 0] /= abs(min_val)

        map_img = np.zeros_like(img)
        map_img[img < 0] = img[img < 0] * 128 + 128
        map_img[img >= 0] = img[img >= 0] * 127 + 128

        output_img.append(map_img)

    output_img = np.stack(output_img, axis=0)
    return output_img

def dynamic_clipping(data, percentile=99.5, min_threshold=30):
    nonzero_data = data[data > 0]
    if len(nonzero_data) == 0:
        return data
    threshold = np.percentile(nonzero_data, percentile)
    threshold = max(threshold, min_threshold)
    return np.clip(data, a_min=None, a_max=threshold)

def micro_sum_deal(data):
    time_size, channels, height, width = data.shape
    normalized_data = np.zeros_like(data)

    for t in range(time_size):
        for c in range(channels):
            channel_data = data[t, c, :, :]

            clipped_data = dynamic_clipping(channel_data)

            normalized_data[t, c, :, :] = clipped_data

    return normalized_data

def dynamic_clipping_for_counts(counts_tensor, percentile=99.5, min_threshold=40):
    """
    专为事件计数定制的高效动态截断
    counts_tensor: 形状为 [T, 2, H, W] 的事件计数张量
    """

    nonzero_mask = counts_tensor > 0
    if not np.any(nonzero_mask):
        return counts_tensor

    nonzero_data = counts_tensor[nonzero_mask]
    threshold = np.percentile(nonzero_data, percentile)

    final_threshold = max(threshold, min_threshold)

    return np.clip(counts_tensor, a_min=0, a_max=final_threshold)

def finalize_fused_tensor(merged_frame):

    counts = merged_frame[:, 0:2, :, :]

    clipped_counts = dynamic_clipping_for_counts(counts, percentile=99.5, min_threshold=30)

    merged_frame[:, 0:2, :, :] = clipped_counts

    return merged_frame

class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        super().__init__()
        """Initialize BaseDataset with given configuration and options."""
        self.hyp = hyp
        self.image_size = tuple(int(x) for x in cfg_get(hyp, 'event_image_size', [240, 304]))
        self.micro_slice = int(cfg_get(hyp, 'event_micro_slice', 6))
        self.event_representation = cfg_get(hyp, 'event_representation', 'fused_count_surface')
        self.event_tau = float(cfg_get(hyp, 'event_tau', 50e3))
        self.event_time_reverse = bool(cfg_get(hyp, 'event_time_reverse', True))
        event_column_order = cfg_get(hyp, 'event_column_order', [0, 1, 3, 2])
        self.event_column_order = [int(x) for x in event_column_order] if event_column_order else None
        self.event_no_letterbox_pad = bool(cfg_get(hyp, 'event_no_letterbox_pad', False))
        event_train_shape = cfg_get(hyp, 'event_train_shape', None)
        self.event_train_shape = tuple(int(x) for x in event_train_shape) if event_train_shape else None
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)
        self.ni = len(self.labels)
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)

                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]

                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

            assert im_files, f'{self.prefix}No images found in {img_path}'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                keypoints = self.labels[i]['keypoints']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        index = i
        im, f, fn = self.ims[index], self.im_files[index], self.npy_files[index]
        if im is None:
            if fn.exists():
                event = np.load(fn)
                if self.event_column_order:
                    event = event[:, self.event_column_order]
                im = self.agrregate(event, method=self.event_representation)

                im = im.transpose((0, 2, 3, 1))

                h0, w0 = im.shape[1], im.shape[2]
            else:
                im = None
                h0, w0 = None, None
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    if len(im.shape) == 4:
                        T, H_old, W_old, C = im.shape

                        img_new = np.zeros([T, h, w, C], dtype=im.dtype)
                        for i in range(T):
                            if C == 4:
                                img_new[i] = custom_event_resize(im[i], target_w=w, target_h=h)
                            else:
                                img_new[i] = cv2.resize(im[i], (w, h), interpolation=cv2.INTER_LINEAR)
                        im = img_new
                    else:
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            if self.augment:
                self.ims[index], self.im_hw0[index], self.im_hw[index] = im, (h0, w0), im.shape[
                                                                                       :2]
                self.buffer.append(index)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            if len(im.shape) == 4:
                im_shape = im.shape[1:3]
            else:
                im_shape = im.shape[:2]
            return im, (h0, w0), im_shape

        return self.ims[index], self.im_hw0[index], self.im_hw[index]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))
            ratio = self.imgsz / max(im.shape[0], im.shape[1])
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)
        nb = bi[-1] + 1

        if self.event_no_letterbox_pad and self.event_train_shape:
            target_shape = np.array(self.event_train_shape, dtype=float)
            target_shape = np.ceil(target_shape / self.stride).astype(int) * self.stride
            self.batch_shapes = np.tile(target_shape, (nb, 1))
            self.batch = bi
            return

        s = np.array([x.pop('shape') for x in self.labels])
        ar = s[:, 0] / s[:, 1]
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])
        label.pop('shape', None)
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError

    def agrregate(self, events, method):

        if method == 'sum':
            frame = np.zeros(shape=[2, self.image_size[0] * self.image_size[1]])
            if events is None:

                return frame.reshape((2, self.image_size[0], self.image_size[1]))
            x = events[:, 0].astype(int)
            y = events[:, 1].astype(int)
            p = events[:, 2]

            mask = []
            mask.append(p == 0)
            mask.append(np.logical_not(mask[0]))
            for c in range(2):
                position = y[mask[c]] * self.image_size[1] + x[mask[c]]
                events_number_per_pos = np.bincount(position)
                frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
            frame = frame.reshape((2, self.image_size[0], self.image_size[1]))
        elif method == 'time':
            frame = np.zeros(shape=[2, self.image_size[0] * self.image_size[1]])
            if events is None:

                return frame.reshape((2, self.image_size[0], self.image_size[1]))
            x = events[:, 0].astype(int)
            y = events[:, 1].astype(int)
            p = events[:, 2]
            t = events[:, 3].astype(np.int64)
            t = t - t.min()
            time_interval = t.max() - t.min()
            t_s = t / time_interval

            mask = []
            mask.append(p == 0)
            mask.append(np.logical_not(mask[0]))
            for c in range(2):
                position = y[mask[c]] * self.image_size[1] + x[mask[c]]
                events_time = np.bincount(position, weights=t_s[mask[c]])
                frame[c][np.arange(len(events_time))] += events_time
            frame = frame.reshape((2, self.image_size[0], self.image_size[1]))
        elif method == 'voxel_grid':

            frame = to_voxel_grid_numpy(events, sensor_size=[self.image_size[1], self.image_size[0], 2],
                                        n_time_bins=self.micro_slice)

        elif method == 'micro_sum':
            if events is None:

                return np.zeros(shape=[self.micro_slice, 2, self.image_size[0], self.image_size[1]])
            slices, _ = slice_events(events, self.micro_slice)
            original_slices = slices

            T_max = np.max(events[:, 3]) if len(events) > 0 else 0
            reversed_slices = []

            for slc in reversed(slices):
                if slc is not None and len(slc) > 0:

                    slc_rev = slc.copy()

                    slc_rev[:, 3] = T_max - slc_rev[:, 3]

                    slc_rev = slc_rev[np.argsort(slc_rev[:, 3])]

                    reversed_slices.append(slc_rev)
                else:
                    reversed_slices.append(slc)

            slices = reversed_slices

            frame = np.stack([self.agrregate(slice, method='sum') for slice in slices])
            if not self.event_time_reverse:
                frame = np.stack([self.agrregate(slice, method='sum') for slice in original_slices])

        elif method == 'micro_time':
            if events is None:

                return np.zeros(shape=[self.micro_slice, 2, self.image_size[0], self.image_size[1]])
            slices, _ = slice_events(events, self.micro_slice)
            frame = np.stack([self.agrregate(slice, method='time') for slice in slices])

        elif method == 'timesurface':

            if events is None:

                return np.zeros(shape=[self.micro_slice, 2, self.image_size[0], self.image_size[1]])
            slices, dt = slice_events(events, self.micro_slice)
            frame = to_timesurface_numpy(slices, sensor_size=[self.image_size[1], self.image_size[0], 2], dt=dt,
                                         tau=self.event_tau)

        elif method == 'voxel_cube':
            frame = to_voxel_cube_numpy(events, sensor_size=[self.image_size[1], self.image_size[0], 2],
                                        num_slices=self.micro_slice)

        elif method == 'fused_count_surface':
            if events is None:

                return np.zeros(shape=[self.micro_slice, 4, self.image_size[0], self.image_size[1]])
            slices, dt = slice_events(events, self.micro_slice)
            original_slices = slices

            T_max = np.max(events[:, 3]) if len(events) > 0 else 0
            reversed_slices = []

            for slc in reversed(slices):
                if slc is not None and len(slc) > 0:

                    slc_rev = slc.copy()

                    slc_rev[:, 3] = T_max - slc_rev[:, 3]

                    slc_rev = slc_rev[np.argsort(slc_rev[:, 3])]

                    reversed_slices.append(slc_rev)
                else:
                    reversed_slices.append(slc)

            slices = reversed_slices

            frame = to_fused_count_surface_numpy(slices, sensor_size=[self.image_size[1], self.image_size[0], 2], dt=dt,
                                                 num_slices=self.micro_slice, tau=self.event_tau)
            if not self.event_time_reverse:
                frame = to_fused_count_surface_numpy(
                    original_slices,
                    sensor_size=[self.image_size[1], self.image_size[0], 2],
                    dt=dt,
                    num_slices=self.micro_slice,
                    tau=self.event_tau)
        else:
            frame = None

        return frame
