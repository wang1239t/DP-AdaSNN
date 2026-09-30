# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box

from .utils import polygons2masks, polygons2masks_overlap

class BaseTransform:
    """
    Base class for image transformations.

    This is a generic transformation class that can be extended for specific image processing needs.
    The class is designed to be compatible with both classification and semantic segmentation tasks.

    Methods:
        __init__: Initializes the BaseTransform object.
        apply_image: Applies image transformation to labels.
        apply_instances: Applies transformations to object instances in labels.
        apply_semantic: Applies semantic segmentation to an image.
        __call__: Applies all label transformations to an image, instances, and semantic masks.
    """

    def __init__(self) -> None:
        """Initializes the BaseTransform object."""
        pass

    def apply_image(self, labels):
        """Applies image transformations to labels."""
        pass

    def apply_instances(self, labels):
        """Applies transformations to object instances in labels."""
        pass

    def apply_semantic(self, labels):
        """Applies semantic segmentation to an image."""
        pass

    def __call__(self, labels):
        """Applies all label transformations to an image, instances, and semantic masks."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)

class Compose:
    """Class for composing multiple image transformations."""

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)

        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def tolist(self):
        """Converts the list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"

class BaseMixTransform:
    """
    Class for base mix (MixUp/Mosaic) transformations.
    This implementation is from mmyolo.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        if random.uniform(0, 1) > self.p:
            return labels

        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels

        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _mix_transform(self, labels):
        raise NotImplementedError

    def get_indexes(self):
        raise NotImplementedError

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation adapted for 4D Spatiotemporal Event Data [T, H, W, C]
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        assert n in (4, 9), 'grid must be equal to 4 or 9.'
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)
        self.n = n

    def get_indexes(self, buffer=True):
        if buffer:
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        assert labels.get('rect_shape', None) is None, 'rect and mosaic are mutually exclusive.'
        assert len(labels.get('mix_labels', [])), 'There are no other images for mosaic augment.'
        return self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)

    def _mosaic4(self, labels):
        """Create a 2x2 spatio-temporal mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)

        for i in range(4):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            T_steps, C_dim = img.shape[0], img.shape[3]

            if i == 0:

                img4 = np.zeros((T_steps, s * 2, s * 2, C_dim), dtype=img.dtype)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[:, y1a:y2a, x1a:x2a, :] = img[:, y1b:y2b, x1b:x2b, :]

            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels

    def _mosaic9(self, labels):
        """Create a 3x3 spatio-temporal mosaic."""
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1
        for i in range(9):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            T_steps, C_dim = img.shape[0], img.shape[3]

            if i == 0:

                img9 = np.zeros((T_steps, s * 3, s * 3, C_dim), dtype=img.dtype)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s, s - h, s + w, s
            elif i == 2:
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)

            img9[:, y1:y2, x1:x2, :] = img[:, y1 - padh:, x1 - padw:, :]
            hp, wp = h, w

            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)

        final_labels['img'] = img9[:, -self.border[0]:self.border[0], -self.border[1]:self.border[1], :]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update labels."""

        nh, nw = labels['img'].shape[1:3]

        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(nw, nh)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'cls': np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border}
        final_labels['instances'].clip(imgsz, imgsz)
        good = final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls'] = final_labels['cls'][good]
        return final_labels

class MixUp(BaseMixTransform):
    """Class for applying MixUp augmentation to the dataset."""

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initializes MixUp object with dataset, pre_transform, and probability of applying MixUp."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies MixUp augmentation as per https://arxiv.org/pdf/1710.09412.pdf."""
        r = np.random.beta(32.0, 32.0)
        labels2 = labels['mix_labels'][0]
        labels['img'] = (labels['img'] * r + labels2['img'] * (1 - r)).astype(np.uint8)
        labels['instances'] = Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels

class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding bounding boxes, segments, and
    keypoints. These transformations include rotation, translation, scaling, and shearing. The class also offers the
    option to apply these transformations conditionally with a specified probability.

    [Modified for 4D Event Tensors (T, H, W, C)]
    """

    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None):
        """Initializes RandomPerspective object with transformation parameters."""
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """
        Applies a sequence of affine transformations centered around the image center.
        Supports both 3D (H, W, C) and 4D (T, H, W, C) tensors.
        """

        if len(img.shape) == 4:

            height, width = img.shape[1], img.shape[2]
        else:

            height, width = img.shape[0], img.shape[1]

        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2

        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]

        M = T @ S @ R @ P @ C

        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if len(img.shape) == 4:

                T_dim, H_dim, W_dim, C_dim = img.shape

                img_new = np.zeros((T_dim, self.size[1], self.size[0], C_dim), dtype=img.dtype)

                for t in range(T_dim):
                    if C_dim == 4:

                        img_t = img[t]

                        counts = img_t[..., 0:2]
                        times = img_t[..., 2:4]

                        if self.perspective:
                            warped_counts = cv2.warpPerspective(counts, M, dsize=self.size,
                                                                flags=cv2.INTER_LINEAR, borderValue=0)
                            warped_times = cv2.warpPerspective(times, M, dsize=self.size,
                                                               flags=cv2.INTER_NEAREST, borderValue=0)
                        else:
                            warped_counts = cv2.warpAffine(counts, M[:2], dsize=self.size,
                                                           flags=cv2.INTER_LINEAR, borderValue=0)
                            warped_times = cv2.warpAffine(times, M[:2], dsize=self.size,
                                                          flags=cv2.INTER_NEAREST, borderValue=0)
                        img_new[t] = np.concatenate([warped_counts, warped_times], axis=-1)
                    else:
                        if self.perspective:
                            img_new[t] = cv2.warpPerspective(img[t], M, dsize=self.size, borderValue=0)
                        else:
                            img_new[t] = cv2.warpAffine(img[t], M[:2], dsize=self.size, borderValue=0)

                img = img_new
            else:

                if self.perspective:
                    img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
                else:
                    img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))

        return img, M, s

    def apply_bboxes(self, bboxes, M):

        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):

        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):

        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T
        xy = xy[:, :2] / xy[:, 2:3]
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):

        if self.pre_transform and 'mosaic_border' not in labels:
            labels = self.pre_transform(labels)
        labels.pop('ratio_pad', None)

        img = labels['img']
        cls = labels['cls']
        instances = labels.pop('instances')

        instances.convert_bbox(format='xyxy')

        if len(img.shape) == 4:
            img_h, img_w = img.shape[1:3]
        else:
            img_h, img_w = img.shape[:2]
        instances.denormalize(img_w, img_h)

        border = labels.pop('mosaic_border', self.border)

        if len(img.shape) == 4:
            self.size = img.shape[2] + border[1] * 2, img.shape[1] + border[0] * 2
        else:
            self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2

        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)

        new_instances = Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
        new_instances.clip(*self.size)

        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(box1=instances.bboxes.T,
                                box2=new_instances.bboxes.T,
                                area_thr=0.01 if len(segments) else 0.10)

        labels['instances'] = new_instances[i]
        labels['cls'] = cls[i]
        labels['img'] = img

        if len(img.shape) == 4:
            labels['resized_shape'] = img.shape[1:3]
        else:
            labels['resized_shape'] = img.shape[:2]

        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):

        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)

class RandomHSV:
    """
    This class is responsible for performing random adjustments to the Hue, Saturation, and Value (HSV) channels of an
    image.

    The adjustments are random but within limits set by hgain, sgain, and vgain.
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """
        Initialize RandomHSV class with gains for each HSV channel.

        Args:
            hgain (float, optional): Maximum variation for hue. Default is 0.5.
            sgain (float, optional): Maximum variation for saturation. Default is 0.5.
            vgain (float, optional): Maximum variation for value. Default is 0.5.
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """
        Applies random HSV augmentation to an image within the predefined limits.

        The modified image replaces the original image in the input 'labels' dict.
        """
        img = labels['img']
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels

class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.
    [Modified for 4D Event Tensors (T, H, W, C)]
    """

    def __init__(self, p=0.5, direction='horizontal', flip_idx=None) -> None:
        assert direction in ['horizontal', 'vertical'], f'Support direction `horizontal` or `vertical`, got {direction}'
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        img = labels['img']
        instances = labels.pop('instances')

        instances.convert_bbox(format='xywh')

        if len(img.shape) == 4:
            h, w = img.shape[1:3]
        else:
            h, w = img.shape[:2]

        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        if self.direction == 'vertical' and random.random() < self.p:
            if len(img.shape) == 4:
                img = np.flip(img, axis=1)
            else:
                img = np.flipud(img)
            instances.flipud(h)

        if self.direction == 'horizontal' and random.random() < self.p:
            if len(img.shape) == 4:
                img = np.flip(img, axis=2)
            else:
                img = np.fliplr(img)
            instances.fliplr(w)

            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])

        labels['img'] = np.ascontiguousarray(img)
        labels['instances'] = instances
        return labels

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        if len(img.shape) == 4:
            shape = img.shape[1:3]
        else:
            shape = img.shape[:2]
        new_shape = labels.pop('rect_shape', self.new_shape)

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            if len(img.shape) == 4:
                T, H, W, C = img.shape
                img_new = np.zeros([T, new_unpad[1], new_unpad[0], C], dtype=img.dtype)
                for i in range(T):
                    if C == 4:

                        img_new[i] = custom_event_resize(img[i], target_w=new_unpad[0], target_h=new_unpad[1])
                    else:
                        img_new[i] = cv2.resize(img[i], (new_unpad[0], new_unpad[1]), interpolation=cv2.INTER_LINEAR)
                img = img_new
            else:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

        if len(img.shape) == 4:
            size = max(new_shape[1], new_shape[0])
            T, H, W, C = img.shape
            img_final = np.zeros([T, new_unpad[1] + top + bottom, new_unpad[0] + left + right, C], dtype=img.dtype)
            for i in range(T):
                img_final[i] = cv2.copyMakeBorder(img[i], top, bottom, left, right, cv2.BORDER_CONSTANT,

                                                  value=0)
            img = img_final
        else:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,

                                     value=0)

        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (left, top))

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        if len(labels['img'].shape) == 4:
            labels['instances'].denormalize(*labels['img'].shape[1:3][::-1])
        else:
            labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

class CopyPaste:
    """
    Implements the Copy-Paste augmentation as described in the paper https://arxiv.org/abs/2012.07177. This class is
    responsible for applying the Copy-Paste augmentation on images and their corresponding instances.
    """

    def __init__(self, p=0.5) -> None:
        """
        Initializes the CopyPaste class with a given probability.

        Args:
            p (float, optional): The probability of applying the Copy-Paste augmentation. Must be between 0 and 1.
                                 Default is 0.5.
        """
        self.p = p

    def __call__(self, labels):
        """
        Applies the Copy-Paste augmentation to the given image and instances.

        Args:
            labels (dict): A dictionary containing:
                           - 'img': The image to augment.
                           - 'cls': Class labels associated with the instances.
                           - 'instances': Object containing bounding boxes, and optionally, keypoints and segments.

        Returns:
            (dict): Dict with augmented image and updated instances under the 'img', 'cls', and 'instances' keys.

        Notes:
            1. Instances are expected to have 'segments' as one of their attributes for this augmentation to work.
            2. This method modifies the input dictionary 'labels' in place.
        """
        im = labels['img']
        cls = labels['cls']
        if len(im.shape) == 4:
            h, w = im.shape[1:3]
        else:
            h, w = im.shape[:2]
        instances = labels.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape
            im_new = np.zeros(im.shape, np.uint8)

            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]

        labels['img'] = im
        labels['cls'] = cls
        labels['instances'] = instances
        return labels

class Albumentations:
    """
    Albumentations transformations.

    Optional, uninstall package to disable. Applies Blur, Median Blur, convert to grayscale, Contrast Limited Adaptive
    Histogram Equalization, random change of brightness and contrast, RandomGamma and lowering of image quality by
    compression.
    """

    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr('albumentations: ')
        try:
            import albumentations as A

            check_version(A.__version__, '1.0.3', hard=True)

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f'{prefix}{e}')

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            labels['instances'].convert_bbox('xywh')
            labels['instances'].normalize(*im.shape[:2][::-1])
            bboxes = labels['instances'].bboxes

            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)
                if len(new['class_labels']) > 0:
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'], dtype=np.float32)
            labels['instances'].update(bboxes=bboxes)
        return labels

class Format:
    """
    Formats image annotations for object detection, instance segmentation, and pose estimation tasks. The class
    standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Default is 'xywh'.
        normalize (bool): Whether to normalize bounding boxes. Default is True.
        return_mask (bool): Return instance masks for segmentation. Default is False.
        return_keypoint (bool): Return keypoints for pose estimation. Default is False.
        mask_ratio (int): Downsample ratio for masks. Default is 4.
        mask_overlap (bool): Whether to overlap masks. Default is True.
        batch_idx (bool): Keep batch indexes. Default is True.
    """

    def __init__(self,
                 bbox_format='xywh',
                 normalize=True,
                 return_mask=False,
                 return_keypoint=False,
                 mask_ratio=4,
                 mask_overlap=True,
                 batch_idx=True):
        """Initializes the Format class with given parameters."""
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_keypoint = return_keypoint
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx

    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels.pop('img')
        if len(img.shape) == 4:
            h, w = img.shape[1:3]
        else:
            h, w = img.shape[:2]
        cls = labels.pop('cls')
        instances = labels.pop('instances')
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio,
                                    img.shape[1] // self.mask_ratio)
            labels['masks'] = masks
        if self.normalize:
            instances.normalize(w, h)
        labels['img'] = self._format_img(img)
        labels['cls'] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels['bboxes'] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels['keypoints'] = torch.from_numpy(instances.keypoints)

        if self.batch_idx:
            labels['batch_idx'] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """Format the image for YOLO from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if len(img.shape) == 4:
            img = np.ascontiguousarray(img.transpose(0, 3, 1, 2)[::-1])
        else:
            img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """Convert polygon points to bitmap."""
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls

def v8_transforms(dataset, imgsz, hyp, stretch=False, no_pad_shape=None):
    """Convert images to a size suitable for YOLOv8 training (Customized for SNN)."""

    transforms_list = []
    letterbox_shape = tuple(no_pad_shape) if no_pad_shape else (imgsz, imgsz)

    if not stretch:
        transforms_list.append(LetterBox(new_shape=letterbox_shape))

    transforms_list.append(Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic))

    transforms_list.append(CopyPaste(p=hyp.copy_paste))

    transforms_list.append(RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None
    ))

    pre_transform = Compose(transforms_list)

    flip_idx = dataset.data.get('flip_idx', [])
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),

        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)
    ])

def classify_transforms(size=224, rect=False, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
    """Transforms to apply if albumentations not installed."""
    if not isinstance(size, int):
        raise TypeError(f'classify_transforms() size {size} must be integer, not (list, tuple)')
    transforms = [ClassifyLetterBox(size, auto=True) if rect else CenterCrop(size), ToTensor()]
    if any(mean) or any(std):
        transforms.append(T.Normalize(mean, std, inplace=True))
    return T.Compose(transforms)

def hsv2colorjitter(h, s, v):
    """Map HSV (hue, saturation, value) jitter into ColorJitter values (brightness, contrast, saturation, hue)"""
    return v, v, s, h

def classify_albumentations(
        augment=True,
        size=224,
        scale=(0.08, 1.0),
        hflip=0.5,
        vflip=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        auto_aug=False,
):
    """YOLOv8 classification Albumentations (optional, only used if package is installed)."""
    prefix = colorstr('albumentations: ')
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, '1.0.3', hard=True)
        if augment:
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale)]
            if auto_aug:

                LOGGER.info(f'{prefix}auto augmentations are currently not supported')
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if any((hsv_h, hsv_s, hsv_v)):
                    T += [A.ColorJitter(*hsv2colorjitter(hsv_h, hsv_s, hsv_v))]
        else:
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]
        LOGGER.info(prefix + ', '.join(f'{x}'.replace('always_apply=False, ', '') for x in T if x.p))
        return A.Compose(T)

    except ImportError:
        pass
    except Exception as e:
        LOGGER.info(f'{prefix}{e}')

class ClassifyLetterBox:
    """
    YOLOv8 LetterBox class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([LetterBox(size), ToTensor()]).

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically solves for short side using stride.
        stride (int): The stride value, used when 'auto' is True.
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox class with a target size, auto-flag, and stride.

        Args:
            size (Union[int, Tuple[int, int]]): The target dimensions (height, width) for the letterbox.
            auto (bool): If True, automatically calculates the short side based on stride.
            stride (int): The stride value, used when 'auto' is True.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto
        self.stride = stride

    def __call__(self, im):
        """
        Resizes the image and pads it with a letterbox method.

        Args:
            im (numpy.ndarray): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The letterboxed and resized image as a numpy array.
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)
        h, w = round(imh * r), round(imw * r)

        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out

class CenterCrop:
    """YOLOv8 CenterCrop class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([CenterCrop(size), ToTensor()]).
    """

    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Resizes and crops the center of the image using a letterbox method.

        Args:
            im (numpy.ndarray): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The center-cropped and resized image as a numpy array.
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top:top + m, left:left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)

class ToTensor:
    """YOLOv8 ToTensor class for image preprocessing, i.e., T.Compose([LetterBox(size), ToTensor()])."""

    def __init__(self, half=False):
        """Initialize YOLOv8 ToTensor object with optional half-precision support."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Transforms an image from a numpy array to a PyTorch tensor, applying optional half-precision and normalization.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized to [0, 1].
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])
        im = torch.from_numpy(im)
        im = im.half() if self.half else im.float()
        im /= 255.0
        return im

def custom_event_resize(img_tensor, target_w, target_h):
    """
    解耦缩放：保护事件计数能量不流失，保护时间曲面状态不模糊
    img_tensor: 形状为 (H, W, 4) 的 numpy 数组
    """

    counts = img_tensor[..., 0:2]
    times = img_tensor[..., 2:4]

    resized_counts = cv2.resize(counts, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    resized_times = cv2.resize(times, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return np.concatenate([resized_counts, resized_times], axis=-1)
