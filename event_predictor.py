"""Event-file inference and ordinary bounding-box rendering (AGPL-3.0)."""
from pathlib import Path

import numpy as np
import cv2

from ultralytics.data.augment import LetterBox
from ultralytics.data.loaders import SourceTypes, autocast_list
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.checks import check_imgsz

class EventFileLoader:
    """Read one event file per batch so memory does not grow with dataset size."""
    mode = 'image'
    bs = 1
    source_type = SourceTypes(from_img=True)

    def __init__(self, paths, cfg):
        self.paths = paths
        self.cfg = cfg
        self.count = 0

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.paths):
            raise StopIteration
        path = self.paths[self.count]
        self.count += 1
        images = autocast_list([path], event_config=self.cfg)
        return [path], images, None, ''

class EventDetectionPredictor(DetectionPredictor):
    def setup_source(self, source):
        paths = [str(p) for p in source] if isinstance(source, (list, tuple)) else [str(source)]
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)
        self.dataset = EventFileLoader(paths, self.args)
        self.source_type = self.dataset.source_type
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs

    def save_preds(self, vid_cap, idx, save_path):
        target = str(Path(save_path).with_suffix('.png'))
        if not cv2.imwrite(target, self.plotted_img):
            raise OSError(f'Unable to save prediction image: {target}')

    def pre_transform(self, images):
        if self.args.event_no_letterbox_pad:
            expected = tuple(self.args.event_train_shape or self.args.event_image_size)
            for image in images:
                if tuple(image.shape[1:3]) != expected:
                    raise ValueError(f'Expected no-pad event shape {expected}, got {image.shape[1:3]}')
            return images
        letterbox = LetterBox(self.imgsz, auto=False, stride=self.model.stride)
        return [letterbox(image=image) for image in images]

    def postprocess(self, preds, img, orig_imgs):
        predictions = ops.non_max_suppression(
            preds, self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det, classes=self.args.classes)
        results = []
        for index, (prediction, event_image) in enumerate(zip(predictions, orig_imgs)):

            counts = event_image[..., :2].sum(axis=0)
            level = np.clip(counts / 24.0, 0.0, 1.0)
            display = np.full((*counts.shape[:2], 3), 255.0, dtype=np.float32)
            display[..., 0] *= 1.0 - level[..., 0]
            display[..., 2] *= 1.0 - level[..., 1]
            display[..., 1] *= 1.0 - np.maximum(level[..., 0], level[..., 1])
            display = display.astype(np.uint8)
            prediction[:, :4] = ops.scale_boxes(img.shape[-2:], prediction[:, :4], display.shape)
            results.append(Results(display, path=str(Path(self.batch[0][index])),
                                   names=self.model.names, boxes=prediction))
        return results
