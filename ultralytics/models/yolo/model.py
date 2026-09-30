# Ultralytics YOLO, AGPL-3.0
from ultralytics.engine.model import Model
from ultralytics.models.yolo import detect
from ultralytics.nn.tasks import DetectionModel

class YOLO(Model):
    @property
    def task_map(self):
        return {'detect': {'model': DetectionModel, 'trainer': detect.DetectionTrainer,
                           'validator': detect.DetectionValidator, 'predictor': detect.DetectionPredictor}}
