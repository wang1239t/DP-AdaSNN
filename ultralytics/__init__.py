# Ultralytics YOLO, AGPL-3.0
import os
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
__version__ = '8.0.197'
from ultralytics.models import YOLO
__all__ = ('YOLO', '__version__')
