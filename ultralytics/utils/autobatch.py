# Ultralytics YOLO, AGPL-3.0
def check_train_batch_size(model, imgsz=640, amp=True):
    """Require an explicit batch size; automatic profiling is outside this release."""
    raise ValueError('Set an explicit positive batch size in the training YAML.')
