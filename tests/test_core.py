"""Small regression checks for the public core and input protocol."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml

from ultralytics.nn.modules.embedding import DPAdaSNNEmbedding
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).resolve().parents[1]

def make_core(dataset):
    cfg = yaml.safe_load((ROOT / f'ultralytics/cfg/models/v8/dpadasnn_{dataset}_yolov8.yaml').read_text())
    args = cfg['backbone'][0][3]
    return DPAdaSNNEmbedding(2, args[0], args[1], args[2], args[3], **args[4])

class CoreProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_train_loss_has_finite_threshold_gradient(self):
        core = make_core('gen1').train()
        events = torch.rand(2, 6, 4, 8, 12)
        events[:, :, :2] *= 12
        output, auxiliary_loss = core(events)
        self.assertEqual(tuple(output.shape), (4, 2, 2, 8, 12))
        self.assertEqual(auxiliary_loss.ndim, 0)
        (output.square().mean() + auxiliary_loss).backward()
        self.assertTrue(torch.isfinite(core.thresh_param.grad).all())
        self.assertFalse(hasattr(core, 'monitor_enabled'))
        self.assertFalse(hasattr(core, 'get_metrics'))

    def test_eval_is_tensor_and_resets_between_batches(self):
        core = make_core('dsec').eval()
        events = torch.rand(1, 4, 4, 8, 12)
        with torch.no_grad():
            first = core(events)
            core(torch.zeros_like(events))
            repeated = core(events)
        self.assertIsInstance(first, torch.Tensor)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)

    def test_gen1_count_cap_does_not_clip_time_surface(self):
        core = make_core('gen1').eval()
        events = torch.rand(1, 6, 4, 8, 12)
        events[:, :, 0] = 2000
        clipped = events.clone()
        clipped[:, :, :2].clamp_(max=512)
        with torch.no_grad():
            torch.testing.assert_close(core(events), core(clipped), rtol=0, atol=0)

    def test_wrong_temporal_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nb_steps'):
            make_core('dsec')(torch.zeros(1, 6, 4, 8, 12))

    def test_frozen_configs_keep_no_pad_and_test_split(self):
        cfg = yaml.safe_load((ROOT / 'ultralytics/cfg/experiments/dsec_train.yaml').read_text())
        self.assertTrue(cfg['event_no_letterbox_pad'])
        self.assertEqual(cfg['event_train_shape'], [480, 640])
        for dataset in ('gen1', 'dsec'):
            cfg = yaml.safe_load((ROOT / f'ultralytics/cfg/experiments/{dataset}_test.yaml').read_text())
            self.assertEqual(cfg['split'], 'test')

    def test_gpu_warmup_uses_event_steps_and_four_channels(self):
        from ultralytics.nn.autobackend import AutoBackend
        for steps in (6, 4):
            backend = object.__new__(AutoBackend)
            torch.nn.Module.__init__(backend)
            backend.device = SimpleNamespace(type='cuda')
            backend.fp16 = False
            backend.model = SimpleNamespace(model=[SimpleNamespace(nb_steps=steps)])
            with patch('ultralytics.nn.autobackend.torch.zeros', return_value=torch.empty(0)) as allocate:
                with patch.object(backend, 'forward') as forward:
                    backend.warmup((1, 2, 480, 640))
                    self.assertEqual(allocate.call_args.args, (1, steps, 4, 480, 640))
                    forward.assert_called_once()

    def test_event_predictions_scale_to_original_image_and_draw_boxes(self):
        from event_predictor import EventDetectionPredictor
        predictor = object.__new__(EventDetectionPredictor)
        predictor.args = SimpleNamespace(conf=0.25, iou=0.7, agnostic_nms=False,
                                         max_det=300, classes=None)
        predictor.model = SimpleNamespace(names={0: 'car', 1: 'pedestrian'})
        predictor.batch = (['synthetic.npy'],)
        predictions = torch.tensor([[[32.0], [32.0], [16.0], [16.0], [1.0], [0.0]]])
        events = np.zeros((6, 32, 64, 4), dtype=np.float32)
        results = predictor.postprocess(predictions, torch.zeros(1, 6, 4, 64, 64), [events])
        self.assertEqual(len(results[0].boxes), 1)
        self.assertEqual(results[0].orig_shape, (32, 64))
        torch.testing.assert_close(results[0].boxes.xyxy, torch.tensor([[24.0, 8.0, 40.0, 24.0]]))
        drawn = results[0].plot()
        self.assertEqual(drawn.shape, (32, 64, 3))
        self.assertFalse(np.array_equal(drawn, results[0].orig_img))

if __name__ == '__main__':
    unittest.main()
