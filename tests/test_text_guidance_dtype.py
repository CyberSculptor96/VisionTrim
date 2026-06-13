import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from llava.model.multimodal_encoder import clip_encoder


class DummyVisionTower(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype


class DummyGuidanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)


class TextGuidanceDtypeTest(unittest.TestCase):
    def test_lazy_text_guidance_uses_vision_tower_dtype(self):
        tower = clip_encoder.CLIPVisionTower.__new__(clip_encoder.CLIPVisionTower)
        nn.Module.__init__(tower)
        tower.vision_tower_name = "dummy-clip"
        tower.vision_tower = DummyVisionTower(torch.float16)
        tower.clip_model = None
        tower.text_tokenizer = None
        tower.text_encoder = None

        clip_model = DummyGuidanceModel()
        text_encoder = DummyGuidanceModel()

        with patch.object(clip_encoder.CLIPModel, "from_pretrained", return_value=clip_model), \
                patch.object(clip_encoder.CLIPTokenizer, "from_pretrained", return_value=object()), \
                patch.object(clip_encoder.CLIPTextModel, "from_pretrained", return_value=text_encoder):
            tower.load_text_guidance_model()

        self.assertEqual(next(tower.clip_model.parameters()).dtype, torch.float16)
        self.assertEqual(next(tower.text_encoder.parameters()).dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
