import unittest
from argparse import Namespace


from llava.model.language_model.llava_llama import resolve_visiontrim_config, resolve_visiontrim_aggregation_plan
from llava.model.multimodal_encoder.clip_encoder import normalize_visiontrim_method


class VisionTrimConfigTest(unittest.TestCase):
    def test_normalizes_method_name(self):
        self.assertEqual(normalize_visiontrim_method("visiontrim"), "VisionTrim")
        self.assertEqual(normalize_visiontrim_method("VisionTrim"), "VisionTrim")
        self.assertEqual(normalize_visiontrim_method("none"), "none")

    def test_supports_explicit_dvts_and_tgvc_counts(self):
        config = resolve_visiontrim_config(
            Namespace(method="visiontrim", DVTS_token_num=48, TGVC_token_num=16, token_num=None, visual_token_num=None, layer=None)
        )
        self.assertEqual(config.method, "VisionTrim")
        self.assertEqual(config.dvts_token_num, 48)
        self.assertEqual(config.tgvc_token_num, 16)
        self.assertEqual(config.total_visual_tokens, 64)

    def test_readme_token_num_uses_paper_l15_dvts_tgvc_split(self):
        config = resolve_visiontrim_config(
            Namespace(method="VisionTrim", token_num=64, DVTS_token_num=None, TGVC_token_num=None, visual_token_num=None, layer=None)
        )
        self.assertEqual(config.dvts_token_num, 48)
        self.assertEqual(config.tgvc_token_num, 16)
        self.assertEqual(config.total_visual_tokens, 64)

    def test_visual_token_num_alias_maps_to_legacy_token_num(self):
        config = resolve_visiontrim_config(
            Namespace(method="visiontrim", visual_token_num=64, token_num=None, DVTS_token_num=None, TGVC_token_num=None, layer=None)
        )
        self.assertEqual(config.dvts_token_num, 48)
        self.assertEqual(config.tgvc_token_num, 16)
        self.assertEqual(config.total_visual_tokens, 64)

    def test_explicit_zero_tgvc_keeps_pure_dvts_ablation(self):
        config = resolve_visiontrim_config(
            Namespace(method="VisionTrim", token_num=64, DVTS_token_num=64, TGVC_token_num=0, visual_token_num=None, layer=None)
        )
        self.assertEqual(config.dvts_token_num, 64)
        self.assertEqual(config.tgvc_token_num, 0)
        self.assertEqual(config.total_visual_tokens, 64)

    def test_no_llm_aggregation_when_rank_equals_image_tokens(self):
        plan = resolve_visiontrim_aggregation_plan(
            method="VisionTrim",
            requested_layer=4,
            sys_length=35,
            image_token_length=64,
            attention_rank=64,
            seq_length=125,
        )
        self.assertEqual(plan.agg_layer, -1)
        self.assertFalse(plan.should_aggregate)

    def test_enables_llm_aggregation_only_when_tokens_are_reduced(self):
        plan = resolve_visiontrim_aggregation_plan(
            method="VisionTrim",
            requested_layer=4,
            sys_length=35,
            image_token_length=128,
            attention_rank=64,
            seq_length=189,
        )
        self.assertEqual(plan.agg_layer, 4)
        self.assertTrue(plan.should_aggregate)


if __name__ == "__main__":
    unittest.main()
