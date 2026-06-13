import unittest
from pathlib import Path


class PopeScriptArgsTest(unittest.TestCase):
    def test_pope_script_exposes_dvts_and_tgvc_counts(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "v1_5" / "eval" / "pope.sh"
        text = script.read_text()

        self.assertIn("<DVTS_token_num> <TGVC_token_num>", text)
        self.assertIn("--DVTS_token_num", text)
        self.assertIn("--TGVC_token_num", text)

    def test_pope_script_defaults_to_local_llava_checkpoint(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "v1_5" / "eval" / "pope.sh"
        text = script.read_text()

        self.assertIn('CKPT=${CKPT:-"$ROOT_DIR/../models/llava-v1.5-7b"}', text)

    def test_pope_script_defaults_to_workspace_hf_cache(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "v1_5" / "eval" / "pope.sh"
        text = script.read_text()

        self.assertIn('export HF_HOME=${HF_HOME:-"$ROOT_DIR/../.cache/huggingface"}', text)
        self.assertIn("export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}", text)
        self.assertIn("export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}", text)

    def test_tgvc_text_guidance_loads_from_local_cache_when_offline(self):
        repo_root = Path(__file__).resolve().parents[1]
        clip_encoder = repo_root / "llava" / "model" / "multimodal_encoder" / "clip_encoder.py"
        text = clip_encoder.read_text()

        self.assertIn("local_files_only=local_files_only", text)


if __name__ == "__main__":
    unittest.main()
