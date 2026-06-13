import unittest

import torch

from llava.model.multimodal_encoder.clip_encoder import aggregate_rest_tokens_into_selected


class VisionTrimRestAggregationTest(unittest.TestCase):
    def test_merges_rest_tokens_into_nearest_selected_tokens(self):
        selected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        rest = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

        aggregated = aggregate_rest_tokens_into_selected(selected, rest)

        expected = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        self.assertTrue(torch.allclose(aggregated, expected))

    def test_uses_rest_token_scores_as_assignment_weights(self):
        selected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        rest = torch.tensor([[[2.0, 0.0], [10.0, 0.0], [0.0, 4.0]]])
        rest_scores = torch.tensor([[1.0, 3.0, 1.0]])

        aggregated = aggregate_rest_tokens_into_selected(
            selected,
            rest,
            rest_token_scores=rest_scores,
        )

        expected = torch.tensor([[[9.0, 0.0], [0.0, 5.0]]])
        self.assertTrue(torch.allclose(aggregated, expected))

    def test_keeps_selected_tokens_when_there_are_no_rest_tokens(self):
        selected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        rest = selected[:, :0, :]

        aggregated = aggregate_rest_tokens_into_selected(selected, rest)

        self.assertTrue(torch.equal(aggregated, selected))


if __name__ == "__main__":
    unittest.main()
