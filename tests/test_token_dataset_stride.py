from __future__ import annotations

import unittest

from llm_trainer.training import TokenDataset


class TokenDatasetStrideTests(unittest.TestCase):
    def test_stride_reduces_sample_count_and_advances_offsets(self) -> None:
        dataset = TokenDataset(tokens=list(range(12)), context_length=4, stride=2)
        self.assertEqual(len(dataset), 4)

        first_x, first_y = dataset[0]
        second_x, second_y = dataset[1]
        self.assertEqual(first_x.tolist(), [0, 1, 2, 3])
        self.assertEqual(first_y.tolist(), [1, 2, 3, 4])
        self.assertEqual(second_x.tolist(), [2, 3, 4, 5])
        self.assertEqual(second_y.tolist(), [3, 4, 5, 6])

    def test_stride_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TokenDataset(tokens=list(range(12)), context_length=4, stride=0)


if __name__ == "__main__":
    unittest.main()
