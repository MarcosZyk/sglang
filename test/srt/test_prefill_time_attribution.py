import unittest

from sglang.srt.managers.prefill_time_attribution import (
    calculate_attributed_prefill_times,
)


class TestPrefillTimeAttribution(unittest.TestCase):
    def test_single_request_gets_full_batch_time(self):
        self.assertEqual(calculate_attributed_prefill_times(2.0, [16384]), [2.0])

    def test_equal_token_contributions_split_batch_time_evenly(self):
        self.assertEqual(
            calculate_attributed_prefill_times(2.0, [8192, 8192]), [1.0, 1.0]
        )

    def test_unequal_token_contributions_use_token_ratio(self):
        self.assertEqual(
            calculate_attributed_prefill_times(2.0, [12288, 4096]), [1.5, 0.5]
        )

    def test_attributed_times_conserve_batch_time(self):
        attributed_times = calculate_attributed_prefill_times(
            1.7, [8000, 4000, 2000]
        )
        self.assertIsNotNone(attributed_times)
        self.assertAlmostEqual(sum(attributed_times), 1.7)

    def test_non_positive_lengths_do_not_receive_time(self):
        self.assertEqual(
            calculate_attributed_prefill_times(1.0, [1024, 0, -1]),
            [1.0, 0.0, 0.0],
        )
        self.assertIsNone(calculate_attributed_prefill_times(1.0, [0, -1]))


if __name__ == "__main__":
    unittest.main()
