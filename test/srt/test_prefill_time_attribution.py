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

    def test_artesia_load_time_is_removed_before_token_attribution(self):
        self.assertEqual(
            calculate_attributed_prefill_times(
                batch_elapsed=10.0,
                extend_lens=[6000, 4000],
                batch_load_kv_elapsed=3.0,
            ),
            [4.2, 2.8],
        )

    def test_requested_formula_with_request_owned_io(self):
        shared_times = calculate_attributed_prefill_times(
            batch_elapsed=10.0,
            extend_lens=[6000, 4000],
            batch_load_kv_elapsed=3.0,
        )
        request_a = shared_times[0] + 1.0 + 0.5
        request_b = shared_times[1] + 2.0 + 0.75
        self.assertAlmostEqual(request_a, 5.7)
        self.assertAlmostEqual(request_b, 5.55)
        self.assertAlmostEqual(request_a + request_b, 10.0 + 0.5 + 0.75)

    def test_load_time_cannot_make_shared_time_negative(self):
        self.assertEqual(
            calculate_attributed_prefill_times(
                batch_elapsed=1.0,
                extend_lens=[1, 1],
                batch_load_kv_elapsed=2.0,
            ),
            [0.0, 0.0],
        )

    def test_waiting_request_load_is_not_shared_by_running_batch(self):
        shared_times = calculate_attributed_prefill_times(
            batch_elapsed=10.0,
            extend_lens=[6000, 4000],
            # Two seconds belong to the running requests and one second belongs
            # to a waiting request that was inspected but not admitted.
            batch_load_kv_elapsed=3.0,
        )
        running_requests = sum(shared_times) + 1.0 + 1.0
        waiting_request_pending_load = 1.0
        self.assertAlmostEqual(
            running_requests + waiting_request_pending_load, 10.0
        )


if __name__ == "__main__":
    unittest.main()
