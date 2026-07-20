from typing import List, Optional


def calculate_attributed_prefill_times(
    batch_elapsed: float, extend_lens: List[int]
) -> Optional[List[float]]:
    """Split one prefill batch's elapsed time in proportion to new tokens."""
    normalized_lens = [max(int(extend_len), 0) for extend_len in extend_lens]
    total_prefill_tokens = sum(normalized_lens)
    if total_prefill_tokens == 0:
        return None
    return [
        batch_elapsed * extend_len / total_prefill_tokens
        for extend_len in normalized_lens
    ]
