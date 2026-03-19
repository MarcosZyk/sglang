#include "common.h"
#include "gemm.h"
#include "vec.h"

void build_tree_kernel_efficient(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
    constexpr int64_t FULL_MASK = 0;
    constexpr int64_t QLEN_ONLY = 1;
    constexpr int64_t QLEN_ONLY_BITPACKING = 2;

    TORCH_CHECK(parent_list.device().is_cpu(), "parent_list must be a CPU tensor");
    TORCH_CHECK(selected_index.device().is_cpu(), "selected_index must be a CPU tensor");
    TORCH_CHECK(verified_seq_len.device().is_cpu(), "verified_seq_len must be a CPU tensor");
    TORCH_CHECK(tree_mask.device().is_cpu(), "tree_mask must be a CPU tensor");
    TORCH_CHECK(positions.device().is_cpu(), "positions must be a CPU tensor");
    TORCH_CHECK(retrive_index.device().is_cpu(), "retrive_index must be a CPU tensor");
    TORCH_CHECK(retrive_next_token.device().is_cpu(), "retrive_next_token must be a CPU tensor");
    TORCH_CHECK(retrive_next_sibling.device().is_cpu(), "retrive_next_sibling must be a CPU tensor");

    TORCH_CHECK(parent_list.dim() == 2, "parent_list must be a 2D tensor");
    TORCH_CHECK(selected_index.dim() == 2, "selected_index must be a 2D tensor");
    TORCH_CHECK(verified_seq_len.dim() == 1, "verified_seq_len must be a 1D tensor");
    TORCH_CHECK(tree_mask.dim() == 1, "tree_mask must be a 1D tensor");
    TORCH_CHECK(positions.dim() == 1, "positions must be a 1D tensor");
    TORCH_CHECK(retrive_index.dim() == 2, "retrive_index must be a 2D tensor");
    TORCH_CHECK(retrive_next_token.dim() == 2, "retrive_next_token must be a 2D tensor");
    TORCH_CHECK(retrive_next_sibling.dim() == 2, "retrive_next_sibling must be a 2D tensor");


    const int64_t bs = parent_list.size(0);
    TORCH_CHECK(selected_index.size(0) == bs, "selected_index batch size mismatch");
    TORCH_CHECK(verified_seq_len.size(0) == bs, "verified_seq_len batch size mismatch");
    TORCH_CHECK(retrive_index.size(0) == bs, "retrive_index batch size mismatch");
    TORCH_CHECK(retrive_next_token.size(0) == bs, "retrive_next_token batch size mismatch");
    TORCH_CHECK(retrive_next_sibling.size(0) == bs, "retrive_next_sibling batch size mismatch");

    TORCH_CHECK(retrive_index.size(1) == draft_token_num, "retrive_index width mismatch");
    TORCH_CHECK(retrive_next_token.size(1) == draft_token_num, "retrive_next_token width mismatch");
    TORCH_CHECK(retrive_next_sibling.size(1) == draft_token_num, "retrive_next_sibling width mismatch");
    TORCH_CHECK(positions.numel() >= bs * draft_token_num, "positions numel mismatch");
    TORCH_CHECK(selected_index.size(1) == draft_token_num - 1, "selected_index width mismatch");
    TORCH_CHECK(parent_list.size(1) >= 0, "parent_list width mismatch");

    TORCH_CHECK(topk > 0, "topk must be > 0");
    TORCH_CHECK(depth > 0, "depth must be > 0");
    TORCH_CHECK(draft_token_num > 0, "draft_token_num must be > 0");

    TORCH_CHECK(
        tree_mask_mode == FULL_MASK || tree_mask_mode == QLEN_ONLY ||
            tree_mask_mode == QLEN_ONLY_BITPACKING,
        "Invalid tree_mask_mode");

    auto parent_list_contig = parent_list.contiguous();
    auto selected_index_contig = selected_index.contiguous();
    auto verified_seq_len_contig = verified_seq_len.contiguous();
    auto tree_mask_contig = tree_mask.contiguous();
    auto positions_contig = positions.contiguous();
    auto retrive_index_contig = retrive_index.contiguous();
    auto retrive_next_token_contig = retrive_next_token.contiguous();
    auto retrive_next_sibling_contig = retrive_next_sibling.contiguous();

    const int64_t* parent_list_ptr = parent_list_contig.data_ptr<int64_t>();
    const int64_t parent_stride = parent_list_contig.size(1);
    const int64_t* selected_index_ptr = selected_index_contig.data_ptr<int64_t>();
    const int64_t* verified_seq_len_ptr = verified_seq_len_contig.data_ptr<int64_t>();
    int64_t* positions_ptr = positions_contig.data_ptr<int64_t>();
    int64_t* retrive_index_ptr = retrive_index_contig.data_ptr<int64_t>();
    int64_t* retrive_next_token_ptr = retrive_next_token_contig.data_ptr<int64_t>();
    int64_t* retrive_next_sibling_ptr = retrive_next_sibling_contig.data_ptr<int64_t>();

    auto resolve_parent_position = [&](int64_t bid, int64_t parent_tb_idx) {
        if (parent_tb_idx <= 0) {
            return int64_t(0);
        }

        const int64_t parent_offset = bid * parent_stride;
        const int64_t selected_offset = bid * (draft_token_num - 1);
        if (parent_tb_idx < 0 || parent_tb_idx >= parent_stride) {
            return draft_token_num;
        }
        const int64_t parent_token_idx = parent_list_ptr[parent_offset + parent_tb_idx];

        for (int64_t p = 0; p < draft_token_num - 1; ++p) {
            if (selected_index_ptr[selected_offset + p] == parent_token_idx) {
                return p + 1;
            }
        }
        return draft_token_num;
    };

    if (tree_mask_mode == QLEN_ONLY_BITPACKING) {
        TORCH_CHECK(
            tree_mask_contig.scalar_type() == at::kByte ||
                tree_mask_contig.scalar_type() == at::kUInt16 ||
                tree_mask_contig.scalar_type() == at::kUInt32,
            "tree_mask must be uint8/uint16/uint32 for QLEN_ONLY_BITPACKING");

        size_t num_bytes_per_item = 1;
        if (draft_token_num > 16) {
            num_bytes_per_item = 4;
        } else if (draft_token_num > 8) {
            num_bytes_per_item = 2;
        }

        uint8_t* tree_mask_ptr = static_cast<uint8_t*>(tree_mask_contig.data_ptr());

        // Each batch is independent — parallelize over bs
        at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
          for (int64_t bid = begin; bid < end; ++bid) {
            const int64_t seq_len = verified_seq_len_ptr[bid];
            const int64_t selected_offset = bid * (draft_token_num - 1);
            const int64_t retrive_offset = bid * draft_token_num;

            for (int64_t tid = 0; tid < draft_token_num; ++tid) {
                const int64_t token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item;
                for (size_t b = 0; b < num_bytes_per_item; ++b) {
                    tree_mask_ptr[token_tree_idx + b] = 0;
                }
                tree_mask_ptr[token_tree_idx] = 1;

                if (tid == 0) {
                    positions_ptr[retrive_offset] = seq_len;

                    for (int64_t i = draft_token_num - 1; i > 0; --i) {
                        retrive_index_ptr[retrive_offset + i] = retrive_offset + i;

                        const int64_t parent_tb_idx =
                            selected_index_ptr[selected_offset + i - 1] / topk;
                        const int64_t parent_position =
                            resolve_parent_position(bid, parent_tb_idx);

                        if (parent_position == draft_token_num) {
                            continue;
                        }

                        TORCH_CHECK(
                            parent_position >= 0 && parent_position < draft_token_num,
                            "parent_position out of bounds in tid==0 (QLEN_ONLY_BITPACKING): ", parent_position,
                            " (expected [0, ", draft_token_num, "))");

                        if (retrive_next_token_ptr[retrive_offset + parent_position] == -1) {
                            retrive_next_token_ptr[retrive_offset + parent_position] = i;
                        } else {
                            const int64_t origin_next_token =
                                retrive_next_token_ptr[retrive_offset + parent_position];
                            retrive_next_token_ptr[retrive_offset + parent_position] = i;
                            retrive_next_sibling_ptr[retrive_offset + i] = origin_next_token;
                        }
                    }
                    retrive_index_ptr[retrive_offset] = retrive_offset;
                } else {
                    int64_t position = 0;
                    int64_t cur_position = tid - 1;

                    while (true) {
                        TORCH_CHECK(
                            cur_position >= 0 && cur_position < draft_token_num,
                            "cur_position out of bounds in bitpacking tid!=0: ", cur_position,
                            " (expected [0, ", draft_token_num, "))");

                        position += 1;
                        const int64_t byte_idx = (cur_position + 1) / 8;
                        const int64_t bit_idx = (cur_position + 1) % 8;
                        tree_mask_ptr[token_tree_idx + byte_idx] |= (1 << bit_idx);

                        const int64_t parent_tb_idx =
                            selected_index_ptr[selected_offset + cur_position] / topk;
                        if (parent_tb_idx == 0) {
                            break;
                        }

                        const int64_t next_position =
                            resolve_parent_position(bid, parent_tb_idx) - 1;
                        if (next_position < 0 || next_position >= draft_token_num - 1) {
                            break;
                        }
                        cur_position = next_position;
                    }

                    positions_ptr[retrive_offset + tid] = position + seq_len;
                }
            }
          }
        });
    } else {
        TORCH_CHECK(
            tree_mask_contig.scalar_type() == at::kBool,
            "tree_mask must be torch.bool for FULL_MASK/QLEN_ONLY");

        bool* tree_mask_ptr = tree_mask_contig.data_ptr<bool>();

        // Pre-compute prefix_seq_sum for each batch so that batches can run in parallel.
        // prefix_seq_sums[bid] = sum of verified_seq_len[0..bid-1]
        std::vector<int64_t> prefix_seq_sums(bs);
        {
            int64_t running_sum = 0;
            for (int64_t b = 0; b < bs; ++b) {
                prefix_seq_sums[b] = running_sum;
                running_sum += verified_seq_len_ptr[b];
            }
        }

        // Each batch is independent — parallelize over bs
        at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
          for (int64_t bid = begin; bid < end; ++bid) {
            const int64_t seq_len = verified_seq_len_ptr[bid];
            const int64_t selected_offset = bid * (draft_token_num - 1);
            const int64_t retrive_offset = bid * draft_token_num;

            const int64_t seq_tree_idx =
                draft_token_num * draft_token_num * bid + prefix_seq_sums[bid] * draft_token_num;

            for (int64_t tid = 0; tid < draft_token_num; ++tid) {
                int64_t token_tree_idx;
                if (tree_mask_mode == FULL_MASK) {
                    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
                } else {
                    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
                }

                tree_mask_ptr[token_tree_idx - 1] = true;
                for (int64_t i = 0; i < draft_token_num - 1; ++i) {
                    tree_mask_ptr[token_tree_idx + i] = false;
                }

                if (tid == 0) {
                    positions_ptr[retrive_offset] = seq_len;

                    for (int64_t i = draft_token_num - 1; i > 0; --i) {
                        retrive_index_ptr[retrive_offset + i] = retrive_offset + i;

                        const int64_t parent_tb_idx =
                            selected_index_ptr[selected_offset + i - 1] / topk;
                        const int64_t parent_position =
                            resolve_parent_position(bid, parent_tb_idx);

                        if (parent_position == draft_token_num) {
                            continue;
                        }

                        TORCH_CHECK(
                            parent_position >= 0 && parent_position < draft_token_num,
                            "parent_position out of bounds in tid==0 (FULL_MASK): ", parent_position,
                            " (expected [0, ", draft_token_num, "))");

                        if (retrive_next_token_ptr[retrive_offset + parent_position] == -1) {
                            retrive_next_token_ptr[retrive_offset + parent_position] = i;
                        } else {
                            const int64_t origin_next_token =
                                retrive_next_token_ptr[retrive_offset + parent_position];
                            retrive_next_token_ptr[retrive_offset + parent_position] = i;
                            retrive_next_sibling_ptr[retrive_offset + i] = origin_next_token;
                        }
                    }
                    retrive_index_ptr[retrive_offset] = retrive_offset;
                } else {
                    int64_t position = 0;
                    int64_t cur_position = tid - 1;

                    while (true) {
                        TORCH_CHECK(
                            cur_position >= 0 && cur_position < draft_token_num,
                            "cur_position out of bounds in full_mask tid!=0: ", cur_position,
                            " (expected [0, ", draft_token_num, "))");

                        position += 1;
                        tree_mask_ptr[token_tree_idx + cur_position] = true;

                        const int64_t parent_tb_idx =
                            selected_index_ptr[selected_offset + cur_position] / topk;
                        if (parent_tb_idx == 0) {
                            break;
                        }

                        const int64_t next_position =
                            resolve_parent_position(bid, parent_tb_idx) - 1;
                        if (next_position < 0 || next_position >= draft_token_num - 1) {
                            break;
                        }
                        cur_position = next_position;
                    }

                    positions_ptr[retrive_offset + tid] = position + seq_len;
                }
            }
          }
        });
    }

    if (!tree_mask.is_contiguous()) {
        tree_mask.copy_(tree_mask_contig);
    }
    if (!positions.is_contiguous()) {
        positions.copy_(positions_contig);
    }
    if (!retrive_index.is_contiguous()) {
        retrive_index.copy_(retrive_index_contig);
    }
    if (!retrive_next_token.is_contiguous()) {
        retrive_next_token.copy_(retrive_next_token_contig);
    }
    if (!retrive_next_sibling.is_contiguous()) {
        retrive_next_sibling.copy_(retrive_next_sibling_contig);
    }
}

void verify_tree_greedy(
    at::Tensor predicts,          // mutable
    at::Tensor accept_index,      // mutable
    at::Tensor accept_token_num,  // mutable
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict,
    int64_t cuda_stream) {
    (void)cuda_stream;

    TORCH_CHECK(predicts.device().is_cpu(), "predicts must be a CPU tensor");
    TORCH_CHECK(accept_index.device().is_cpu(), "accept_index must be a CPU tensor");
    TORCH_CHECK(accept_token_num.device().is_cpu(), "accept_token_num must be a CPU tensor");
    TORCH_CHECK(candidates.device().is_cpu(), "candidates must be a CPU tensor");
    TORCH_CHECK(retrive_index.device().is_cpu(), "retrive_index must be a CPU tensor");
    TORCH_CHECK(retrive_next_token.device().is_cpu(), "retrive_next_token must be a CPU tensor");
    TORCH_CHECK(retrive_next_sibling.device().is_cpu(), "retrive_next_sibling must be a CPU tensor");
    TORCH_CHECK(target_predict.device().is_cpu(), "target_predict must be a CPU tensor");

    TORCH_CHECK(predicts.dim() == 1, "predicts must be a 1D tensor");
    TORCH_CHECK(accept_index.dim() == 2, "accept_index must be a 2D tensor");
    TORCH_CHECK(accept_token_num.dim() == 1, "accept_token_num must be a 1D tensor");
    TORCH_CHECK(candidates.dim() == 2, "candidates must be a 2D tensor");
    TORCH_CHECK(retrive_index.dim() == 2, "retrive_index must be a 2D tensor");
    TORCH_CHECK(retrive_next_token.dim() == 2, "retrive_next_token must be a 2D tensor");
    TORCH_CHECK(retrive_next_sibling.dim() == 2, "retrive_next_sibling must be a 2D tensor");
    TORCH_CHECK(target_predict.dim() == 2, "target_predict must be a 2D tensor");

    TORCH_CHECK(predicts.scalar_type() == at::kInt, "Expected 'predicts' to be torch.int32.");
    TORCH_CHECK(accept_index.scalar_type() == at::kInt, "Expected 'accept_index' to be torch.int32.");
    TORCH_CHECK(accept_token_num.scalar_type() == at::kInt, "Expected 'accept_token_num' to be torch.int32.");
    TORCH_CHECK(candidates.scalar_type() == at::kLong, "Expected 'candidates' to be torch.int64.");
    TORCH_CHECK(retrive_index.scalar_type() == at::kLong, "Expected 'retrive_index' to be torch.int64.");
    TORCH_CHECK(retrive_next_token.scalar_type() == at::kLong, "Expected 'retrive_next_token' to be torch.int64.");
    TORCH_CHECK(retrive_next_sibling.scalar_type() == at::kLong, "Expected 'retrive_next_sibling' to be torch.int64.");
    TORCH_CHECK(target_predict.scalar_type() == at::kLong, "Expected 'target_predict' to be torch.int64.");

    const auto batch_size = candidates.size(0);
    const auto num_draft_tokens = candidates.size(1);
    const auto num_speculative_tokens = accept_index.size(1);

    TORCH_CHECK(accept_index.size(0) == batch_size, "accept_index batch size mismatch");
    TORCH_CHECK(accept_token_num.size(0) == batch_size, "accept_token_num batch size mismatch");
    TORCH_CHECK(retrive_index.size(0) == batch_size, "retrive_index batch size mismatch");
    TORCH_CHECK(retrive_next_token.size(0) == batch_size, "retrive_next_token batch size mismatch");
    TORCH_CHECK(retrive_next_sibling.size(0) == batch_size, "retrive_next_sibling batch size mismatch");
    TORCH_CHECK(target_predict.size(0) == batch_size, "target_predict batch size mismatch");

    TORCH_CHECK(retrive_index.size(1) == num_draft_tokens, "retrive_index draft-token size mismatch");
    TORCH_CHECK(
            retrive_next_token.size(1) == num_draft_tokens,
            "retrive_next_token draft-token size mismatch");
    TORCH_CHECK(
            retrive_next_sibling.size(1) == num_draft_tokens,
            "retrive_next_sibling draft-token size mismatch");
    TORCH_CHECK(target_predict.size(1) == num_draft_tokens, "target_predict draft-token size mismatch");

    TORCH_CHECK(
            predicts.numel() >= batch_size * num_draft_tokens,
            "predicts numel mismatch: expected at least batch_size * num_draft_tokens");
    
    TORCH_CHECK(
            target_predict.numel() >= batch_size * num_draft_tokens,
            "target_predict numel mismatch: expected at least batch_size * num_draft_tokens");

    auto predicts_contig = predicts.contiguous();
    auto accept_index_contig = accept_index.contiguous();
    auto accept_token_num_contig = accept_token_num.contiguous();
    auto candidates_contig = candidates.contiguous();
    auto retrive_index_contig = retrive_index.contiguous();
    auto retrive_next_token_contig = retrive_next_token.contiguous();
    auto retrive_next_sibling_contig = retrive_next_sibling.contiguous();
    auto target_predict_contig = target_predict.contiguous();

    int32_t* predicts_ptr = predicts_contig.data_ptr<int32_t>();
    int32_t* accept_index_ptr = accept_index_contig.data_ptr<int32_t>();
    int32_t* accept_token_num_ptr = accept_token_num_contig.data_ptr<int32_t>();
    const int64_t* candidates_ptr = candidates_contig.data_ptr<int64_t>();
    const int64_t* retrive_index_ptr = retrive_index_contig.data_ptr<int64_t>();
    const int64_t* retrive_next_token_ptr = retrive_next_token_contig.data_ptr<int64_t>();
    const int64_t* retrive_next_sibling_ptr = retrive_next_sibling_contig.data_ptr<int64_t>();
    const int64_t* target_predict_ptr = target_predict_contig.data_ptr<int64_t>();

    // Each batch is independent — parallelize over batch_size.
    // Each batch writes to its own slice of predicts (via retrive_index),
    // accept_index, and accept_token_num, so there are no data races.
    const int64_t total_draft_elements = batch_size * num_draft_tokens;
    const int64_t target_predict_numel = target_predict.numel();

    at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
      for (int64_t bx = begin; bx < end; ++bx) {
        int64_t last_accepted_retrive_idx = retrive_index_ptr[bx * num_draft_tokens];
        TORCH_CHECK(
            last_accepted_retrive_idx >= 0 &&
                last_accepted_retrive_idx < total_draft_elements,
            "invalid retrive_index[", bx, ",0]=", last_accepted_retrive_idx,
            ", expected in [0, ", total_draft_elements, ")");

        TORCH_CHECK(
            last_accepted_retrive_idx < target_predict_numel,
            "target_predict index out of bounds: ", last_accepted_retrive_idx,
            " >= ", target_predict_numel);

        accept_index_ptr[bx * num_speculative_tokens] = static_cast<int32_t>(last_accepted_retrive_idx);

        int32_t num_accepted_tokens = 0;
        int64_t cur_index = 0;
        for (int64_t j = 1; j < num_speculative_tokens; ++j) {
            cur_index = retrive_next_token_ptr[bx * num_draft_tokens + cur_index];
            while (cur_index != -1) {
                TORCH_CHECK(
                    cur_index >= 0 && cur_index < num_draft_tokens,
                    "invalid retrive_next_token index at batch ", bx,
                    ": ", cur_index,
                    ", expected in [-1, ", num_draft_tokens, ")");
                const int64_t draft_index = retrive_index_ptr[bx * num_draft_tokens + cur_index];
                TORCH_CHECK(
                    draft_index >= 0 && draft_index < total_draft_elements,
                    "invalid draft_index at batch ", bx,
                    ": ", draft_index,
                    ", expected in [0, ", total_draft_elements, ")");
                const int64_t draft_token_id = candidates_ptr[bx * num_draft_tokens + cur_index];
                TORCH_CHECK(
                    last_accepted_retrive_idx >= 0 &&
                        last_accepted_retrive_idx < total_draft_elements,
                    "invalid last_accepted_retrive_idx at batch ", bx,
                    ": ", last_accepted_retrive_idx,
                    ", expected in [0, ", total_draft_elements, ")");
                const int64_t target_token_id = target_predict_ptr[last_accepted_retrive_idx];

                if (draft_token_id == target_token_id) {
                    predicts_ptr[last_accepted_retrive_idx] = static_cast<int32_t>(target_token_id);
                    ++num_accepted_tokens;
                    accept_index_ptr[bx * num_speculative_tokens + num_accepted_tokens] = static_cast<int32_t>(draft_index);
                    last_accepted_retrive_idx = draft_index;
                    break;
                }

                cur_index = retrive_next_sibling_ptr[bx * num_draft_tokens + cur_index];
                TORCH_CHECK(
                    cur_index == -1 || (cur_index >= 0 && cur_index < num_draft_tokens),
                    "invalid retrive_next_sibling index at batch ", bx,
                    ": ", cur_index,
                    ", expected in [-1, ", num_draft_tokens, ")");
            }

            if (cur_index == -1) {
                break;
            }
        }

        accept_token_num_ptr[bx] = num_accepted_tokens;
        TORCH_CHECK(
            last_accepted_retrive_idx >= 0 &&
                last_accepted_retrive_idx < total_draft_elements,
            "invalid final last_accepted_retrive_idx at batch ", bx,
            ": ", last_accepted_retrive_idx,
            ", expected in [0, ", total_draft_elements, ")");

        TORCH_CHECK(
            last_accepted_retrive_idx < target_predict_numel,
            "target_predict final access out of bounds: ", last_accepted_retrive_idx,
            " >= ", target_predict_numel);

        predicts_ptr[last_accepted_retrive_idx] = static_cast<int32_t>(target_predict_ptr[last_accepted_retrive_idx]);
      }
    });

    if (!predicts.is_contiguous()) {
        predicts.copy_(predicts_contig);
    }
    if (!accept_index.is_contiguous()) {
        accept_index.copy_(accept_index_contig);
    }
    if (!accept_token_num.is_contiguous()) {
        accept_token_num.copy_(accept_token_num_contig);
    }
}

// ---------------------------------------------------------------------------
// create_extend_after_decode_spec_info_cpu
// ---------------------------------------------------------------------------
// Python equivalent (serial):
//   accept_prefix = cumsum(accept_lens) - accept_lens
//   for pid in range(bs):
//       positions[base:base+accept_length] = arange(accept_length) + seq_length - accept_length
//       new_verified_id[pid] = verified_id[base + accept_length - 1]
//
void create_extend_after_decode_spec_info_cpu(
    at::Tensor verified_id,       // [total_accepted] int32
    at::Tensor seq_lens,          // [bs] int64
    at::Tensor accept_lens,       // [bs] int32
    at::Tensor positions,         // [total_accepted] int64, mutable
    at::Tensor new_verified_id    // [bs] int32, mutable
) {
    TORCH_CHECK(verified_id.device().is_cpu());
    TORCH_CHECK(seq_lens.device().is_cpu());
    TORCH_CHECK(accept_lens.device().is_cpu());
    TORCH_CHECK(positions.device().is_cpu());
    TORCH_CHECK(new_verified_id.device().is_cpu());

    const int64_t bs = seq_lens.size(0);
    if (bs == 0) return;

    auto seq_lens_contig = seq_lens.contiguous().to(at::kLong);
    auto accept_lens_contig = accept_lens.contiguous().to(at::kLong);
    auto verified_id_contig = verified_id.contiguous();
    auto positions_contig = positions.contiguous();
    auto new_verified_id_contig = new_verified_id.contiguous();

    const int64_t* seq_lens_ptr = seq_lens_contig.data_ptr<int64_t>();
    const int64_t* accept_lens_ptr = accept_lens_contig.data_ptr<int64_t>();
    const int32_t* verified_id_ptr = verified_id_contig.data_ptr<int32_t>();
    int64_t* positions_ptr = positions_contig.data_ptr<int64_t>();
    int32_t* new_verified_id_ptr = new_verified_id_contig.data_ptr<int32_t>();

    // Pre-compute prefix sums of accept_lens
    std::vector<int64_t> accept_prefix(bs);
    {
        int64_t running = 0;
        for (int64_t i = 0; i < bs; ++i) {
            accept_prefix[i] = running;
            running += accept_lens_ptr[i];
        }
    }

    at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
        for (int64_t pid = begin; pid < end; ++pid) {
            const int64_t seq_length = seq_lens_ptr[pid];
            const int64_t accept_length = accept_lens_ptr[pid];
            const int64_t base = accept_prefix[pid];
            if (accept_length > 0) {
                for (int64_t k = 0; k < accept_length; ++k) {
                    positions_ptr[base + k] = seq_length - accept_length + k;
                }
                new_verified_id_ptr[pid] = verified_id_ptr[base + accept_length - 1];
            }
        }
    });

    if (!positions.is_contiguous()) positions.copy_(positions_contig);
    if (!new_verified_id.is_contiguous()) new_verified_id.copy_(new_verified_id_contig);
}

// ---------------------------------------------------------------------------
// assign_req_to_token_pool_cpu
// ---------------------------------------------------------------------------
// Python equivalent (serial):
//   lengths = end_offset - start_offset
//   out_starts = cumsum(lengths) - lengths
//   for bid in range(bs):
//       req_to_token[req_pool_indices[bid], start:end] = out_cache_loc[src_start:src_end]
//
void assign_req_to_token_pool_cpu(
    at::Tensor req_pool_indices,  // [bs] int64
    at::Tensor req_to_token,      // [max_reqs, pool_len] int32 or int64, mutable
    at::Tensor start_offset,      // [bs] int64
    at::Tensor end_offset,        // [bs] int64
    at::Tensor out_cache_loc,     // [total] int64
    int64_t pool_len,
    int64_t bs
) {
    if (bs == 0) return;

    TORCH_CHECK(req_pool_indices.device().is_cpu());
    TORCH_CHECK(req_to_token.device().is_cpu());

    auto req_pool_indices_c = req_pool_indices.contiguous().to(at::kLong);
    auto start_offset_c = start_offset.contiguous().to(at::kLong);
    auto end_offset_c = end_offset.contiguous().to(at::kLong);
    // Unify req_to_token and out_cache_loc to int64 to avoid AT_DISPATCH issues
    auto req_to_token_i64 = req_to_token.contiguous().to(at::kLong);
    auto out_cache_loc_c = out_cache_loc.contiguous().to(at::kLong);

    const int64_t* req_pool_ptr = req_pool_indices_c.data_ptr<int64_t>();
    const int64_t* start_ptr = start_offset_c.data_ptr<int64_t>();
    const int64_t* end_ptr = end_offset_c.data_ptr<int64_t>();
    int64_t* req_to_token_ptr = req_to_token_i64.data_ptr<int64_t>();
    const int64_t* out_cache_ptr = out_cache_loc_c.data_ptr<int64_t>();

    // Pre-compute out_starts = cumsum(lengths) - lengths
    std::vector<int64_t> out_starts(bs);
    {
        int64_t running = 0;
        for (int64_t i = 0; i < bs; ++i) {
            out_starts[i] = running;
            running += (end_ptr[i] - start_ptr[i]);
        }
    }

    at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end_b) {
        for (int64_t bid = begin; bid < end_b; ++bid) {
            const int64_t req_idx = req_pool_ptr[bid];
            const int64_t kv_start = start_ptr[bid];
            const int64_t kv_end = end_ptr[bid];
            const int64_t copy_len = kv_end - kv_start;
            if (copy_len <= 0) continue;

            const int64_t src_start = out_starts[bid];
            int64_t* dst = req_to_token_ptr + req_idx * pool_len + kv_start;
            for (int64_t j = 0; j < copy_len; ++j) {
                dst[j] = out_cache_ptr[src_start + j];
            }
        }
    });

    // Write back to original tensor if dtype differs
    if (req_to_token.scalar_type() != at::kLong) {
        req_to_token.copy_(req_to_token_i64.to(req_to_token.scalar_type()));
    } else if (!req_to_token.is_contiguous()) {
        req_to_token.copy_(req_to_token_i64);
    }
}

// ---------------------------------------------------------------------------
// assign_draft_cache_locs_cpu
// ---------------------------------------------------------------------------
void assign_draft_cache_locs_cpu(
    at::Tensor req_pool_indices,      // [num_seqs] int64
    at::Tensor req_to_token,          // [max_reqs, pool_len] mutable
    at::Tensor seq_lens,              // [num_seqs] int64
    at::Tensor extend_lens,           // [num_seqs] int64 or int32
    at::Tensor num_new_pages_per_topk,// [num_seqs] int64 or int32
    at::Tensor out_cache_loc,         // mutable
    int64_t pool_len,
    int64_t topk,
    int64_t speculative_num_steps,
    int64_t page_size,
    int64_t num_seqs
) {
    if (num_seqs == 0) return;

    TORCH_CHECK(req_pool_indices.device().is_cpu());
    TORCH_CHECK(req_to_token.device().is_cpu());

    auto req_pool_c = req_pool_indices.contiguous().to(at::kLong);
    auto seq_lens_c = seq_lens.contiguous().to(at::kLong);
    auto extend_lens_c = extend_lens.contiguous().to(at::kLong);
    auto num_new_pages_c = num_new_pages_per_topk.contiguous().to(at::kLong);
    // Unify req_to_token and out_cache_loc to int64 to avoid AT_DISPATCH issues
    auto req_to_token_i64 = req_to_token.contiguous().to(at::kLong);
    auto out_cache_i64 = out_cache_loc.contiguous().to(at::kLong);

    const int64_t* req_pool_ptr = req_pool_c.data_ptr<int64_t>();
    const int64_t* seq_lens_ptr = seq_lens_c.data_ptr<int64_t>();
    const int64_t* extend_lens_ptr = extend_lens_c.data_ptr<int64_t>();
    const int64_t* num_new_pages_ptr = num_new_pages_c.data_ptr<int64_t>();
    int64_t* req_to_token_ptr = req_to_token_i64.data_ptr<int64_t>();
    int64_t* out_cache_ptr = out_cache_i64.data_ptr<int64_t>();

    if (page_size == 1 || topk == 1) {
        // Simple case: each batch copies topk * speculative_num_steps elements
        const int64_t row_copy_len = topk * speculative_num_steps;

        at::parallel_for(0, num_seqs, 1, [&](int64_t begin, int64_t end) {
            for (int64_t bid = begin; bid < end; ++bid) {
                const int64_t req_idx = req_pool_ptr[bid];
                const int64_t kv_start = seq_lens_ptr[bid];
                const int64_t src_start = bid * row_copy_len;
                int64_t* dst = req_to_token_ptr + req_idx * pool_len + kv_start;
                for (int64_t j = 0; j < row_copy_len; ++j) {
                    dst[j] = out_cache_ptr[src_start + j];
                }
            }
        });
    } else {
        // Complex case with page duplication

        // Pre-compute cum_copy_len prefix sums
        std::vector<int64_t> cum_copy_lens(num_seqs);
        {
            int64_t running = 0;
            for (int64_t i = 0; i < num_seqs; ++i) {
                cum_copy_lens[i] = running;
                running += extend_lens_ptr[i];
            }
        }

        at::parallel_for(0, num_seqs, 1, [&](int64_t begin, int64_t end) {
            for (int64_t bid = begin; bid < end; ++bid) {
                const int64_t req_idx = req_pool_ptr[bid];
                const int64_t prefix_len = seq_lens_ptr[bid];
                const int64_t copy_len = extend_lens_ptr[bid];
                const int64_t num_new_pages = num_new_pages_ptr[bid];

                // Part 1: Copy from out_cache_loc to req_to_token
                const int64_t src_start = cum_copy_lens[bid];
                int64_t* dst = req_to_token_ptr + req_idx * pool_len + prefix_len;
                for (int64_t j = 0; j < copy_len; ++j) {
                    dst[j] = out_cache_ptr[src_start + j];
                }

                // Part 2: Duplicate last partial page for each topk
                const int64_t last_page_len = prefix_len % page_size;
                const int64_t prefix_base = prefix_len - last_page_len;
                int64_t* token_row = req_to_token_ptr + req_idx * pool_len;

                for (int64_t topk_id = 0; topk_id < topk; ++topk_id) {
                    int64_t dst_start = prefix_base + topk_id * num_new_pages * page_size;
                    for (int64_t j = 0; j < last_page_len; ++j) {
                        token_row[dst_start + j] = token_row[prefix_base + j];
                    }
                }

                // Part 3: Remove padding in out_cache_loc
                for (int64_t topk_id = 0; topk_id < topk; ++topk_id) {
                    int64_t src_idx_start = prefix_base + topk_id * num_new_pages * page_size + last_page_len;
                    int64_t dst_idx_start = bid * topk * speculative_num_steps + topk_id * speculative_num_steps;
                    for (int64_t j = 0; j < speculative_num_steps; ++j) {
                        out_cache_ptr[dst_idx_start + j] = token_row[src_idx_start + j];
                    }
                }
            }
        });
    }

    // Write back to original tensors if dtype differs
    if (req_to_token.scalar_type() != at::kLong) {
        req_to_token.copy_(req_to_token_i64.to(req_to_token.scalar_type()));
    } else if (!req_to_token.is_contiguous()) {
        req_to_token.copy_(req_to_token_i64);
    }
    if (out_cache_loc.scalar_type() != at::kLong) {
        out_cache_loc.copy_(out_cache_i64.to(out_cache_loc.scalar_type()));
    } else if (!out_cache_loc.is_contiguous()) {
        out_cache_loc.copy_(out_cache_i64);
    }
}

// ---------------------------------------------------------------------------
// generate_draft_decode_kv_indices_cpu
// ---------------------------------------------------------------------------
void generate_draft_decode_kv_indices_cpu(
    at::Tensor req_pool_indices,   // [num_seqs] int64
    at::Tensor req_to_token,       // [max_reqs, pool_len]
    at::Tensor paged_kernel_lens,  // [num_seqs] int64
    at::Tensor kv_indices,         // [num_steps, kv_indices_stride] mutable
    at::Tensor kv_indptr,          // [num_steps, kv_indptr_stride] mutable
    at::Tensor positions,          // [num_seqs * topk] int64
    int64_t pool_len,
    int64_t kv_indices_stride,
    int64_t kv_indptr_stride,
    int64_t page_size
) {
    const int64_t num_seqs = req_pool_indices.size(0);
    if (num_seqs == 0) return;

    TORCH_CHECK(req_pool_indices.device().is_cpu());
    TORCH_CHECK(req_to_token.device().is_cpu());

    const int64_t topk = positions.size(0) / num_seqs;
    const int64_t num_steps = kv_indices.size(0);

    auto req_pool_c = req_pool_indices.contiguous().to(at::kLong);
    auto paged_lens_c = paged_kernel_lens.contiguous().to(at::kLong);
    auto positions_c = positions.contiguous().to(at::kLong);
    // Unify req_to_token to int64 to avoid AT_DISPATCH issues
    auto req_to_token_i64 = req_to_token.contiguous().to(at::kLong);

    const int64_t* req_pool_ptr = req_pool_c.data_ptr<int64_t>();
    const int64_t* paged_lens_ptr = paged_lens_c.data_ptr<int64_t>();
    const int64_t* positions_ptr = positions_c.data_ptr<int64_t>();
    const int64_t* req_to_token_ptr = req_to_token_i64.data_ptr<int64_t>();

    // Pre-compute seq_prefix = cumsum(paged_lens) - paged_lens
    std::vector<int64_t> seq_prefix(num_seqs);
    {
        int64_t running = 0;
        for (int64_t i = 0; i < num_seqs; ++i) {
            seq_prefix[i] = running;
            running += paged_lens_ptr[i];
        }
    }

    for (int64_t step = 0; step < num_steps; ++step) {
        const int64_t iters = step + 1;
        int32_t* kv_indices_step = kv_indices.data_ptr<int32_t>() + step * kv_indices_stride;
        int32_t* kv_indptr_step = kv_indptr.data_ptr<int32_t>() + step * kv_indptr_stride;

        // Zero out kv_indptr for this step
        std::memset(kv_indptr_step, 0, kv_indptr_stride * sizeof(int32_t));

        // Parallelize over (bid, topk_id) pairs
        at::parallel_for(0, num_seqs * topk, 1, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                const int64_t bid = idx / topk;
                const int64_t topk_id = idx % topk;

                const int64_t req_idx = req_pool_ptr[bid];
                const int64_t seq_len = paged_lens_ptr[bid];
                const int64_t cum_seq_len = seq_prefix[bid];
                const int64_t* token_pool_row = req_to_token_ptr + req_idx * pool_len;

                const int64_t kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters);

                // Copy prefix KV indices
                for (int64_t j = 0; j < seq_len; ++j) {
                    kv_indices_step[kv_offset + j] = static_cast<int32_t>(token_pool_row[j]);
                }

                // Copy extend KV indices
                int64_t ext_start;
                if (page_size == 1 || topk == 1) {
                    ext_start = seq_len + topk_id * num_steps;
                } else {
                    const int64_t last_page_len = seq_len % page_size;
                    const int64_t num_new_pages_per_topk_val = (last_page_len + num_steps + page_size - 1) / page_size;
                    const int64_t prefix_base = (seq_len / page_size) * page_size;
                    ext_start = prefix_base + topk_id * num_new_pages_per_topk_val * page_size + last_page_len;
                }

                for (int64_t j = 0; j < iters; ++j) {
                    kv_indices_step[kv_offset + seq_len + j] = static_cast<int32_t>(token_pool_row[ext_start + j]);
                }

                // Update kv_indptr
                const int64_t zid = bid * topk + topk_id;
                const int64_t zid_for_indptr = (zid == 0) ? (num_seqs * topk) : zid;
                int64_t base = 0;
                for (int64_t p = 0; p < zid_for_indptr; ++p) {
                    base += positions_ptr[p];
                }
                kv_indptr_step[zid_for_indptr] = static_cast<int32_t>(base + zid_for_indptr * iters);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// align_evict_mask_to_page_size_cpu
// ---------------------------------------------------------------------------
void align_evict_mask_to_page_size_cpu(
    at::Tensor seq_lens,           // [bs] int64
    at::Tensor evict_mask,         // [bs * num_draft_tokens] bool, mutable
    int64_t page_size,
    int64_t num_draft_tokens
) {
    const int64_t bs = seq_lens.size(0);
    if (bs == 0) return;

    TORCH_CHECK(seq_lens.device().is_cpu());
    TORCH_CHECK(evict_mask.device().is_cpu());

    auto seq_lens_c = seq_lens.contiguous().to(at::kLong);
    auto evict_mask_c = evict_mask.contiguous();

    const int64_t* seq_lens_ptr = seq_lens_c.data_ptr<int64_t>();
    bool* evict_mask_ptr = evict_mask_c.data_ptr<bool>();

    at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bid = begin; bid < end; ++bid) {
            const int64_t seq_len = seq_lens_ptr[bid];
            const int64_t row_start = bid * num_draft_tokens;
            bool* row = evict_mask_ptr + row_start;

            // Count true values
            int64_t num_trues = 0;
            for (int64_t j = 0; j < num_draft_tokens; ++j) {
                if (row[j]) ++num_trues;
            }
            const int64_t num_false = num_draft_tokens - num_trues;
            const int64_t start = (seq_len + num_false - 1) / page_size * page_size - seq_len;
            const int64_t range_begin = std::max(start, int64_t(0));
            const int64_t range_end = std::min(start + page_size, num_draft_tokens);
            for (int64_t j = range_begin; j < range_end; ++j) {
                row[j] = false;
            }
        }
    });

    if (!evict_mask.is_contiguous()) evict_mask.copy_(evict_mask_c);
}

// ---------------------------------------------------------------------------
// get_target_cache_loc_cpu
// ---------------------------------------------------------------------------
void get_target_cache_loc_cpu(
    at::Tensor tgt_cache_loc,      // mutable
    at::Tensor to_free_slots,      // mutable
    at::Tensor accept_length,      // [bs] int32
    at::Tensor to_free_num_slots,  // [bs]
    at::Tensor out_cache_loc,      // [bs * num_verify_tokens]
    int64_t num_verify_tokens,
    int64_t bs
) {
    if (bs == 0) return;

    TORCH_CHECK(accept_length.device().is_cpu());
    TORCH_CHECK(out_cache_loc.device().is_cpu());

    auto accept_len_c = accept_length.contiguous().to(at::kLong);
    auto to_free_num_c = to_free_num_slots.contiguous().to(at::kLong);
    auto out_cache_c = out_cache_loc.contiguous();
    auto tgt_cache_c = tgt_cache_loc.contiguous();
    auto to_free_c = to_free_slots.contiguous();

    const int64_t* accept_len_ptr = accept_len_c.data_ptr<int64_t>();
    const int64_t* to_free_num_ptr = to_free_num_c.data_ptr<int64_t>();
    const int64_t* out_cache_ptr = out_cache_c.data_ptr<int64_t>();
    int64_t* tgt_cache_ptr = tgt_cache_c.data_ptr<int64_t>();
    int64_t* to_free_ptr = to_free_c.data_ptr<int64_t>();

    // Pre-compute prefix sums
    std::vector<int64_t> accept_prefix(bs);
    std::vector<int64_t> free_prefix(bs);
    {
        int64_t accept_running = 0;
        int64_t free_running = 0;
        for (int64_t i = 0; i < bs; ++i) {
            accept_prefix[i] = accept_running;
            free_prefix[i] = free_running;
            accept_running += accept_len_ptr[i];
            free_running += to_free_num_ptr[i];
        }
    }

    at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bid = begin; bid < end; ++bid) {
            // Part 1: Copy accepted tokens to tgt_cache_loc
            const int64_t copy_len = accept_len_ptr[bid] + 1;
            const int64_t out_row_start = bid * num_verify_tokens;
            const int64_t tgt_start = accept_prefix[bid] + bid;
            for (int64_t j = 0; j < copy_len; ++j) {
                tgt_cache_ptr[tgt_start + j] = out_cache_ptr[out_row_start + j];
            }

            // Part 2: Copy free slots
            const int64_t to_free_cur = to_free_num_ptr[bid];
            if (to_free_cur <= 0) continue;
            const int64_t out_free_start = out_row_start + (num_verify_tokens - to_free_cur);
            const int64_t free_start = free_prefix[bid];
            for (int64_t j = 0; j < to_free_cur; ++j) {
                to_free_ptr[free_start + j] = out_cache_ptr[out_free_start + j];
            }
        }
    });

    if (!tgt_cache_loc.is_contiguous()) tgt_cache_loc.copy_(tgt_cache_c);
    if (!to_free_slots.is_contiguous()) to_free_slots.copy_(to_free_c);
}

// ---------------------------------------------------------------------------
// filter_finished_cache_loc_cpu
// ---------------------------------------------------------------------------
void filter_finished_cache_loc_cpu(
    at::Tensor out_cache_loc,          // mutable
    at::Tensor tgt_cache_loc,          // input
    at::Tensor accept_length,          // [bs] int32
    at::Tensor accept_length_filter,   // [bs] int64
    int64_t bs,
    int64_t draft_token_num
) {
    if (bs == 0) return;

    TORCH_CHECK(out_cache_loc.device().is_cpu());
    TORCH_CHECK(tgt_cache_loc.device().is_cpu());

    auto accept_len_c = accept_length.contiguous().to(at::kLong);
    auto filter_c = accept_length_filter.contiguous().to(at::kLong);
    auto tgt_c = tgt_cache_loc.contiguous();
    auto out_c = out_cache_loc.contiguous();

    const int64_t* accept_len_ptr = accept_len_c.data_ptr<int64_t>();
    const int64_t* filter_ptr = filter_c.data_ptr<int64_t>();
    const int64_t* tgt_ptr = tgt_c.data_ptr<int64_t>();
    int64_t* out_ptr = out_c.data_ptr<int64_t>();

    // Pre-compute prefix sums
    std::vector<int64_t> accept_prefix(bs);
    std::vector<int64_t> filter_prefix(bs);
    {
        int64_t accept_running = 0;
        int64_t filter_running = 0;
        for (int64_t i = 0; i < bs; ++i) {
            accept_prefix[i] = accept_running;
            filter_prefix[i] = filter_running;
            accept_running += accept_len_ptr[i];
            filter_running += filter_ptr[i];
        }
    }

    at::parallel_for(0, bs, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bid = begin; bid < end; ++bid) {
            const int64_t old_start = accept_prefix[bid] + bid;
            const int64_t new_start = filter_prefix[bid];
            const int64_t copy_len = filter_ptr[bid];
            if (copy_len <= 0) continue;
            for (int64_t j = 0; j < copy_len; ++j) {
                out_ptr[new_start + j] = tgt_ptr[old_start + j];
            }
        }
    });

    if (!out_cache_loc.is_contiguous()) out_cache_loc.copy_(out_c);
}

