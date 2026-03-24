#include "decode_timer.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace decode_timer {

namespace {

constexpr int kNumStages = static_cast<int>(Stage::kCount);

struct StageAccum {
  uint64_t count{0};
  uint64_t sum_ns{0};
};

struct TimerState {
  bool active{false};
  int num_threads{0};
  std::vector<std::array<StageAccum, kNumStages>> per_thread;
  uint64_t profiled_calls{0};
  uint64_t skipped_calls{0};
};

TimerState& state() {
  static TimerState s;
  return s;
}

void reset_stats(TimerState& s) {
  for (auto& thread_stats : s.per_thread) {
    for (auto& stage : thread_stats) {
      stage.count = 0;
      stage.sum_ns = 0;
    }
  }
  s.profiled_calls = 0;
  s.skipped_calls = 0;
}

}  // namespace

bool is_active() {
  return state().active;
}

void start(bool reset) {
  auto& s = state();
  const int nt = std::max(1, at::get_num_threads());
  if (s.num_threads != nt || static_cast<int>(s.per_thread.size()) != nt) {
    s.num_threads = nt;
    s.per_thread.assign(nt, std::array<StageAccum, kNumStages>{});
    reset = true;
  }
  if (reset) {
    reset_stats(s);
  }
  s.active = true;
}

void stop_and_print() {
  auto& s = state();
  s.active = false;

  std::array<StageAccum, kNumStages> merged{};
  for (const auto& thread_stats : s.per_thread) {
    for (int i = 0; i < kNumStages; ++i) {
      merged[i].count += thread_stats[i].count;
      merged[i].sum_ns += thread_stats[i].sum_ns;
    }
  }

  const int kernel_stage_id = static_cast<int>(Stage::kKernelImplTotal);
  const int single_thread_stage_id = static_cast<int>(Stage::kSingleThread);
  const uint64_t kernel_inv = merged[kernel_stage_id].count;
  const uint64_t single_thread_inv = merged[single_thread_stage_id].count;
  const double kernel_sum_us = static_cast<double>(merged[kernel_stage_id].sum_ns) / 1000.0;

  double derived_thread_num = 0.0;
  bool has_valid_thread_ratio = false;
  bool ratio_is_integer = false;
  if (kernel_inv > 0) {
    derived_thread_num = static_cast<double>(single_thread_inv) / static_cast<double>(kernel_inv);
    has_valid_thread_ratio = derived_thread_num > 0.0;
    if (has_valid_thread_ratio) {
      const double rounded = std::round(derived_thread_num);
      ratio_is_integer = std::abs(derived_thread_num - rounded) < 1e-6;
    }
  }

  std::printf("\nDecode Timer Statistics\n");
  std::printf(
      "note: denominator=kernel_impl_total(raw_sum_us), derived_thread_num=%.3f, transform: "
      "if stage_inv==kernel_inv => linear=raw; if stage_inv>=single_thread_inv => linear=raw/derived_thread_num; else linear=raw\n",
      derived_thread_num);
  if (kernel_inv == 0) {
    std::printf("warning: kernel_impl_total invocation_times is 0, proportion_%% will be 0.\n");
  }
  if (kernel_inv > 0 && !ratio_is_integer) {
    std::printf(
        "warning: single_thread/kernel_impl_total is non-integer (single_thread_inv=%llu, kernel_inv=%llu, ratio=%.6f).\n",
        static_cast<unsigned long long>(single_thread_inv),
        static_cast<unsigned long long>(kernel_inv),
        derived_thread_num);
  }
  if (kernel_inv > 0 && single_thread_inv % kernel_inv != 0) {
    std::printf(
        "warning: non-divisible invocation ratio detected, transformation uses floating ratio fallback.\n");
  }

  // Thread-local average stage proportion:
  // For each active thread t (single_thread_sum_ns(t) > 0), compute
  // p_t(stage) = stage_sum_ns(t) / single_thread_sum_ns(t), then average p_t.
  std::array<double, kNumStages> thread_prop_sum{};
  uint64_t num_active_threads = 0;
  for (const auto& thread_stats : s.per_thread) {
    const uint64_t denom_ns = thread_stats[single_thread_stage_id].sum_ns;
    if (denom_ns == 0) {
      continue;
    }
    num_active_threads += 1;
    for (int i = 0; i < kNumStages; ++i) {
      thread_prop_sum[i] += static_cast<double>(thread_stats[i].sum_ns) / static_cast<double>(denom_ns);
    }
  }
  std::array<double, kNumStages> avg_thread_prop_pct{};
  if (num_active_threads > 0) {
    for (int i = 0; i < kNumStages; ++i) {
      avg_thread_prop_pct[i] = 100.0 * thread_prop_sum[i] / static_cast<double>(num_active_threads);
    }
  }

  auto is_parallel_stage = [](int stage_id) {
    const Stage stage = static_cast<Stage>(stage_id);
    return stage == Stage::kKvPackMla || stage == Stage::kKvPackGqa || stage == Stage::kAttnCompute ||
        stage == Stage::kSingleThread;
  };

  std::printf(
      "%-20s | %16s | %16s | %16s | %16s | %12s | %16s\n",
      "stage",
      "invocation_times",
      "avg_duration_us",
      "sum_duration_us",
      "linear_sum_us",
      "proportion_%",
      "avg_thread_prop_%");
  std::printf("--------------------------------------------------------------------------------------------------------------------------------\n");
  for (int i = 0; i < kNumStages; ++i) {
    const auto cnt = merged[i].count;
    const auto sum_ns = merged[i].sum_ns;
    const double sum_us = static_cast<double>(sum_ns) / 1000.0;
    const double avg_us = cnt > 0 ? (sum_us / static_cast<double>(cnt)) : 0.0;
    double linear_sum_us = sum_us;
    if (cnt != kernel_inv && cnt >= single_thread_inv && has_valid_thread_ratio) {
      linear_sum_us = sum_us / derived_thread_num;
    }
    const double proportion_pct = kernel_sum_us > 0.0 ? (100.0 * linear_sum_us / kernel_sum_us) : 0.0;
    char avg_thread_prop_buf[32];
    if (is_parallel_stage(i)) {
      std::snprintf(avg_thread_prop_buf, sizeof(avg_thread_prop_buf), "%.3f", avg_thread_prop_pct[i]);
    } else {
      std::snprintf(avg_thread_prop_buf, sizeof(avg_thread_prop_buf), "-");
    }

    std::printf(
        "%-20.*s | %16llu | %16.3f | %16.3f | %16.3f | %12.3f | %16s\n",
        static_cast<int>(stage_name(static_cast<Stage>(i)).size()),
        stage_name(static_cast<Stage>(i)).data(),
        static_cast<unsigned long long>(cnt),
        avg_us,
        sum_us,
        linear_sum_us,
        proportion_pct,
        avg_thread_prop_buf);
  }
  if (s.skipped_calls > 0) {
    std::printf("skipped_calls: %llu\n", static_cast<unsigned long long>(s.skipped_calls));
  }
  std::printf("profiled_calls: %llu\n", static_cast<unsigned long long>(s.profiled_calls));
}

void mark_profiled_call() {
  state().profiled_calls += 1;
}

void mark_skipped_call() {
  state().skipped_calls += 1;
}

void record_stage_duration(int tid, Stage stage, uint64_t duration_ns) {
  auto& s = state();
  if (tid < 0 || tid >= s.num_threads || s.per_thread.empty()) {
    return;
  }
  const int stage_id = static_cast<int>(stage);
  if (stage_id < 0 || stage_id >= kNumStages) {
    return;
  }
  auto& slot = s.per_thread[tid][stage_id];
  slot.count += 1;
  slot.sum_ns += duration_ns;
}

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::string_view stage_name(Stage stage) {
  switch (stage) {
    case Stage::kKernelImplTotal:
      return "kernel_impl_total";
    case Stage::kSetKvBuffer:
      return "set_kv_buffer";
    case Stage::kKvPackMla:
      return "kv_pack_mla";
    case Stage::kKvPackGqa:
      return "kv_pack_gqa";
    case Stage::kAttnCompute:
      return "attn_compute";
    case Stage::kLogitsAccum:
      return "logits_accum";
    case Stage::kSingleThread:
      return "single_thread";
    case Stage::kRealThreadTotal:
      return "real_thread_total";
    case Stage::kCount:
      return "count";
  }
  return "unknown";
}

}  // namespace decode_timer

void decode_timer_start(at::Tensor& placeholder, bool reset) {
  (void)placeholder;
  decode_timer::start(reset);
}

void decode_timer_stop_and_print(at::Tensor& placeholder) {
  (void)placeholder;
  decode_timer::stop_and_print();
}
