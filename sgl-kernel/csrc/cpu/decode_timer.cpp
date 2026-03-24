#include "decode_timer.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <chrono>
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

  std::printf("\nDecode Timer Statistics\n");
  std::printf("%-20s | %16s | %16s | %16s\n", "stage", "invocation_times", "avg_duration_us", "sum_duration_us");
  std::printf("--------------------------------------------------------------------------------------\n");
  for (int i = 0; i < kNumStages; ++i) {
    const auto cnt = merged[i].count;
    const auto sum_ns = merged[i].sum_ns;
    const double sum_us = static_cast<double>(sum_ns) / 1000.0;
    const double avg_us = cnt > 0 ? (sum_us / static_cast<double>(cnt)) : 0.0;
    std::printf(
        "%-20.*s | %16llu | %16.3f | %16.3f\n",
        static_cast<int>(stage_name(static_cast<Stage>(i)).size()),
        stage_name(static_cast<Stage>(i)).data(),
        static_cast<unsigned long long>(cnt),
        avg_us,
        sum_us);
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
    case Stage::kThreadTotal:
      return "thread_total";
    case Stage::kCount:
      return "count";
  }
  return "unknown";
}

}  // namespace decode_timer

void decode_timer_start(at::Tensor& placeholder, bool reset) {
  UNUSED(placeholder);
  decode_timer::start(reset);
}

void decode_timer_stop_and_print(at::Tensor& placeholder) {
  UNUSED(placeholder);
  decode_timer::stop_and_print();
}
