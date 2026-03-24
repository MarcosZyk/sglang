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

struct OnlineNsStats {
  uint64_t count{0};
  uint64_t sum_ns{0};
  long double sum_sq_ns{0.0L};
  uint64_t min_ns{0};
  uint64_t max_ns{0};
};

struct TimerState {
  bool active{false};
  int num_threads{0};
  std::vector<std::array<StageAccum, kNumStages>> per_thread;
  uint64_t profiled_calls{0};
  uint64_t skipped_calls{0};
  OnlineNsStats real_thread_total_inv_ns{};
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
  s.real_thread_total_inv_ns = OnlineNsStats{};
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
  const uint64_t kernel_inv = merged[kernel_stage_id].count;
  const double kernel_sum_us = static_cast<double>(merged[kernel_stage_id].sum_ns) / 1000.0;
  const int64_t thread_num = std::max<int64_t>(1, at::get_num_threads());
  const double total_cpu_time_us = kernel_sum_us * static_cast<double>(thread_num);

  auto is_global_timed_stage = [](int stage_id) {
    const Stage stage = static_cast<Stage>(stage_id);
    return stage == Stage::kKernelImplTotal || stage == Stage::kSetKvBuffer || stage == Stage::kRealThreadTotal ||
        stage == Stage::kLogitsAccum;
  };

  std::printf("\nDecode Timer Statistics\n");
  std::printf(
      "note: thread_num=%lld, stage_cpu_policy={global_timed:*thread_num, thread_local:raw}, "
      "total_cpu_time_us=kernel_impl_total_sum_us*thread_num\n",
      static_cast<long long>(thread_num));
  if (kernel_inv == 0) {
    std::printf("warning: kernel_impl_total invocation_times is 0, cpu_proportion_%% will be 0.\n");
  }
  std::printf(
      "note: single_thread min/max/std are across active threads; real_thread_total min/max/std are across "
      "invocations.\n");

  auto pop_std_from_sum_sq = [](double avg, long double sum_sq, uint64_t n) {
    if (n == 0) {
      return 0.0;
    }
    long double var = (sum_sq / static_cast<long double>(n)) - static_cast<long double>(avg) * static_cast<long double>(avg);
    if (var < 0.0L) {
      var = 0.0L;
    }
    return std::sqrt(static_cast<double>(var));
  };

  const int single_thread_stage_id = static_cast<int>(Stage::kSingleThread);
  uint64_t st_active_threads = 0;
  uint64_t st_min_ns = 0;
  uint64_t st_max_ns = 0;
  long double st_sum_ns = 0.0L;
  long double st_sum_sq_ns = 0.0L;
  for (int tid = 0; tid < s.num_threads; ++tid) {
    const uint64_t v_ns = s.per_thread[tid][single_thread_stage_id].sum_ns;
    if (v_ns == 0) {
      continue;
    }
    if (st_active_threads == 0) {
      st_min_ns = v_ns;
      st_max_ns = v_ns;
    } else {
      st_min_ns = std::min(st_min_ns, v_ns);
      st_max_ns = std::max(st_max_ns, v_ns);
    }
    st_active_threads += 1;
    st_sum_ns += static_cast<long double>(v_ns);
    st_sum_sq_ns += static_cast<long double>(v_ns) * static_cast<long double>(v_ns);
  }
  const double st_avg_ns = st_active_threads > 0 ? static_cast<double>(st_sum_ns / static_cast<long double>(st_active_threads)) : 0.0;
  const double st_std_ns = pop_std_from_sum_sq(st_avg_ns, st_sum_sq_ns, st_active_threads);
  const double st_idle_ratio_pct = st_max_ns > 0
      ? (100.0 * (static_cast<double>(st_max_ns) - st_avg_ns) / static_cast<double>(st_max_ns))
      : 0.0;

  const auto& rt_stats = s.real_thread_total_inv_ns;
  const double rt_avg_ns = rt_stats.count > 0 ? static_cast<double>(rt_stats.sum_ns) / static_cast<double>(rt_stats.count) : 0.0;
  const double rt_std_ns = pop_std_from_sum_sq(rt_avg_ns, rt_stats.sum_sq_ns, rt_stats.count);

  std::printf(
      "%-20s | %16s | %16s | %16s | %16s | %16s | %16s | %16s | %16s | %16s\n",
      "stage",
      "invocation_times",
      "avg_duration_us",
      "sum_duration_us",
      "cpu_time_us",
      "cpu_proportion_%",
      "min_duration_us",
      "max_duration_us",
      "std_duration_us",
      "idle_ratio_%");
  std::printf("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  for (int i = 0; i < kNumStages; ++i) {
    const auto cnt = merged[i].count;
    const auto sum_ns = merged[i].sum_ns;
    const double sum_us = static_cast<double>(sum_ns) / 1000.0;
    const double avg_us = cnt > 0 ? (sum_us / static_cast<double>(cnt)) : 0.0;
    const double cpu_time_us =
        is_global_timed_stage(i) ? (sum_us * static_cast<double>(thread_num)) : sum_us;
    const double cpu_proportion_pct = total_cpu_time_us > 0.0 ? (100.0 * cpu_time_us / total_cpu_time_us) : 0.0;

    char min_us_buf[32];
    char max_us_buf[32];
    char std_us_buf[32];
    char idle_ratio_buf[32];
    std::snprintf(min_us_buf, sizeof(min_us_buf), "%s", "-");
    std::snprintf(max_us_buf, sizeof(max_us_buf), "%s", "-");
    std::snprintf(std_us_buf, sizeof(std_us_buf), "%s", "-");
    std::snprintf(idle_ratio_buf, sizeof(idle_ratio_buf), "%s", "-");

    if (i == single_thread_stage_id) {
      std::snprintf(min_us_buf, sizeof(min_us_buf), "%.3f", static_cast<double>(st_min_ns) / 1000.0);
      std::snprintf(max_us_buf, sizeof(max_us_buf), "%.3f", static_cast<double>(st_max_ns) / 1000.0);
      std::snprintf(std_us_buf, sizeof(std_us_buf), "%.3f", st_std_ns / 1000.0);
      std::snprintf(idle_ratio_buf, sizeof(idle_ratio_buf), "%.3f", st_idle_ratio_pct);
    } else if (i == static_cast<int>(Stage::kRealThreadTotal)) {
      std::snprintf(min_us_buf, sizeof(min_us_buf), "%.3f", static_cast<double>(rt_stats.min_ns) / 1000.0);
      std::snprintf(max_us_buf, sizeof(max_us_buf), "%.3f", static_cast<double>(rt_stats.max_ns) / 1000.0);
      std::snprintf(std_us_buf, sizeof(std_us_buf), "%.3f", rt_std_ns / 1000.0);
    }

    std::printf(
        "%-20.*s | %16llu | %16.3f | %16.3f | %16.3f | %16.3f | %16s | %16s | %16s | %16s\n",
        static_cast<int>(stage_name(static_cast<Stage>(i)).size()),
        stage_name(static_cast<Stage>(i)).data(),
        static_cast<unsigned long long>(cnt),
        avg_us,
        sum_us,
        cpu_time_us,
        cpu_proportion_pct,
        min_us_buf,
        max_us_buf,
        std_us_buf,
        idle_ratio_buf);
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
  if (stage == Stage::kRealThreadTotal) {
    auto& rt = s.real_thread_total_inv_ns;
    if (rt.count == 0) {
      rt.min_ns = duration_ns;
      rt.max_ns = duration_ns;
    } else {
      rt.min_ns = std::min(rt.min_ns, duration_ns);
      rt.max_ns = std::max(rt.max_ns, duration_ns);
    }
    rt.count += 1;
    rt.sum_ns += duration_ns;
    rt.sum_sq_ns += static_cast<long double>(duration_ns) * static_cast<long double>(duration_ns);
  }
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
