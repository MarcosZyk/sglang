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
  std::printf("note: idleness table uses active threads only (single_thread.sum_ns > 0).\n");

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
  std::vector<uint64_t> active_thread_ns;
  active_thread_ns.reserve(static_cast<size_t>(std::max(0, s.num_threads)));
  uint64_t st_min_ns = 0;
  uint64_t st_max_ns = 0;
  long double st_sum_ns = 0.0L;
  long double st_sum_sq_ns = 0.0L;
  for (int tid = 0; tid < s.num_threads; ++tid) {
    const uint64_t v_ns = s.per_thread[tid][single_thread_stage_id].sum_ns;
    if (v_ns == 0) {
      continue;
    }
    if (active_thread_ns.empty()) {
      st_min_ns = v_ns;
      st_max_ns = v_ns;
    } else {
      st_min_ns = std::min(st_min_ns, v_ns);
      st_max_ns = std::max(st_max_ns, v_ns);
    }
    active_thread_ns.push_back(v_ns);
    st_sum_ns += static_cast<long double>(v_ns);
    st_sum_sq_ns += static_cast<long double>(v_ns) * static_cast<long double>(v_ns);
  }
  const uint64_t st_active_threads = static_cast<uint64_t>(active_thread_ns.size());
  const double st_avg_ns =
      st_active_threads > 0 ? static_cast<double>(st_sum_ns / static_cast<long double>(st_active_threads)) : 0.0;
  const double st_std_ns = pop_std_from_sum_sq(st_avg_ns, st_sum_sq_ns, st_active_threads);
  double st_median_ns = 0.0;
  if (!active_thread_ns.empty()) {
    std::sort(active_thread_ns.begin(), active_thread_ns.end());
    const size_t n = active_thread_ns.size();
    if (n % 2 == 1) {
      st_median_ns = static_cast<double>(active_thread_ns[n / 2]);
    } else {
      st_median_ns =
          0.5 * (static_cast<double>(active_thread_ns[n / 2 - 1]) + static_cast<double>(active_thread_ns[n / 2]));
    }
  }
  const double st_idle_ratio_pct = st_max_ns > 0
      ? (100.0 * (static_cast<double>(st_max_ns) - st_avg_ns) / static_cast<double>(st_max_ns))
      : 0.0;

  std::printf(
      "%-20s | %16s | %16s | %16s | %16s | %16s\n",
      "stage",
      "invocation_times",
      "avg_duration_us",
      "sum_duration_us",
      "cpu_time_us",
      "cpu_proportion_%");
  std::printf("----------------------------------------------------------------------------------------------------------------\n");
  for (int i = 0; i < kNumStages; ++i) {
    const auto cnt = merged[i].count;
    const auto sum_ns = merged[i].sum_ns;
    const double sum_us = static_cast<double>(sum_ns) / 1000.0;
    const double avg_us = cnt > 0 ? (sum_us / static_cast<double>(cnt)) : 0.0;
    const double cpu_time_us =
        is_global_timed_stage(i) ? (sum_us * static_cast<double>(thread_num)) : sum_us;
    const double cpu_proportion_pct = total_cpu_time_us > 0.0 ? (100.0 * cpu_time_us / total_cpu_time_us) : 0.0;

    std::printf(
        "%-20.*s | %16llu | %16.3f | %16.3f | %16.3f | %16.3f\n",
        static_cast<int>(stage_name(static_cast<Stage>(i)).size()),
        stage_name(static_cast<Stage>(i)).data(),
        static_cast<unsigned long long>(cnt),
        avg_us,
        sum_us,
        cpu_time_us,
        cpu_proportion_pct);
  }

  const int real_thread_total_stage_id = static_cast<int>(Stage::kRealThreadTotal);
  const double st_min_us = static_cast<double>(st_min_ns) / 1000.0;
  const double st_median_us = st_median_ns / 1000.0;
  const double st_max_us = static_cast<double>(st_max_ns) / 1000.0;
  const double st_avg_us = st_avg_ns / 1000.0;
  const double st_std_us = st_std_ns / 1000.0;
  const double real_thread_total_cpu_time_us =
      static_cast<double>(merged[real_thread_total_stage_id].sum_ns) / 1000.0 * static_cast<double>(thread_num);
  const double total_cpu_time_bound_us = st_max_us * static_cast<double>(thread_num);
  const double bound_gap_us = real_thread_total_cpu_time_us - total_cpu_time_bound_us;
  const double bound_utilization_pct =
      real_thread_total_cpu_time_us > 0.0 ? (100.0 * total_cpu_time_bound_us / real_thread_total_cpu_time_us) : 0.0;

  std::printf("\nDecode Timer Idleness\n");
  std::printf("%-30s | %16s | %-40s\n", "metric", "value_us", "note");
  std::printf("----------------------------------------------------------------------------------------------\n");
  std::printf("%-30s | %16.3f | %-40s\n", "thread_cpu_min", st_min_us, "active single_thread totals");
  std::printf("%-30s | %16.3f | %-40s\n", "thread_cpu_median", st_median_us, "active single_thread totals");
  std::printf("%-30s | %16.3f | %-40s\n", "thread_cpu_max", st_max_us, "active single_thread totals");
  std::printf("%-30s | %16.3f | %-40s\n", "thread_cpu_avg", st_avg_us, "active single_thread totals");
  std::printf("%-30s | %16.3f | %-40s\n", "thread_cpu_std", st_std_us, "population std");
  std::printf("%-30s | %16.3f | %-40s\n", "idleness_%", st_idle_ratio_pct, "(max-avg)/max*100");
  std::printf(
      "%-30s | %16.3f | %-40s\n",
      "configured_thread_num",
      static_cast<double>(thread_num),
      "thread count (not us)");
  std::printf(
      "%-30s | %16.3f | %-40s\n",
      "total_cpu_time_bound_us",
      total_cpu_time_bound_us,
      "thread_cpu_max * configured_thread_num");
  std::printf(
      "%-30s | %16.3f | %-40s\n",
      "real_thread_total_cpu_time_us",
      real_thread_total_cpu_time_us,
      "real_thread_total sum_us * configured_thread_num");
  std::printf("%-30s | %16.3f | %-40s\n", "bound_gap_us", bound_gap_us, "real_thread_total_cpu_time - bound");
  std::printf("%-30s | %16.3f | %-40s\n", "bound_utilization_%", bound_utilization_pct, "bound / real * 100");

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
