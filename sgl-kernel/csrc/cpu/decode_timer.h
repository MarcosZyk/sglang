#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace decode_timer {

enum class Stage : int {
  kKernelImplTotal = 0,
  kSetKvBuffer = 1,
  kKvPackMla = 2,
  kKvPackGqa = 3,
  kAttnCompute = 4,
  kLogitsAccum = 5,
  kThreadTotal = 6,
  kCount = 7,
};

bool is_active();
void start(bool reset);
void stop_and_print();

void mark_profiled_call();
void mark_skipped_call();

void record_stage_duration(int tid, Stage stage, uint64_t duration_ns);
uint64_t now_ns();
std::string_view stage_name(Stage stage);

template <bool kEnableTimer>
class ScopedStageTimer {
 public:
  ScopedStageTimer(Stage stage, int tid) : stage_(stage), tid_(tid), start_ns_(0) {
    if constexpr (kEnableTimer) {
      start_ns_ = now_ns();
    }
  }

  ~ScopedStageTimer() {
    if constexpr (kEnableTimer) {
      uint64_t end_ns = now_ns();
      record_stage_duration(tid_, stage_, end_ns - start_ns_);
    }
  }

 private:
  Stage stage_;
  int tid_;
  uint64_t start_ns_;
};

}  // namespace decode_timer
