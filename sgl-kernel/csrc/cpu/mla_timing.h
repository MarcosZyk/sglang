// mla_timing.h

#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <fstream>

#include "common.h"

namespace mla_timing {

// ========== 1. 时钟配置 ==========
using Clock = std::chrono::high_resolution_clock;
using TickType = uint64_t;

struct ClockConfig {
    static double ticks_to_us_factor;
    static void init() {
        using period = Clock::period;
        ticks_to_us_factor = static_cast<double>(period::num) / static_cast<double>(period::den) * 1e6;
    }
};
double ClockConfig::ticks_to_us_factor = 0.0;

// ========== 2. 前向声明 ==========
class GlobalManager;  // ← 关键：前向声明

// ========== 3. 线程本地统计 ==========
struct ThreadLocalStats {
    std::vector<TickType> accum_ticks;
    std::vector<int64_t> counts;
    std::vector<TickType> start_ticks;
    bool is_registered = false;

    ThreadLocalStats();   // 只声明，定义后置
    ~ThreadLocalStats();  // 只声明，定义后置

    void ensure_size(int num_stages) {
        if (accum_ticks.size() < static_cast<size_t>(num_stages)) {
            accum_ticks.resize(num_stages, 0);
            counts.resize(num_stages, 0);
            start_ticks.resize(num_stages, 0);
        }
    }
};

// 声明 thread_local 实例
extern thread_local ThreadLocalStats g_tls_stats;

// ========== 4. 后端接口 ==========
struct TimingBackend {
    void (*start)(int stage_id);
    void (*end)(int stage_id);
};

// ========== 5. 后端实现 ==========
namespace backend_impl {
    inline void enabled_start(int stage_id) {
        g_tls_stats.ensure_size(stage_id + 1);
        g_tls_stats.start_ticks[stage_id] = Clock::now().time_since_epoch().count();
    }

    inline void enabled_end(int stage_id) {
        auto& tls = g_tls_stats;
        if (stage_id >= static_cast<int>(tls.start_ticks.size())) return;

        TickType end = Clock::now().time_since_epoch().count();
        TickType elapsed = end - tls.start_ticks[stage_id];

        if (stage_id >= static_cast<int>(tls.accum_ticks.size()))
            tls.accum_ticks.resize(stage_id + 1, 0);
        if (stage_id >= static_cast<int>(tls.counts.size()))
            tls.counts.resize(stage_id + 1, 0);

        tls.accum_ticks[stage_id] += elapsed;
        tls.counts[stage_id]++;
    }

    inline void disabled_start(int) {}
    inline void disabled_end(int) {}
}

// 全局后端指针
extern TimingBackend g_current_backend;

// ========== 6. 全局管理器（完整定义） ==========
class GlobalManager {
public:
    static GlobalManager& instance() {
        static GlobalManager inst;
        return inst;
    }

    int register_stage(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stage_map_.find(name);
        if (it != stage_map_.end()) return it->second;

        int id = static_cast<int>(stage_names_.size());
        stage_map_[name] = id;
        stage_names_.push_back(name);
        return id;
    }

    void register_thread(ThreadLocalStats* stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stats->is_registered) {
            threads_.push_back(stats);
            stats->is_registered = true;
        }
    }

    void unregister_thread(ThreadLocalStats* stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        threads_.erase(std::remove(threads_.begin(), threads_.end(), stats), threads_.end());
        stats->is_registered = false;
    }

    void set_enabled(bool enabled) {
        if (enabled) {
            g_current_backend = {backend_impl::enabled_start, backend_impl::enabled_end};
        } else {
            g_current_backend = {backend_impl::disabled_start, backend_impl::disabled_end};
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto* t : threads_) {
            std::fill(t->accum_ticks.begin(), t->accum_ticks.end(), 0);
            std::fill(t->counts.begin(), t->counts.end(), 0);
        }
    }

    void export_to_file(const std::string& filepath) {
        if (ClockConfig::ticks_to_us_factor == 0.0) ClockConfig::init();

        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "[MLA Timing] Failed to open " << filepath << "\n";
            return;
        }

        ofs << "stage,count,avg_us,min_us,max_us,total_us\n";

        std::lock_guard<std::mutex> lock(mutex_);
        size_t max_stages = stage_names_.size();
        std::vector<TickType> total_ticks(max_stages, 0);
        std::vector<int64_t> total_counts(max_stages, 0);

        for (auto* t : threads_) {
            for (size_t i = 0; i < t->accum_ticks.size() && i < max_stages; ++i) {
                total_ticks[i] += t->accum_ticks[i];
                total_counts[i] += t->counts[i];
            }
        }

        for (size_t i = 0; i < max_stages; ++i) {
            if (total_counts[i] == 0) continue;
            double total_us = total_ticks[i] * ClockConfig::ticks_to_us_factor;
            double avg_us = total_us / total_counts[i];
            ofs << stage_names_[i] << ","
                << total_counts[i] << ","
                << avg_us << ","
                << "N/A,N/A,"
                << total_us << "\n";
        }
        ofs.close();
        std::cout << "[MLA Timing] Exported to: " << filepath << "\n";
    }

    void print_summary() {
        if (ClockConfig::ticks_to_us_factor == 0.0) ClockConfig::init();

        std::cout << "\n========== MLA Timing Summary ==========\n";
        std::lock_guard<std::mutex> lock(mutex_);

        size_t max_stages = stage_names_.size();
        std::vector<TickType> total_ticks(max_stages, 0);
        std::vector<int64_t> total_counts(max_stages, 0);

        for (auto* t : threads_) {
            for (size_t i = 0; i < t->accum_ticks.size() && i < max_stages; ++i) {
                total_ticks[i] += t->accum_ticks[i];
                total_counts[i] += t->counts[i];
            }
        }

        double grand_total = 0;
        for (size_t i = 0; i < max_stages; ++i) {
            if (total_counts[i] == 0) continue;
            double total_us = total_ticks[i] * ClockConfig::ticks_to_us_factor;
            double avg_us = total_us / total_counts[i];
            grand_total += total_us;

            std::cout << std::left << std::setw(20) << stage_names_[i]
                      << std::right << std::setw(10) << total_counts[i]
                      << std::setw(12) << std::fixed << std::setprecision(2) << avg_us
                      << std::setw(12) << total_us << "\n";
        }
        std::cout << std::string(54, '-') << "\n";
        std::cout << std::left << std::setw(20) << "TOTAL"
                  << std::right << std::setw(34) << grand_total << " us\n";
        std::cout << "========================================\n\n";
    }

    const std::vector<std::string>& get_stage_names() const { return stage_names_; }

private:
    GlobalManager() = default;
    std::vector<std::string> stage_names_;
    std::unordered_map<std::string, int> stage_map_;
    std::vector<ThreadLocalStats*> threads_;
    std::mutex mutex_;
};

// 全局后端初始化（默认禁用）
TimingBackend g_current_backend = {backend_impl::disabled_start, backend_impl::disabled_end};

// ========== 7. ThreadLocalStats 构造函数定义（在 GlobalManager 之后） ==========
inline ThreadLocalStats::ThreadLocalStats() {
    GlobalManager::instance().register_thread(this);
}

inline ThreadLocalStats::~ThreadLocalStats() {
    GlobalManager::instance().unregister_thread(this);
}

// 定义 thread_local 实例
thread_local ThreadLocalStats g_tls_stats;

// ========== 8. 用户接口 ==========
inline void mla_timing_enable(bool enabled = true) {
    ClockConfig::init();
    GlobalManager::instance().set_enabled(enabled);
}

inline void mla_timing_reset() {
    GlobalManager::instance().reset();
}

inline void mla_timing_export(const std::string& filepath = "mla_timing_stats.csv") {
    GlobalManager::instance().export_to_file(filepath);
}

inline void mla_timing_print() {
    GlobalManager::instance().print_summary();
}

inline int mla_timing_register_stage(const std::string& name) {
    return GlobalManager::instance().register_stage(name);
}

// ========== 9. RAII 计时器 ==========
class ScopedTimer {
public:
    explicit ScopedTimer(int stage_id) : stage_id_(stage_id) {
        g_current_backend.start(stage_id_);
    }
    ~ScopedTimer() {
        g_current_backend.end(stage_id_);
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
private:
    int stage_id_;
};

} // namespace mla_timing

// 在程序启动或 kernel 第一次调用前
struct TimingIds {
    int load_kv, pack, qk_gemm, softmax_prep, sv_gemm, final_norm, task_loop;

    TimingIds() {
        load_kv = mla_timing::mla_timing_register_stage("load_kv");
        pack = mla_timing::mla_timing_register_stage("pack");
        qk_gemm = mla_timing::mla_timing_register_stage("qk_gemm");
        softmax_prep = mla_timing::mla_timing_register_stage("softmax_prep");
        sv_gemm = mla_timing::mla_timing_register_stage("sv_gemm");
        final_norm = mla_timing::mla_timing_register_stage("final_norm");
        task_loop = mla_timing::mla_timing_register_stage("task_loop");
    }
};

// 全局静态实例，确保 ID 固定
static const TimingIds g_timing_ids;


void enable_timing(at::Tensor& placeholder){
    mla_timing::mla_timing_enable(true);
    mla_timing::mla_timing_reset();
}

void export_timing(at::Tensor& placeholder){
    mla_timing::mla_timing_print();
    mla_timing::mla_timing_export(
        std::string("/sgl-workspace/sglang/dev/perf/") +
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
        std::string("_result.csv")
    );
}
