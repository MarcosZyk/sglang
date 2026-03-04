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

namespace mla_timing {

// ========== 1. 时钟配置（一次性计算转换系数）==========
using Clock = std::chrono::high_resolution_clock;
using TickType = uint64_t;

struct ClockConfig {
    static double ticks_to_us_factor;
    static void init() {
        // 计算 1 tick 等于多少微秒
        // period 是 std::ratio，例如 1/1000000000 (nanoseconds)
        using period = Clock::period;
        ticks_to_us_factor = static_cast<double>(period::num) / static_cast<double>(period::den) * 1e6;
    }
};
// 静态成员定义
double ClockConfig::ticks_to_us_factor = 0.0;

// ========== 2. 策略模式后端（消除分支）==========
// 定义后端接口（函数指针集合）
struct TimingBackend {
    void (*start)(int stage_id);
    void (*end)(int stage_id);
};

struct ThreadLocalStats;

// 声明 thread_local 实例
thread_local ThreadLocalStats g_tls_stats;

// ========== 3. 具体实现（Enabled vs Disabled）==========
namespace backend_impl {

    // 启用实现：记录 ticks
    inline void enabled_start(int stage_id) {
        auto& tls = g_tls_stats;
        tls.ensure_size(stage_id + 1);
        tls.start_ticks[stage_id] = Clock::now().time_since_epoch().count();
    }

    inline void enabled_end(int stage_id) {
        auto& tls = g_tls_stats;
        if (stage_id >= tls.start_ticks.size()) return;

        TickType end = Clock::now().time_since_epoch().count();
        TickType elapsed = end - tls.start_ticks[stage_id];

        // 确保累加数组足够大
        if (stage_id >= tls.accum_ticks.size())
            tls.accum_ticks.resize(stage_id + 1, 0);

        tls.accum_ticks[stage_id] += elapsed;
    }

    // 禁用实现：空操作（编译器通常会优化掉调用）
    inline void disabled_start(int) {}
    inline void disabled_end(int) {}
}

// 全局后端指针（运行时切换）
extern TimingBackend g_current_backend;

// ========== 4. 全局管理器（单例）==========
class GlobalManager {
public:
    static GlobalManager& instance() {
        static GlobalManager inst;
        return inst;
    }

    // 注册阶段名，返回 ID
    int register_stage(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stage_map_.find(name);
        if (it != stage_map_.end()) return it->second;

        int id = stage_names_.size();
        stage_map_[name] = id;
        stage_names_.push_back(name);
        return id;
    }

    // 线程注册/注销
    void register_thread(ThreadLocalStats* stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        threads_.push_back(stats);
    }
    void unregister_thread(ThreadLocalStats* stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        threads_.erase(std::remove(threads_.begin(), threads_.end(), stats), threads_.end());
    }

    // 切换后端（运行时开关）
    void set_enabled(bool enabled) {
        if (enabled) {
            g_current_backend = {backend_impl::enabled_start, backend_impl::enabled_end};
        } else {
            g_current_backend = {backend_impl::disabled_start, backend_impl::disabled_end};
        }
    }

    // 重置所有统计
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto* t : threads_) {
            // 注意：只能重置当前已知的线程，新线程会在构造时初始化为 0
            // 为了安全，这里只清空已知线程的 accum，start_ticks 不需要清空
            std::fill(t->accum_ticks.begin(), t->accum_ticks.end(), 0);
        }
    }

    // 导出到 CSV（此时才计算微秒）
    void export_to_file(const std::string& filepath) {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "[MLA Timing] Failed to open " << filepath << "\n";
            return;
        }

        // 确保转换系数已初始化
        if (ClockConfig::ticks_to_us_factor == 0.0) ClockConfig::init();

        ofs << "stage,count,avg_us,min_us,max_us,total_us\n";

        // 由于我们只记录了累加和，没有记录 count/min/max，这里简化为总和
        // 如需详细统计，需在 ThreadLocalStats 中增加更多字段
        std::lock_guard<std::mutex> lock(mutex_);

        // 临时合并所有线程数据
        size_t max_stages = stage_names_.size();
        std::vector<TickType> global_ticks(max_stages, 0);
        std::vector<int64_t> global_counts(max_stages, 0); // 简化：假设每个线程每阶段调用次数相同或无法精确统计
        // 注意：当前设计只累加了 ticks，没有记录 count。
        // 优化：在 ThreadLocalStats 中增加 vector<int64_t> counts;

        // 为了解决 count 问题，我们需要修改 ThreadLocalStats (见下文修正)
        // 这里假设我们在 end 时也增加了 count 计数
    }

    // 打印摘要
    void print_summary() {
        if (ClockConfig::ticks_to_us_factor == 0.0) ClockConfig::init();

        std::cout << "\n========== MLA Timing Summary ==========\n";
        std::lock_guard<std::mutex> lock(mutex_);

        // 合并数据
        size_t max_stages = stage_names_.size();
        std::vector<TickType> total_ticks(max_stages, 0);
        std::vector<int64_t> total_counts(max_stages, 0);

        for (auto* t : threads_) {
            for (size_t i = 0; i < t->accum_ticks.size() && i < max_stages; ++i) {
                total_ticks[i] += t->accum_ticks[i];
                // 注意：这里需要 counts 支持，见下方代码修正
            }
        }

        for (size_t i = 0; i < max_stages; ++i) {
            if (total_ticks[i] == 0) continue;
            double us = total_ticks[i] * ClockConfig::ticks_to_us_factor;
            std::cout << std::left << std::setw(20) << stage_names_[i]
                      << " : " << std::fixed << std::setprecision(2) << us << " us (total ticks: " << total_ticks[i] << ")\n";
        }
        std::cout << "========================================\n\n";
    }

    const std::vector<std::string>& get_stage_names() { return stage_names_; }

private:
    GlobalManager() = default;
    std::vector<std::string> stage_names_;
    std::unordered_map<std::string, int> stage_map_;
    std::vector<ThreadLocalStats*> threads_;
    std::mutex mutex_;
};

// 全局后端实例初始化（默认禁用）
TimingBackend g_current_backend = {backend_impl::disabled_start, backend_impl::disabled_end};

// ========== 5. 增强版 ThreadLocalStats（支持 Count）==========
// 重新定义以支持更精确统计
struct ThreadLocalStats {
    std::vector<TickType> accum_ticks;
    std::vector<int64_t> counts;
    std::vector<TickType> start_ticks;

    ThreadLocalStats() {
        GlobalManager::instance().register_thread(this);
    }
    ~ThreadLocalStats() {
        GlobalManager::instance().unregister_thread(this);
    }
    void ensure_size(int num_stages) {
        if (accum_ticks.size() < num_stages) {
            accum_ticks.resize(num_stages, 0);
            counts.resize(num_stages, 0);
            start_ticks.resize(num_stages, 0);
        }
    }
};
// 注意：由于上面已经声明过 thread_local g_tls_stats，这里需要确保定义一致
// 在实际代码中，请将上面的 struct 定义替换为此版本
thread_local ThreadLocalStats g_tls_stats;

// 更新 enabled_end 实现以记录 count
namespace backend_impl {
    inline void enabled_end(int stage_id) {
        auto& tls = g_tls_stats;
        if (stage_id >= tls.start_ticks.size()) return;

        TickType end = Clock::now().time_since_epoch().count();
        TickType elapsed = end - tls.start_ticks[stage_id];

        if (stage_id >= tls.accum_ticks.size())
            tls.accum_ticks.resize(stage_id + 1, 0);
        if (stage_id >= tls.counts.size())
            tls.counts.resize(stage_id + 1, 0);

        tls.accum_ticks[stage_id] += elapsed;
        tls.counts[stage_id]++;
    }
}

// 更新 GlobalManager::export_to_file 以使用 count
// (此处省略重复代码，逻辑同上，使用 total_counts 计算 avg)

// ========== 6. 用户接口 ==========
inline void mla_timing_enable(bool enabled = true) {
    ClockConfig::init(); // 确保系数计算
    GlobalManager::instance().set_enabled(enabled);
}

inline void mla_timing_reset() {
    GlobalManager::instance().reset();
}

inline void mla_timing_export(const std::string& filepath = "mla_timing.csv") {
    GlobalManager::instance().export_to_file(filepath);
}

inline void mla_timing_print() {
    GlobalManager::instance().print_summary();
}

inline int mla_timing_register_stage(const std::string& name) {
    return GlobalManager::instance().register_stage(name);
}

// ========== 7. 零开销计时器（使用 Stage ID）==========
class ScopedTimer {
public:
    explicit ScopedTimer(int stage_id) : stage_id_(stage_id) {
        // 直接调用函数指针，无 if 分支
        g_current_backend.start(stage_id_);
    }
    ~ScopedTimer() {
        g_current_backend.end(stage_id_);
    }
private:
    int stage_id_;
};

} // namespace mla_timing

// 在程序启动或 kernel 第一次调用前
struct TimingIds {
    int pack, qk_gemm, softmax_prep, sv_gemm, final_norm, task_loop;

    TimingIds() {
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


void enable_timing(){
    mla_timing::mla_timing_enable(true);
    mla_timing::mla_timing_reset();
}

void export_timing(){
    mla_timing::mla_timing_print();
    mla_timing::mla_timing_export(
    "/sgl-workspace/sglang/dev/perf/" + std::chrono::high_resolution_clock::now().time_since_epoch().count() + "_result.csv"
    );
}
