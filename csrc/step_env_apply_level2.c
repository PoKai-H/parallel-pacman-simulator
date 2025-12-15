// csrc/step_env_apply_level2.c
// Level-2: Environment-level OpenMP parallelism.
// Owner: Team Member B.

#include "common.h"
#include <omp.h>
#include <stdio.h>

/**
 * [Baseline] Run one step for a batch of environments SEQUENTIALLY.
 * * This function loops over all environments one by one.
 * It serves as the "Ground Truth" for Level-2 correctness.
 */
void step_env_apply_actions_batch_sequential(EnvState *states, int n_envs) {
    for (int i = 0; i < n_envs; i++) {
        // 因為現在架構是 Pass-by-Pointer，
        // step_env_apply_actions_sequential 會直接修改 states[i] 內部的記憶體
        // (例如直接寫入 pacman_x_out, obs_out 等)
        // 所以這裡不需要手動 unpacking / packing 變數，非常乾淨。
        step_env_apply_actions_sequential(&states[i]);
    }
}

/**
 * [Target] Level-2 Environment-Parallel Version.
 * * This is the function exposed to Python as "step_env_apply_actions_batch".
 * * TODO (Team B):
 * 1. Replace the sequential call below with an OpenMP parallel loop.
 * 2. Use: #pragma omp parallel for schedule(dynamic)
 * 3. Ensure correct syntax and performance.
 */
void step_env_apply_actions_batch(EnvState *states, int n_envs) {
    
    // --- Member B 的任務區域 ---
    
    // 目前先呼叫 Sequential 版，確保編譯通過且能跑
    #pragma omp parallel for default(none) shared(states, n_envs) schedule(static)
    // 當你開始實作時，請註解掉下面這行，並寫入你的 OpenMP 迴圈
    for (int i = 0; i < n_envs; i++) {
        // 每個 iteration i 對應「第 i 個環境」的單步更新
        // 只會讀/寫 states[i] 內部的欄位
        // [MOD-2] 每個 thread 只處理自己的 states[i]，避免互相寫到同一份資料
        step_env_apply_actions_sequential(&states[i]);
    }
    // 提示: 你的實作應該長這樣:
    // #pragma omp parallel for ...
    // for (int i = 0; i < n_envs; i++) {
    //     step_env_apply_actions_sequential(&states[i]);
    // }
}

    // #pragma omp parallel for：OpenMP 建立一個 thread team，
    // 把 for 迴圈的 iteration（這裡是每個 i 代表一個 env）分配給不同 thread 執行。
    // default(none)：
    //   - 強制你必須明確標註每個變數是 shared 還是 private，
    //   - 避免不小心把某個變數「默認共享」造成 data race，
    //   - 這是寫平行程式時非常推薦的安全做法（能讓 compiler 幫你抓錯）。
    // shared(states, n_envs)：
    //   - states 指向整個 EnvState 陣列，所有 threads 都看得到同一個指標（共享讀取/索引）
    //   - n_envs 是迴圈上界，也需要共享
    //   - 但每個 thread 實際只會寫它負責的 states[i]，因此不會互相覆寫（前提是 state 真的是 per-env）。
    // schedule(static)：
    //   - 用「靜態分配」的方式分工：一開始就把 i 的範圍切好分給 threads
    //   - overhead 最低，通常適合「每個 env step 工作量差不多」的情況
    //   - 若 env 的工作量差很大才考慮 dynamic/guided