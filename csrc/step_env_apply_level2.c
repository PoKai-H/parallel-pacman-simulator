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
    // 當你開始實作時，請註解掉下面這行，並寫入你的 OpenMP 迴圈
    step_env_apply_actions_batch_sequential(states, n_envs);

    // 提示: 你的實作應該長這樣:
    // #pragma omp parallel for ...
    // for (int i = 0; i < n_envs; i++) {
    //     step_env_apply_actions_sequential(&states[i]);
    // }
}