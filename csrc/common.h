#include <stdint.h>
// csrc/common.h

#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

// ==========================================
// Constants & Configuration
// ==========================================
// KNN 策略: 每個 Agent 只觀察最近的 4 個鄰居
#define MAX_NEIGHBORS 4

// Observation Vector Dimension:
// [0,1]: Self (x, y) normalized
// [2,3,4]: Pacman (rel_x, rel_y, confidence)
// [5-16]: 4 Neighbors * 3 features (rel_x, rel_y, dist)
// Total = 2 + 3 + 12 = 17
#define OBS_DIM 17

// ==========================================
// Structures
// ==========================================

// Agent State (Plain Old Data)
typedef struct {
    int x;
    int y;
    int alive; 
} AgentState;


// 核心環境狀態 Context
// 所有 Level 1/2/3 的函式都只傳遞這個 Struct 的指標
typedef struct {
    // --- Config (Read Only) ---
    int grid_h;
    int grid_w;
    int n_agents;
    const int8_t *grid; // 優化: 使用 int8 節省頻寬

    // --- Input State (Read Only) ---
    const AgentState *ghosts_in;  // [n_agents]
    const int *ghost_actions;     // [n_agents]
    
    int pacman_x_in;
    int pacman_y_in;
    int pacman_action;
    int pacman_speed;             // 0, 1, or 2

    // --- Random Number Generation (Input) ---
    // 從 Python 傳入預先生成的亂數池，解決 C 語言 rand() 不安全的問題
    const float *rand_pool; 
    int rand_pool_size;
    int *rand_idx;                // Pointer to scalar (update across steps)

    // --- Output State (Write Only) ---
    AgentState *ghosts_out;       // [n_agents]
    
    // C 直接修改這些數值，不用 return
    int pacman_x_out;
    int pacman_y_out;
    
    float *ghost_rewards;         // [n_agents]
    float pacman_reward;
    int done;

    // --- Observation Output (New!) ---
    // 每個 Agent 寫入屬於自己的一段記憶體 (長度 OBS_DIM)
    // Total Size: n_agents * OBS_DIM
    float *obs_out;

} EnvState;

// ==========================================
// Function Prototypes
// ==========================================

void step_env_apply_actions_sequential(EnvState *env_state);

void step_env_apply_actions_batch(EnvState *states, int n_envs);
// =========================
// Notes for parallel versions
// =========================
//
// - Level 1 (intra-environment, OpenMP):
//     Implement step_env_apply_actions_parallel(...) with the SAME logical
//     behavior as the sequential kernel, but parallelize over ghosts.
//
// - Level 2 (inter-environment, OpenMP):
//     Implement a function that loops over multiple environments and calls
//     the (sequential or Level 1) kernel inside an OpenMP parallel-for
//     over env_id.
//
// - Level 3 (inter-episode, MPI):
//     Implement a function that distributes different episodes across MPI
//     ranks, each rank repeatedly calling the (Level 1/2) kernel for its
//     assigned episodes, and gathering statistics at rank 0.
//
// All these parallel functions should include this header and must be
// tested against the sequential baseline for correctness.

#ifdef __cplusplus
}
#endif

#endif // COMMON_H