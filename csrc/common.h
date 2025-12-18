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
#define MAX_NEIGHBORS 4

// Observation Vector Dimension:
// [0,1]: Self (x, y) normalized
// [2,3,4]: Pacman (rel_x, rel_y, confidence)
// [5-16]: 4 Neighbors * 3 features (rel_x, rel_y, dist)
// Total = 2 + 3 + 12 = 17
#define OBS_DIM 17
#define OBS_DIM_ALIGNED 32 // for Cache Alignment
// ==========================================
// Structures
// ==========================================

// Agent State (Plain Old Data)
typedef struct {
    int x;
    int y;
    int alive; 
} AgentState;



typedef struct {
    // --- Config (Read Only) ---
    int grid_h;
    int grid_w;
    int n_agents;
    const int8_t *grid; 

    // --- Input State (Read Only) ---
    const AgentState *ghosts_in;  // [n_agents]
    const int *ghost_actions;     // [n_agents]
    
    int pacman_x_in;
    int pacman_y_in;
    int pacman_action;
    int pacman_speed;             // 0, 1, or 2

    // --- Random Number Generation (Input) ---
    // using random generated from python to secure thread safety
    const float *rand_pool; 
    int rand_pool_size;
    int *rand_idx;                // Pointer to scalar (update across steps)

    // --- Output State (Write Only) ---
    AgentState *ghosts_out;       // [n_agents]
    
    int pacman_x_out;
    int pacman_y_out;
    
    float *ghost_rewards;         // [n_agents]
    float pacman_reward;
    int done;

    // --- Observation Output (New!) ---
    // 每個 Agent 寫入屬於自己的一段記憶體 (長度 OBS_DIM)
    // Total Size: n_agents * OBS_DIM
    float *obs_out;
    char _padding[128];
} EnvState;

// ==========================================
// Function Prototypes
// ==========================================

void step_env_apply_actions_sequential(EnvState *env_state);

void step_env_apply_actions_batch(EnvState *states, int n_envs);

#ifdef __cplusplus
}
#endif

#endif // COMMON_H