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
    
    
    
    
    #pragma omp parallel for default(none) shared(states, n_envs) schedule(static)
    
    for (int i = 0; i < n_envs; i++) {
        
        // Debug: 
        // if (i < 2) {
        //     printf("--- Debug Env %d ---\n", i);
        //     printf("Base Addr: %p\n", (void*)&states[i]);
        //     printf("grid: %p\n", (void*)states[i].grid);
        //     printf("ghosts_in: %p\n", (void*)states[i].ghosts_in);
        //     printf("ghost_actions: %p\n", (void*)states[i].ghost_actions);
        //     printf("rand_pool: %p\n", (void*)states[i].rand_pool);
        //     printf("obs_out: %p\n", (void*)states[i].obs_out);
        //     fflush(stdout); // 強制輸出
        // }
        
        step_env_apply_actions_sequential(&states[i]);
    }
}

int get_c_struct_size() {
    return sizeof(EnvState);
}