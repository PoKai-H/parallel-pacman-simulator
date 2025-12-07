// csrc/step_env_apply_level2.c
// Level-2: Environment-level OpenMP parallelism.
// Run MANY independent environments in parallel on a single MPI rank.
//
// IMPORTANT:
// - This file does NOT change the environment rules.
// - It only orchestrates multiple calls to the (sequential or Level-1) kernel.
// - Team Member B will parallelize over `env_id` using OpenMP.
//
// Owner: Team Member B.

#include "common.h"
#include <omp.h>

// We use the sequential kernel as the reference implementation.
// Optionally, after Team A finishes Level-1, we can switch to that.
void step_env_apply_actions_sequential(
    int grid_h, int grid_w,
    const int *grid,
    int n_agents,
    const AgentState *ghosts_in,
    const int *ghost_actions,
    AgentState *ghosts_out,
    int pacman_x_in,
    int pacman_y_in,
    int pacman_action,
    int pacman_speed,
    int *pacman_x_out,
    int *pacman_y_out,
    float *ghost_rewards,
    float *pacman_reward,
    int *done
);

/**
 * Run one step for a batch of environments (sequential baseline).
 *
 * Each EnvState represents one independent world. For each environment:
 *   - We call the sequential kernel.
 *   - We update env->pacman_x, env->pacman_y, env->pacman_reward, env->done.
 *
 * NOTE:
 *   - We treat ghosts_in / ghosts_out as separate buffers.
 *   - The caller is responsible for swapping ghosts_in / ghosts_out
 *     between steps if needed.
 */
void step_env_apply_actions_batch_sequential(
    int n_envs,
    EnvState *envs
) {
    for (int e = 0; e < n_envs; ++e) {
        EnvState *env = &envs[e];

        int pac_x_out = 0;
        int pac_y_out = 0;
        float pac_reward = 0.0f;
        int done = 0;

        step_env_apply_actions_sequential(
            env->grid_h,
            env->grid_w,
            env->grid,
            env->n_agents,
            env->ghosts_in,
            env->ghost_actions,
            env->ghosts_out,
            env->pacman_x,       // input position
            env->pacman_y,
            env->pacman_action,
            env->pacman_speed,
            &pac_x_out,          // output position
            &pac_y_out,
            env->ghost_rewards,
            &pac_reward,
            &done
        );

        // Write back outputs into the EnvState struct
        env->pacman_x      = pac_x_out;
        env->pacman_y      = pac_y_out;
        env->pacman_reward = pac_reward;
        env->done          = done;

        // NOTE:
        // - env->ghosts_out and env->ghost_rewards were written in-place.
        // - If you want "ghosts_in" to always hold the current state,
        //   you can swap ghosts_in / ghosts_out in the caller after each step.
    }
}

/**
 * Level-2 environment-parallel version.
 *
 * TODO (Team B):
 *   1. Parallelize the outer loop over `e` using OpenMP:
 *
 *        #pragma omp parallel for
 *        for (int e = 0; e < n_envs; ++e) { ... }
 *
 *   2. Make sure that each thread only touches its own EnvState.
 *      (No shared mutable state across environments.)
 *
 *   3. Compare the outputs against step_env_apply_actions_batch_sequential
 *      to ensure correctness (bit-by-bit equality of all fields).
 */
void step_env_apply_actions_batch_level2(
    int n_envs,
    EnvState *envs
) {
    // For now, just call the sequential batch version.
    // This guarantees correctness until Team B adds OpenMP.
    step_env_apply_actions_batch_sequential(n_envs, envs);
}
