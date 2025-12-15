// csrc/step_env_apply_level1.c
// Level-1: Agent-level OpenMP parallelism.
// This file defines a drop-in replacement for the sequential kernel.
//
// IMPORTANT:
// - The function signature MUST match the sequential version.
// - Initially we just call the sequential baseline for correctness.
// - Team Member A will later copy the body from the sequential
//   implementation and parallelize the ghost loop with OpenMP.
//
// Owner: Team Member A.

#include "common.h"
#include <omp.h>   // needed for OpenMP (later)

// Forward declaration of the sequential baseline
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
 * Level-1 agent-parallel version of the environment step.
 * For now, this is a thin wrapper around the sequential version.
 *
 * TODO(Team A):
 *   1. Copy the body of step_env_apply_actions_sequential into this file.
 *   2. Add `#pragma omp parallel for` to the ghost update loop ONLY.
 *   3. Make sure the behavior is bit-by-bit identical to the sequential version.
 *   4. Run the correctness tests against the sequential baseline.
 */
void step_env_apply_actions_level1(
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
) {
    // For now, just call the sequential reference.
    // This guarantees that everything compiles and runs.
    step_env_apply_actions_sequential(
        grid_h, grid_w,
        grid,
        n_agents,
        ghosts_in,
        ghost_actions,
        ghosts_out,
        pacman_x_in,
        pacman_y_in,
        pacman_action,
        pacman_speed,
        pacman_x_out,
        pacman_y_out,
        ghost_rewards,
        pacman_reward,
        done
    );
}
