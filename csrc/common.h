#include <stdint.h>
// csrc/common.h

#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

// =========================
// Grid encoding
// =========================
// grid[y * grid_w + x] =
//   0 : empty cell
//   1 : wall (impassable)
//   2 : pellet (reserved for future use; not consumed in the minimal version)


// =========================
// Agent state
// =========================
// Used for ghosts (and can be reused for other agents in future).
// We intentionally keep this as a simple POD struct (no pointers) so that
// it is friendly to OpenMP, MPI and Python ctypes.
typedef struct {
    int x;
    int y;
    int alive;  // conditional mask for thread-safe operations
} AgentState;  


typedef struct {
    // --- Config / Static Info ---
    int grid_h;
    int grid_w;
    int n_agents;
    const int *grid;          // 指向 Map 資料 (Input)
    
    // --- Input State (Read Only) ---
    const AgentState *ghosts_in; // 指向 Ghost Array (Input)
    const int *ghost_actions;    // 指向 Action Array (Input)
    int pacman_x_in;
    int pacman_y_in;
    int pacman_action;
    int pacman_speed;

    // --- Output State (Write Only) ---
    AgentState *ghosts_out;   // 指向 Ghost Output Buffer
    int pacman_x_out;         // 直接存數值 (因為我們傳 EnvState 指標進去改)
    int pacman_y_out;
    
    // --- Rewards / Flags ---
    float *ghost_rewards;     // 這裡建議維持指標 (因為是 Array)
    float pacman_reward;      // Scalar 直接存數值
    int done;                 // Scalar 直接存數值
    
} EnvState;


// =========================
// Core environment kernel (sequential baseline)
// =========================
//
// Python chooses all actions and the pacman speed.
// This function only applies the actions, updates positions,
// checks walls, detects capture, computes rewards, and sets "done".
//
// Actions (for both ghosts and Pacman):
//   0 = stay
//   1 = move up    (y - 1)
//   2 = move down  (y + 1)
//   3 = move left  (x - 1)
//   4 = move right (x + 1)
//
// Speeds:
//   - Ghosts: implicitly move with speed = 1 (one grid cell per step).
//   - Pacman: moves with speed = pacman_speed (0, 1, or 2 cells per step).
//             The kernel will internally perform "pacman_speed" sub-steps,
//             each time moving one cell in the chosen direction, stopping
//             early if a wall is encountered.
//
// Parameters:
//   grid_h, grid_w : dimensions of the grid.
//   grid           : pointer to flattened grid array [grid_h * grid_w].
//   n_agents       : number of ghosts.
//   ghosts_in      : input ghost states at the beginning of the step.
//   ghost_actions  : array of size [n_agents], each in {0..4}.
//   ghosts_out     : output ghost states after the step.
//   pacman_x_in,
//   pacman_y_in    : Pacman's position at the beginning of the step.
//   pacman_action  : Pacman's action in {0..4}.
//   pacman_speed   : Pacman's speed (0, 1, or 2).
//   pacman_x_out,
//   pacman_y_out   : Pacman's position after the step.
//   ghost_rewards  : output rewards for each ghost [n_agents].
//   pacman_reward  : output reward for Pacman (scalar).
//   done           : set to 1 if the episode ends in this step, else 0.
//
// Minimal version reward / termination rule:
//   - If any ghost occupies the same cell as Pacman after movement:
//       * all ghosts get +1
//       * Pacman gets -1
//       * done = 1
//   - Otherwise:
//       * all rewards = 0
//       * done = 0
//
void step_env_apply_actions_sequential(EnvState *env_state);


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