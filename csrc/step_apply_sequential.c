// csrc/step_apply_sequential.c
// Sequential baseline for the Pacman environment.
// Python chooses all actions and pacman_speed.
// This function only applies actions, updates positions, checks walls,
// detects capture, computes rewards, and sets the done flag.

#include "common.h"

// Helpers
static inline int grid_idx(int y, int x, int width) {
    return y * width + x;
}

static inline int in_bounds(int y, int x, int h, int w) {
    return (y >= 0 && y < h && x >= 0 && x < w);
}

// grid: 0 = empty, 1 = wall, 2 = pellet (unused in minimal version)
static inline int is_wall(const int *grid, int y, int x, int h, int w) {
    if (!in_bounds(y, x, h, w)) return 1; // out-of-bounds treated as wall
    return grid[grid_idx(y, x, w)] == 1;
}

// 0=stay, 1=up, 2=down, 3=left, 4=right
static void action_to_delta(int action, int *dx, int *dy) {
    switch (action) {
        case 1: *dx = 0;  *dy = -1; break; // up
        case 2: *dx = 0;  *dy = 1;  break; // down
        case 3: *dx = -1; *dy = 0;  break; // left
        case 4: *dx = 1;  *dy = 0;  break; // right
        case 0:
        default: *dx = 0; *dy = 0;  break; // stay
    }
}

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
    int pacman_speed,   // 0,1,2 (max speed = 2)
    int *pacman_x_out,
    int *pacman_y_out,
    float *ghost_rewards,
    float *pacman_reward,
    int *done
) {
    // ----- init rewards & done flag -----
    for (int i = 0; i < n_agents; i++) {
        ghost_rewards[i] = 0.0f;
    }
    *pacman_reward = 0.0f;
    *done = 0;

    // ----- copy initial ghost states -----
    for (int i = 0; i < n_agents; i++) {
        ghosts_out[i] = ghosts_in[i];
    }

    int pac_x = pacman_x_in;
    int pac_y = pacman_y_in;

    // Clamp pacman_speed to [0, 2] for safety
    if (pacman_speed < 0) pacman_speed = 0;
    if (pacman_speed > 2) pacman_speed = 2;

    // =========================
    // 1) Move ghosts (speed = 1)
    // =========================
    for (int i = 0; i < n_agents; i++) {
        if (!ghosts_in[i].alive) {
            // keep as is
            ghosts_out[i] = ghosts_in[i];
            continue;
        }

        int dx = 0, dy = 0;
        action_to_delta(ghost_actions[i], &dx, &dy);

        int nx = ghosts_in[i].x + dx;
        int ny = ghosts_in[i].y + dy;

        if (is_wall(grid, ny, nx, grid_h, grid_w)) {
            // stay if next is wall
            ghosts_out[i].x = ghosts_in[i].x;
            ghosts_out[i].y = ghosts_in[i].y;
        } else {
            ghosts_out[i].x = nx;
            ghosts_out[i].y = ny;
        }
        ghosts_out[i].alive = ghosts_in[i].alive; // minimal version: no death
    }

    // =========================
    // 2) Move Pacman (speed = pacman_speed)
    // =========================
    {
        int dx = 0, dy = 0;
        action_to_delta(pacman_action, &dx, &dy);

        for (int s = 0; s < pacman_speed; s++) {
            int nx = pac_x + dx;
            int ny = pac_y + dy;

            // stop if next cell is a wall
            if (is_wall(grid, ny, nx, grid_h, grid_w)) {
                break;
            }

            pac_x = nx;
            pac_y = ny;

            // minimal version: we only check capture after all movement
            // (no sub-step capture)
        }
    }

    // =========================
    // 3) Detect capture (after movement)
    // =========================
    int captured = 0;
    for (int i = 0; i < n_agents; i++) {
        if (!ghosts_out[i].alive) continue;
        if (ghosts_out[i].x == pac_x && ghosts_out[i].y == pac_y) {
            captured = 1;
            break;
        }
    }

    if (captured) {
        for (int i = 0; i < n_agents; i++) {
            ghost_rewards[i] = 1.0f;
        }
        *pacman_reward = -1.0f;
        *done = 1;
    }

    // =========================
    // 4) Write back Pacman position
    // =========================
    *pacman_x_out = pac_x;
    *pacman_y_out = pac_y;
}
