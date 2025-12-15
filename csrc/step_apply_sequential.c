// csrc/step_apply_sequential.c
#include "common.h"

// ==========================================
// Helper Functions (Internal)
// ==========================================

static inline int grid_idx(int y, int x, int width) {
    return y * width + x;
}

static inline int in_bounds(int y, int x, int h, int w) {
    return (y >= 0 && y < h && x >= 0 && x < w);
}

// 注意：這裡配合 common.h 改成了 const int8_t *grid
static inline int is_wall(const int8_t *grid, int y, int x, int h, int w) {
    if (!in_bounds(y, x, h, w)) return 1; 
    return grid[grid_idx(y, x, w)] == 1;
}

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

// ==========================================
// Main Kernel Implementation
// ==========================================

void step_env_apply_actions_sequential(EnvState *s) {
    // 1. Init rewards & done flag
    // ------------------------------------------------
    for (int i = 0; i < s->n_agents; i++) {
        s->ghost_rewards[i] = 0.0f;
    }
    s->pacman_reward = 0.0f;
    s->done = 0;

    // 2. Setup Local Variables for Clean Access
    // ------------------------------------------------
    // Copy input pacman state locally to mutate during sub-steps
    int pac_x = s->pacman_x_in;
    int pac_y = s->pacman_y_in;
    int pm_speed = s->pacman_speed;

    // Safety clamp
    if (pm_speed < 0) pm_speed = 0;
    if (pm_speed > 2) pm_speed = 2;

    // 3. Move Ghosts
    // ------------------------------------------------
    for (int i = 0; i < s->n_agents; i++) {
        // Read from input buffer
        AgentState current = s->ghosts_in[i];

        if (!current.alive) {
            s->ghosts_out[i] = current; // Direct copy if dead
            continue;
        }

        int dx = 0, dy = 0;
        action_to_delta(s->ghost_actions[i], &dx, &dy);

        int nx = current.x + dx;
        int ny = current.y + dy;

        // Use 's->' to access grid dimensions and data
        if (is_wall(s->grid, ny, nx, s->grid_h, s->grid_w)) {
            // Hit wall: stay put
            s->ghosts_out[i].x = current.x;
            s->ghosts_out[i].y = current.y;
        } else {
            // Move
            s->ghosts_out[i].x = nx;
            s->ghosts_out[i].y = ny;
        }
        s->ghosts_out[i].alive = current.alive;
    }

    // 4. Move Pacman (Sub-steps)
    // ------------------------------------------------
    {
        int dx = 0, dy = 0;
        action_to_delta(s->pacman_action, &dx, &dy);

        for (int step = 0; step < pm_speed; step++) {
            int nx = pac_x + dx;
            int ny = pac_y + dy;

            if (is_wall(s->grid, ny, nx, s->grid_h, s->grid_w)) {
                break; // Stop at wall
            }
            pac_x = nx;
            pac_y = ny;
        }
    }

    // 5. Detect Capture
    // ------------------------------------------------
    int captured = 0;
    for (int i = 0; i < s->n_agents; i++) {
        // Check against NEW ghost positions
        if (!s->ghosts_out[i].alive) continue;
        
        if (s->ghosts_out[i].x == pac_x && s->ghosts_out[i].y == pac_y) {
            captured = 1;
            break; 
        }
    }

    // 6. Finalize Outputs
    // ------------------------------------------------
    if (captured) {
        for (int i = 0; i < s->n_agents; i++) {
            s->ghost_rewards[i] = 1.0f;
        }
        s->pacman_reward = -1.0f;
        s->done = 1;
    }

    // Write back Pacman final position to the struct
    s->pacman_x_out = pac_x;
    s->pacman_y_out = pac_y;
}