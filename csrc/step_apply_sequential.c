#include "common.h"
#include <math.h>
#include <stdlib.h>
#include <string.h> // for memset
#include <omp.h> // 1. 引入 OpenMP 頭文件
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================
// Helper Functions (Physics)
// ==========================================

static inline int grid_idx(int y, int x, int width) {
    return y * width + x;
}

static inline int in_bounds(int y, int x, int h, int w) {
    return (y >= 0 && y < h && x >= 0 && x < w);
}

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
// Helper Functions (Math & RNG)
// ==========================================

// 從 Pool 拿亂數 (Thread-safe friendly logic)
static inline float get_rand(const float *pool, int size, int *idx) {
    float val = pool[*idx];
    *idx = (*idx + 1) % size; // 循環使用
    return val;
}

// Box-Muller Transform: 產生高斯分佈 N(mean, std)
static float sample_normal(float mean, float std, const float *pool, int size, int *idx) {
    float u1 = get_rand(pool, size, idx);
    float u2 = get_rand(pool, size, idx);
    
    if(u1 < 1e-6f) u1 = 1e-6f; // 避免 log(0)
    
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + z0 * std;
}

// 內部結構，用於 KNN 排序
typedef struct {
    int id;
    float dx;
    float dy;
    float dist_sq;
} NeighborCandidate;

// ==========================================
// Core Logic: 1. Physics (Movement)
// ==========================================
void apply_physics(EnvState *s) {
    // A. Move Ghosts
    for (int i = 0; i < s->n_agents; i++) {
        AgentState current = s->ghosts_in[i];
        if (!current.alive) {
            s->ghosts_out[i] = current;
            continue;
        }

        int dx = 0, dy = 0;
        action_to_delta(s->ghost_actions[i], &dx, &dy);
        int nx = current.x + dx;
        int ny = current.y + dy;

        if (is_wall(s->grid, ny, nx, s->grid_h, s->grid_w)) {
            s->ghosts_out[i].x = current.x;
            s->ghosts_out[i].y = current.y;
        } else {
            s->ghosts_out[i].x = nx;
            s->ghosts_out[i].y = ny;
        }
        s->ghosts_out[i].alive = current.alive;
    }

    // B. Move Pacman (Sub-steps)
    int px = s->pacman_x_in;
    int py = s->pacman_y_in;
    int pdx = 0, pdy = 0;
    action_to_delta(s->pacman_action, &pdx, &pdy);
    
    int speed = s->pacman_speed;
    if (speed < 0) speed = 0; 
    if (speed > 2) speed = 2;

    for (int step = 0; step < speed; step++) {
        int nx = px + pdx;
        int ny = py + pdy;
        if (is_wall(s->grid, ny, nx, s->grid_h, s->grid_w)) break;
        px = nx;
        py = ny;
    }

    // C. Detect Capture & Rewards
    int captured = 0;
    for (int i = 0; i < s->n_agents; i++) {
        s->ghost_rewards[i] = -0.01f; // Time penalty
        if (s->ghosts_out[i].alive) {
            if (s->ghosts_out[i].x == px && s->ghosts_out[i].y == py) {
                captured = 1;
            }
        }
    }

    if (captured) {
        for (int i = 0; i < s->n_agents; i++) s->ghost_rewards[i] += 1.0f;
        s->pacman_reward = -10.0f;
        s->done = 1;
    } else {
        s->pacman_reward = 1.0f; // Survival reward
        s->done = 0;
    }

    s->pacman_x_out = px;
    s->pacman_y_out = py;
}

// ==========================================
// Core Logic: 2. Sensing (Parallelized)
// ==========================================
void compute_observations(EnvState *s) {
    float gw = (float)s->grid_w;
    float gh = (float)s->grid_h;

    // 1. OpenMP 平行化 (注意 schedule 用 static 對齊 N=16)
    #pragma omp parallel for default(none) shared(s, gw, gh) schedule(static)
    for (int i = 0; i < s->n_agents; i++) {
        
        // 2. 宣告在迴圈內 (Private) & 加上 * 取值
        int local_rand_idx = (*s->rand_idx + i * 131) % s->rand_pool_size;
        int *p_rand_idx = &local_rand_idx; // 指向區域變數

        float *my_obs = &s->obs_out[i * OBS_DIM_ALIGNED];
        AgentState me = s->ghosts_out[i];

        if (!me.alive) {
            memset(my_obs, 0, OBS_DIM_ALIGNED * sizeof(float));
            continue;
        }

        my_obs[0] = (float)me.x / gw;
        my_obs[1] = (float)me.y / gh;

        float px = (float)s->pacman_x_out;
        float py = (float)s->pacman_y_out;
        float dx = px - me.x;
        float dy = py - me.y;
        float dist = sqrtf(dx*dx + dy*dy);

        if (dist <= 3.0f) { 
            float alpha = 0.5f;
            float conf = expf(-alpha * dist);
            float sigma = 0.1f * dist; 
            
            // 3. 關鍵檢查點：這裡一定要傳 p_rand_idx，絕對不能傳 s->rand_idx
            float nx = sample_normal(0.0f, sigma, s->rand_pool, s->rand_pool_size, p_rand_idx);
            float ny = sample_normal(0.0f, sigma, s->rand_pool, s->rand_pool_size, p_rand_idx);
            
            float raw_vx = dx + nx;
            float raw_vy = dy + ny;
            float mag = sqrtf(raw_vx*raw_vx + raw_vy*raw_vy);
            
            if (mag > 1e-6f) {
                my_obs[2] = raw_vx / mag;
                my_obs[3] = raw_vy / mag;
            } else {
                my_obs[2] = 0.0f;
                my_obs[3] = 0.0f;
            }
            my_obs[4] = conf;
        } else {
            my_obs[2] = 0.0f; my_obs[3] = 0.0f; my_obs[4] = 0.0f;
        }

        // --- Neighbor Sensing (KNN) ---
        // 注意：candidates 也要宣告在迴圈內
        NeighborCandidate candidates[16]; 
        int count = 0;

        for (int j = 0; j < s->n_agents; j++) {
            if (i == j) continue;
            if (!s->ghosts_out[j].alive) continue;

            float rdx = (float)(s->ghosts_out[j].x - me.x);
            float rdy = (float)(s->ghosts_out[j].y - me.y);
            float d2 = rdx*rdx + rdy*rdy;

            if (d2 <= 9.0f) {
                candidates[count].id = j;
                candidates[count].dx = rdx / gw;
                candidates[count].dy = rdy / gh;
                candidates[count].dist_sq = d2;
                count++;
                if (count >= 16) break;
            }
        }

        for (int p = 1; p < count; p++) {
            NeighborCandidate key = candidates[p];
            int q = p - 1;
            while (q >= 0 && candidates[q].dist_sq > key.dist_sq) {
                candidates[q + 1] = candidates[q];
                q--;
            }
            candidates[q + 1] = key;
        }

        int offset = 5; 
        for (int k = 0; k < MAX_NEIGHBORS; k++) {
            if (k < count) {
                my_obs[offset + k*3 + 0] = candidates[k].dx;
                my_obs[offset + k*3 + 1] = candidates[k].dy;
                my_obs[offset + k*3 + 2] = sqrtf(candidates[k].dist_sq) / gw;
            } else {
                my_obs[offset + k*3 + 0] = 0.0f;
                my_obs[offset + k*3 + 1] = 0.0f;
                my_obs[offset + k*3 + 2] = 0.0f;
            }
        }
    } 

    // 4. 更新全域亂數 (for next step)
    *s->rand_idx = (*s->rand_idx + s->n_agents * 7) % s->rand_pool_size;
}

// ==========================================
// Main Entry Point
// ==========================================
void step_env_apply_actions_sequential(EnvState *s) {
    // 1. Physics (移動與碰撞)
    apply_physics(s);

    // 2. Sensing (產生 Observation)
    // 這一部分在 Level 1 會被 OpenMP 平行化
    compute_observations(s);
}