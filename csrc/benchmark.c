// csrc/benchmark.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include "common.h"

// 模擬 Python 端的初始化，分配記憶體給一個環境
void init_dummy_env(EnvState *s, int h, int w, int n_agents) {
    s->grid_h = h;
    s->grid_w = w;
    s->n_agents = n_agents;

    // 1. Grid (40x40)
    int8_t *grid = (int8_t*)malloc(h * w * sizeof(int8_t));
    memset(grid, 0, h * w); // 全空地
    s->grid = grid;

    // 2. Agents
    AgentState *ghosts_in = (AgentState*)malloc(n_agents * sizeof(AgentState));
    AgentState *ghosts_out = (AgentState*)malloc(n_agents * sizeof(AgentState));
    int *actions = (int*)malloc(n_agents * sizeof(int));
    float *rewards = (float*)malloc(n_agents * sizeof(float));
    
    // 初始化位置
    for(int i=0; i<n_agents; i++) {
        ghosts_in[i].x = i % w;
        ghosts_in[i].y = (i / w) % h;
        ghosts_in[i].alive = 1;
        actions[i] = 1; // 全部往上走
    }
    s->ghosts_in = ghosts_in;
    s->ghosts_out = ghosts_out;
    s->ghost_actions = actions;
    s->ghost_rewards = rewards;

    // 3. Pacman
    s->pacman_x_in = 20;
    s->pacman_y_in = 20;
    s->pacman_speed = 2;
    s->pacman_action = 0;

    // 4. Random Pool (給 KNN 用的)
    s->rand_pool_size = 100000;
    float *pool = (float*)malloc(s->rand_pool_size * sizeof(float));
    for(int i=0; i<s->rand_pool_size; i++) pool[i] = ((float)rand()/RAND_MAX);
    s->rand_pool = pool;

    s->rand_idx = (int*)malloc(sizeof(int));
    *(s->rand_idx) = 0;

    // 5. Output Observations
    // 每個 Agent 需要 OBS_DIM_ALIGNED (32) floats
    s->obs_out = (float*)malloc(n_agents * OBS_DIM_ALIGNED * sizeof(float));
}

int main(int argc, char *argv[]) {
    // === 設定實驗參數 ===
    int N_ENVS = 64;        // Scenario A
    int N_AGENTS = 4096;    // Heavy Load
    int STEPS = 100;



    printf("==========================================\n");
    printf("🚀 Pure C Benchmark (No Python/NumPy)\n");
    printf("   Envs: %d, Agents: %d, Steps: %d\n", N_ENVS, N_AGENTS, STEPS);
    printf("   Max OMP Levels: %d\n", omp_get_max_active_levels());
    printf("==========================================\n");

    // 1. 準備資料
    printf("Allocating memory...\n");
    EnvState *states = (EnvState*)malloc(N_ENVS * sizeof(EnvState));
    for(int i=0; i<N_ENVS; i++) {
        init_dummy_env(&states[i], 40, 40, N_AGENTS);
    }

    // 2. 暖身 (Warmup)
    printf("Warmup...\n");
    step_env_apply_actions_batch(states, N_ENVS);

    // 3. 正式測速
    printf("Running Loop...\n");
    double start_time = omp_get_wtime();

    for(int t=0; t<STEPS; t++) {
        // 這就是 Python 呼叫的那個 C 函數
        step_env_apply_actions_batch(states, N_ENVS);
        
        // 簡單交換 pointer 模擬 input/output 更新 (避免每次都只跑一樣的邏輯)
        for(int i=0; i<N_ENVS; i++) {
            // 在真實 loop 中 Python 會處理這個，這裡簡化處理
            // 把 out 複製回 in，讓鬼繼續動
            memcpy((void*)states[i].ghosts_in, states[i].ghosts_out, N_AGENTS * sizeof(AgentState));
        }
    }

    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    double throughput = (double)(N_ENVS * STEPS) / total_time;

    printf("\n✅ Done!\n");
    printf("Time: %.4f s\n", total_time);
    printf("Throughput: %.2f steps/s\n", throughput);

    // 4. 清理 (省略 free，因為程式結束 OS 會回收)
    return 0;
}