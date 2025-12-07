// csrc/step_env_apply_level3.c
// Level-3: Episode-level MPI parallelism.
// Each MPI rank runs a subset of episodes and we aggregate statistics.
//
// IMPORTANT:
// - This file does NOT change the environment dynamics.
// - It only partitions episodes across ranks and calls Level-2 (or
//   sequential) kernels inside each rank.
// - Team Member C is responsible for:
//      * Implementing the episode distribution over ranks.
//      * Measuring speedup and efficiency.
//      * Writing small correctness / conceptual tests.
//
// Owner: Team Member C.

#include "common.h"
#include <mpi.h>

// Batch step over multiple environments (Level-2 interface).
// Implemented in step_env_apply_level2.c
void step_env_apply_actions_batch_level2(
    int n_envs,
    EnvState *envs
);

/**
 * Run a given number of episodes on this MPI rank.
 *
 * Arguments:
 *   total_episodes   - total number of episodes across ALL ranks
 *   n_envs_local     - number of environments owned by THIS rank
 *   max_steps        - maximum number of steps per episode
 *   envs_local       - array of EnvState of size n_envs_local
 *
 * Assumptions:
 *   - The caller is responsible for allocating and initializing
 *     envs_local (grid pointers, ghosts buffers, etc.).
 *   - The caller also provides a way to "reset" each EnvState
 *     at the beginning of an episode.
 *
 * TODO (Team C):
 *   1. Use MPI_Comm_rank / MPI_Comm_size to compute [start, end) episode range.
 *   2. For each local episode:
 *        - Reset all local envs (envs_local[0..n_envs_local-1]).
 *        - For up to max_steps:
 *            * Fill ghost_actions / pacman_action for each env
 *              (or use a fixed policy / random policy).
 *            * Call step_env_apply_actions_batch_level2(n_envs_local, envs_local).
 *            * Optionally check if all envs are done and break early.
 *        - Accumulate statistics (e.g., total steps, number of captures).
 *   3. Use MPI_Reduce to aggregate statistics and timing across ranks.
 *   4. Use this to compute speedup vs np=1 and report in the final write-up.
 */
void run_episodes_level3_mpi(
    int total_episodes,
    int n_envs_local,
    int max_steps,
    EnvState *envs_local
) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -----------------------------
    // 1) Compute episode partition
    // -----------------------------
    int base = total_episodes / size;
    int rem  = total_episodes % size;

    // Distribute remainder episodes to the first `rem` ranks
    int local_count = base + (rank < rem ? 1 : 0);

    // Optional: compute a global start index (not strictly needed)
    int start = rank * base + (rank < rem ? rank : rem);
    int end   = start + local_count;  // [start, end) in global episode index

    // -----------------------------
    // 2) Timing start
    // -----------------------------
    double t_start = MPI_Wtime();

    // Local statistics (example: total steps over all local episodes)
    long long local_total_steps    = 0;
    long long local_total_captures = 0;  // you can increment this when env->done == 1

    // -----------------------------
    // 3) Local episode loop (skeleton)
    // -----------------------------
    for (int ep = 0; ep < local_count; ++ep) {
        // TODO (Team C): reset all local environments before each episode.
        // For example, you might have a helper:
        //   reset_env(envs_local[i]);
        // Here we just leave a placeholder.

        // Example: run at most max_steps steps
        for (int step = 0; step < max_steps; ++step) {
            // TODO (Team C):
            //   - Fill envs_local[i].ghost_actions for each env i
            //   - Fill envs_local[i].pacman_action
            //   - These can be:
            //       * random actions
            //       * simple heuristic policy
            //       * fixed scripted policy
            //
            //   - Then call the Level-2 batch step:
            step_env_apply_actions_batch_level2(n_envs_local, envs_local);

            local_total_steps++;

            // Optionally: check if all envs are done and break early.
            // bool all_done = true;
            // for (int i = 0; i < n_envs_local; ++i) {
            //     if (!envs_local[i].done) { all_done = false; break; }
            // }
            // if (all_done) break;
        }

        // TODO (Team C):
        //   - For example, count how many envs captured Pacman in this episode:
        // for (int i = 0; i < n_envs_local; ++i) {
        //     if (envs_local[i].done) {
        //         local_total_captures++;
        //     }
        // }
    }

    double t_end  = MPI_Wtime();
    double local_time = t_end - t_start;

    // -----------------------------
    // 4) Aggregate statistics
    // -----------------------------
    double max_time = 0.0;
    long long global_total_steps    = 0;
    long long global_total_captures = 0;

    MPI_Reduce(&local_time,        &max_time,            1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_steps, &global_total_steps,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_captures, &global_total_captures, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // TODO (Team C):
        // - Print / log the timing and statistics.
        // - Use max_time as the wall-clock time for the full experiment.
        // - Compare against np=1 to compute speedup and efficiency.
        //
        // Example (you can replace with proper logging):
        // printf("MPI ranks = %d, total_episodes = %d, max_time = %f s\n",
        //        size, total_episodes, max_time);
        // printf("Global total steps = %lld, global captures = %lld\n",
        //        global_total_steps, global_total_captures);
    }
}
