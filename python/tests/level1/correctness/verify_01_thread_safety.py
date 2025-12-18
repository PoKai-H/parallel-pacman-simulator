import os
import sys
import ctypes as C
import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
# speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)


from pacman_env import PacmanEnv, OBS_DIM


def _set_omp_threads(n: int) -> None:
    """Best-effort: set OpenMP threads for current process."""
    os.environ["OMP_NUM_THREADS"] = str(int(n))
    for so in ("libgomp.so.1", "libomp.so", "libiomp5.so"):
        try:
            rt = C.CDLL(so)
            if hasattr(rt, "omp_set_dynamic"):
                rt.omp_set_dynamic(0)
            if hasattr(rt, "omp_set_num_threads"):
                rt.omp_set_num_threads(int(n))
            return
        except OSError:
            continue


def _make_wall_grid(h: int = 40, w: int = 40) -> np.ndarray:
    """
    Make a simple grid with an internal wall to trigger is_wall / collision logic.
    0 = empty, 1 = wall.
    """
    g = np.zeros((h, w), dtype=np.int32)
    mid = w // 2
    g[:, mid] = 1                 # vertical wall
    g[h // 2, 1:w-1] = 1          # horizontal wall
    return g


def _rollout(
    *,
    n_agents: int,
    steps: int,
    seed: int,
    threads: int,
    dead_pattern: bool = False,
    all_dead: bool = False,
    grid: np.ndarray | None = None,
    pacman_speed: int = 2,
    init_hook=None,
    ghost_actions_table: np.ndarray | None = None,
    pacman_actions: np.ndarray | None = None,
):
    """
    Run a rollout and record all outputs for exact match checking.
    - deterministic rand_pool by np.random.seed(seed) before env init
    - deterministic actions by fixed table OR rng(seed+12345)
    - optional init_hook(env): customize initial state after reset
    """
    _set_omp_threads(threads)

    np.random.seed(seed)  # 固定 rand_pool
    if grid is None:
        grid = np.zeros((40, 40), dtype=np.int32)

    env = PacmanEnv(grid, n_agents=n_agents, max_steps=max(steps, 200))
    _ = env.reset()

    # allow custom initialization (positions, etc.)
    if init_hook is not None:
        init_hook(env)

    # dead patterns
    if dead_pattern:
        for i in range(n_agents):
            if i % 2 == 0:
                env.ptr_ghosts_in[i].alive = 0

    if all_dead:
        for i in range(n_agents):
            env.ptr_ghosts_in[i].alive = 0

    # actions
    if ghost_actions_table is None or pacman_actions is None:
        rng = np.random.default_rng(seed + 12345)
        if ghost_actions_table is None:
            ghost_actions_table = rng.integers(0, 5, size=(steps, n_agents), dtype=np.int32)
        if pacman_actions is None:
            pacman_actions = rng.integers(0, 5, size=(steps,), dtype=np.int32)
    else:
        ghost_actions_table = np.asarray(ghost_actions_table, dtype=np.int32)
        pacman_actions = np.asarray(pacman_actions, dtype=np.int32)
        assert ghost_actions_table.shape == (steps, n_agents)
        assert pacman_actions.shape == (steps,)

    pac_xy = []
    ghosts_xya = []
    ghost_rewards = []
    pacman_rewards = []
    dones = []
    obs_list = []

    done = False
    for t in range(steps):
        if done:
            break

        obs, reward, done, _info = env.step(
            ghost_actions_table[t],
            int(pacman_actions[t]),
            pacman_speed=int(pacman_speed),
        )

        pac_xy.append([env.pac_x, env.pac_y])

        g = np.zeros((n_agents, 3), dtype=np.int32)
        for i in range(n_agents):
            g[i, 0] = int(env.ptr_ghosts_in[i].x)
            g[i, 1] = int(env.ptr_ghosts_in[i].y)
            g[i, 2] = int(env.ptr_ghosts_in[i].alive)
        ghosts_xya.append(g)

        ghost_rewards.append(reward["ghosts"].astype(np.float32))
        pacman_rewards.append(np.float32(reward["pacman"]))
        dones.append(int(done))
        obs_list.append(obs["ghost_tensor"].astype(np.float32))

    return {
        "pac_xy": np.asarray(pac_xy, dtype=np.int32),
        "ghosts_xya": np.asarray(ghosts_xya, dtype=np.int32),
        "ghost_rewards": np.asarray(ghost_rewards, dtype=np.float32),
        "pacman_rewards": np.asarray(pacman_rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.int32),
        "obs": np.asarray(obs_list, dtype=np.float32),
        "obs_dim": OBS_DIM,
    }


def _assert_same(a, b):
    np.testing.assert_array_equal(a["pac_xy"], b["pac_xy"])
    np.testing.assert_array_equal(a["ghosts_xya"], b["ghosts_xya"])
    np.testing.assert_array_equal(a["dones"], b["dones"])

    np.testing.assert_allclose(a["ghost_rewards"], b["ghost_rewards"], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(a["pacman_rewards"], b["pacman_rewards"], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(a["obs"], b["obs"], rtol=0.0, atol=1e-6)


# ============================================================
# L1-1: threads scaling consistency
# ============================================================
@pytest.mark.parametrize("threads", [2, 4, 8, 16])
@pytest.mark.parametrize("steps", [1, 5, 20])
def test_level1_threads_match_baseline(threads, steps):
    baseline = _rollout(n_agents=16, steps=steps, seed=0, threads=1)
    test = _rollout(n_agents=16, steps=steps, seed=0, threads=threads)
    _assert_same(baseline, test)


# ============================================================
# L1-2: n_agents scaling consistency
# ============================================================
@pytest.mark.parametrize("n_agents", [16, 32, 64, 128])
@pytest.mark.parametrize("threads", [2, 8])
def test_level1_n_agents_scaling_match_baseline(n_agents, threads):
    baseline = _rollout(n_agents=n_agents, steps=5, seed=1, threads=1)
    test = _rollout(n_agents=n_agents, steps=5, seed=1, threads=threads)
    _assert_same(baseline, test)


# ============================================================
# L1-3: edge case：half dead
# ============================================================
@pytest.mark.parametrize("threads", [2, 8, 16])
def test_level1_dead_ghosts_match_baseline(threads):
    baseline = _rollout(n_agents=32, steps=5, seed=2, threads=1, dead_pattern=True)
    test = _rollout(n_agents=32, steps=5, seed=2, threads=threads, dead_pattern=True)
    _assert_same(baseline, test)


# ============================================================
# L1-3b: edge case：all dead
# ============================================================
@pytest.mark.parametrize("threads", [2, 8, 16])
def test_level1_all_dead_ghosts_match_baseline(threads):
    baseline = _rollout(n_agents=32, steps=5, seed=3, threads=1, all_dead=True)
    test = _rollout(n_agents=32, steps=5, seed=3, threads=threads, all_dead=True)

    _assert_same(baseline, test)
    assert np.all(baseline["obs"] == 0.0), "All-dead ghosts should produce zero observations"


# ============================================================
# must be done
# ============================================================
def _init_force_capture(env: PacmanEnv):
    env.pac_x = env.w // 2
    env.pac_y = env.h // 2
    env.ptr_ghosts_in[0].alive = 1
    env.ptr_ghosts_in[0].x = env.pac_x
    env.ptr_ghosts_in[0].y = env.pac_y


@pytest.mark.parametrize("threads", [2, 8, 16])
def test_level1_capture_branch_match_baseline(threads):
    steps = 1
    n_agents = 16
    ghost_actions = np.zeros((steps, n_agents), dtype=np.int32)  # all stay
    pac_actions = np.zeros((steps,), dtype=np.int32)            # pacman stay

    baseline = _rollout(
        n_agents=n_agents, steps=steps, seed=10, threads=1,
        pacman_speed=0, init_hook=_init_force_capture,
        ghost_actions_table=ghost_actions, pacman_actions=pac_actions,
    )
    test = _rollout(
        n_agents=n_agents, steps=steps, seed=10, threads=threads,
        pacman_speed=0, init_hook=_init_force_capture,
        ghost_actions_table=ghost_actions, pacman_actions=pac_actions,
    )

    _assert_same(baseline, test)
    assert baseline["dones"][0] == 1
    assert float(baseline["pacman_rewards"][0]) == -10.0


# ============================================================
# is_wall / no collision
# ============================================================
def _init_wall_positions(env: PacmanEnv):
    mid = env.w // 2
    env.pac_x = mid - 1
    env.pac_y = env.h // 2

    env.ptr_ghosts_in[0].alive = 1
    env.ptr_ghosts_in[0].x = mid - 1
    env.ptr_ghosts_in[0].y = env.h // 2 - 1


    if env.n_agents >= 2:
        env.ptr_ghosts_in[1].alive = 1
        env.ptr_ghosts_in[1].x = 0
        env.ptr_ghosts_in[1].y = 0


@pytest.mark.parametrize("threads", [2, 8, 16])
def test_level1_wall_and_bounds_match_baseline(threads):
    grid = _make_wall_grid(40, 40)
    steps = 5
    n_agents = 16

    # ghost[0]=right(4), ghost[1]=left(3), others=stay(0)
    ghost_actions = np.zeros((steps, n_agents), dtype=np.int32)
    ghost_actions[:, 0] = 4
    if n_agents >= 2:
        ghost_actions[:, 1] = 3

    # pacman keep hitting the wall on the right
    pac_actions = np.full((steps,), 4, dtype=np.int32)

    baseline = _rollout(
        n_agents=n_agents, steps=steps, seed=11, threads=1,
        grid=grid, pacman_speed=2, init_hook=_init_wall_positions,
        ghost_actions_table=ghost_actions, pacman_actions=pac_actions,
    )
    test = _rollout(
        n_agents=n_agents, steps=steps, seed=11, threads=threads,
        grid=grid, pacman_speed=2, init_hook=_init_wall_positions,
        ghost_actions_table=ghost_actions, pacman_actions=pac_actions,
    )
    _assert_same(baseline, test)


# ============================================================
# pacman_speed (0,1,2) threads-invariant
# ============================================================
@pytest.mark.parametrize("pacman_speed", [0, 1, 2])
@pytest.mark.parametrize("threads", [2, 8, 16])
def test_level1_pacman_speed_match_baseline(pacman_speed, threads):
    baseline = _rollout(n_agents=16, steps=10, seed=12, threads=1, pacman_speed=pacman_speed)
    test = _rollout(n_agents=16, steps=10, seed=12, threads=threads, pacman_speed=pacman_speed)
    _assert_same(baseline, test)


# ============================================================
# Multiple Random seeds
# ============================================================
@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("threads", [8, 16])
def test_level1_multiple_seeds_match_baseline(seed, threads):
    baseline = _rollout(n_agents=16, steps=8, seed=seed, threads=1)
    test = _rollout(n_agents=16, steps=8, seed=seed, threads=threads)
    _assert_same(baseline, test)


# ============================================================
# stress test: large n_agents + many steps
# ============================================================
@pytest.mark.parametrize("threads", [16])
def test_level1_stress_large_agents_steps_match_baseline(threads):
    baseline = _rollout(n_agents=128, steps=30, seed=99, threads=1)
    test = _rollout(n_agents=128, steps=30, seed=99, threads=threads)
    _assert_same(baseline, test)