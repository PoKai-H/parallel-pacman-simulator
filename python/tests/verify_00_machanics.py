import pytest
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from pacman_env import PacmanVecEnv, PacmanEnv

# =================================================================
# Group A: basic tests
# =================================================================

def test_init_basic():
    """T1: Not crashing while initialization"""
    env = PacmanVecEnv(np.zeros((20, 20), dtype=np.int8), n_envs=2, n_agents=4)
    assert env is not None

def test_obs_shape_consistency():
    """T2: observation shape consistency"""
    n_envs, n_agents = 4, 10
    env = PacmanVecEnv(np.zeros((20, 20), dtype=np.int8), n_envs=n_envs, n_agents=n_agents)
    obs = env.reset()
    assert obs.shape == (n_envs, n_agents, 17)

def test_action_space_bounds():
    """T3: illigel actions"""
    env = PacmanVecEnv(np.zeros((10, 10), dtype=np.int8), n_envs=1, n_agents=1)
    env.reset()
    actions = np.full((1, 1), 99, dtype=np.int32)
    env.step(actions) 

def test_reward_structure():
    """T4: test Reward format (pacman, ghosts, step)"""
    env = PacmanEnv(np.zeros((10, 10), dtype=np.int32), n_agents=1)
    env.reset()
    obs, reward, done, _ = env.step(np.array([0], dtype=np.int32), 0)
    assert 'pacman' in reward
    assert 'ghosts' in reward
    assert isinstance(reward['pacman'], (float, int))

def test_reset_mechanism():
    """T5: checking state changes after reset"""
    env = PacmanVecEnv(np.zeros((10, 10), dtype=np.int8), n_envs=1, n_agents=1)
    obs1 = env.reset()
    
    assert np.any(obs1 != -999) 


def test_wall_collision():
    """T6: hitting the wall"""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 5] = 1 # Wall
    env = PacmanEnv(grid, n_agents=1)
    env.reset()
    env.ptr_ghosts_in[0].x = 5
    env.ptr_ghosts_in[0].y = 4
    env.step(np.array([2], dtype=np.int32), 0)
    assert env.ptr_ghosts_in[0].y == 4

def test_pacman_capture():
    """T7: Pacman Captured Reward"""
    grid = np.zeros((10, 10), dtype=np.int32)
    env = PacmanEnv(grid, n_agents=1)
    env.reset()
    env.pac_x, env.pac_y = 5, 5
    env.ptr_ghosts_in[0].x = 5
    env.ptr_ghosts_in[0].y = 5
    obs, reward, done, _ = env.step(np.array([0], dtype=np.int32), 0)
    assert done == True
    assert reward['pacman'] < 0 


@pytest.mark.parametrize("grid_size", [10, 50, 100, 200])
@pytest.mark.parametrize("n_agents", [1, 16, 128, 1024, 4096])
def test_scalability_stress(grid_size, n_agents):
    """
    T8-T27: stress tests
    Combinations: 4 maps * 5 # of agents = 20 tests
    
    """
    try:
        env = PacmanVecEnv(np.zeros((grid_size, grid_size), dtype=np.int8), n_envs=2, n_agents=n_agents)
        obs = env.reset()
        actions = np.random.randint(0, 5, size=(2, n_agents), dtype=np.int32)
        next_obs, rewards, dones, _ = env.step(actions)
        
        assert next_obs.shape == (2, n_agents, 17)
        assert len(rewards) == 2
    except Exception as e:
        pytest.fail(f"Crash at Grid={grid_size}, Agents={n_agents}: {e}")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))