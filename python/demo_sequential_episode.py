# python/demo_sequential_episode.py
import numpy as np
from .pacman_env import PacmanEnv


def ghost_policy(obs):
    """
    Simple greedy policy: each ghost moves toward Pacman with speed=1 (fixed in C).
    """
    ghosts = obs["ghosts"]
    pac = obs["pacman"]
    actions = np.zeros(len(ghosts), dtype=np.int32)

    for i, (gx, gy) in enumerate(ghosts):
        dx = pac[0] - gx
        dy = pac[1] - gy
        if abs(dx) > abs(dy):
            actions[i] = 4 if dx > 0 else 3  # right or left
        else:
            actions[i] = 2 if dy > 0 else 1  # down or up

    return actions


def pacman_policy(obs, step_idx: int):
    """
    Toy policy for Pacman:
    - Always moves to the right.
    - Uses speed 2 on even steps, speed 1 on odd steps.
    This is just to demonstrate that pacman_speed works.
    """
    action = 4  # right
    speed = 2 if (step_idx % 2 == 0) else 1
    return action, speed


if __name__ == "__main__":
    # simple empty 40x40 grid (0 = empty)
    grid = np.zeros((40, 40), dtype=np.int32)

    env = PacmanEnv(grid, n_agents=14, max_steps=200)
    obs = env.reset()

    done = False
    ep_return = 0.0
    step = 0

    while not done and step < 50:
        g_act = ghost_policy(obs)
        p_act, p_speed = pacman_policy(obs, step)

        obs, reward, done, info = env.step(
            ghost_actions=g_act,
            pacman_action=p_act,
            pacman_speed=p_speed,
        )

        ep_return += reward["pacman"]
        step += 1

    print(f"Episode finished in {step} steps, pacman_return={ep_return}")
