import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import jax
import numpy as np
import pickle

from tasks.mujoco import MujocoTask
from .setup import experiment, load_best_policy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hparam",
        action="append",
        default=[
            "max_nds=70",
            "max_cs=4900",
            "pop_size=1000",
            "feedforward=True",
            "default_output_activation=True",
            "add_one_c_prob=0.3",
            "reenable_one_c_prob=0.1",
            "add_one_nd_prob=0.01",
            "change_act_prob=0.003",
            "change_agg_prob=0.001",
            "perturb_wght_stdev=0.04",
            "perturb_bias_stdev=0.03",
            "replace_wght_prob=0.0",
            "replace_bias_prob=0.0",
            "disable_c_prob=0.0",
            "erase_split_c=False",
            "crossover_rate=0.3",
            "enabled_recessiveness_prob=0.7",
            "base_stagnation_age=50",
            "stagnation_age_powerlaw_growth=0.2",
            "n_stagnation_exempt_top_species=2",
        ],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--max-rollout-steps", type=int, default=1000)
    parser.add_argument("--rollout-repeats", type=int, default=6)
    parser.add_argument(
        "--normalize-obs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-rollouts", type=int, default=100)
    parser.add_argument("--test-interval", type=int, default=100)
    parser.add_argument(
        "--visualize-training",
        action="store_true",
        help="Communicate with a grpc server",
    )
    parser.add_argument("--neat-checkpoint-path", type=str, default=None)
    parser.add_argument("--obs-normalizer-checkpoint-path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    config, _ = parser.parse_known_args()
    return config


MUJOCO_ENV = "CheetahRun"
INPUT_NAMES = [
    "F. tip z",
    "F. tip θ",
    "B. thigh θ",
    "B. shin θ",
    "B. foot θ",
    "F. thigh θ",
    "F. shin θ",
    "F. foot θ",
    "F. tip vx",
    "F. tip vz",
    "F. tip ω",
    "B. thigh ω",
    "B. shin ω",
    "B. foot ω",
    "F. thigh ω",
    "F. shin ω",
    "F. foot ω",
]
OUTPUT_NAMES = [
    "B. thigh",
    "B. shin",
    "B. foot",
    "F. thigh",
    "F. shin",
    "F. foot",
]
LOG_DIR = f"log/mujoco/{MUJOCO_ENV}"
LOGGER_NAME = MUJOCO_ENV

LOG_INTERVAL = 20
VIZ_SERVER_PORT = 50051


if __name__ == "__main__":
    config = parse_args()
    key = jax.random.PRNGKey(config.seed)
    train_task = MujocoTask(
        env_name=MUJOCO_ENV, key=key, max_steps=config.max_rollout_steps, test=False
    )
    test_task = MujocoTask(
        env_name=MUJOCO_ENV, key=key, max_steps=config.max_rollout_steps, test=True
    )
    n_inputs = train_task.obs_shape[0]
    n_outputs = train_task.act_shape[0]

    trainer, policy, logger = experiment(
        config,
        train_task,
        test_task,
        n_inputs,
        n_outputs,
        INPUT_NAMES,
        OUTPUT_NAMES,
        LOG_DIR,
        LOGGER_NAME,
        LOG_INTERVAL,
        VIZ_SERVER_PORT,
    )

    logger.info(f"Generating episode of the record fitness individual...")
    best_params, best_obs_params = load_best_policy(LOG_DIR, logger)
    best_params = best_params[None, :]
    task_state = test_task.reset(key[None, :])
    policy_state = policy.reset(task_state)

    trajectory_states = []
    total_reward = 0.0
    logger.info(f"Recording up to {config.max_rollout_steps} steps...")
    for step in range(config.max_rollout_steps):
        unbatched_state = jax.tree.map(lambda x: x[0], task_state.state)
        trajectory_states.append(unbatched_state)
        obs = trainer._obs_normalizer.normalize_obs(task_state.obs, best_obs_params)
        task_state = task_state.replace(obs=obs)
        action, policy_state = policy.get_actions(task_state, best_params, policy_state)
        task_state, reward, is_done = test_task.step(task_state, action)
        total_reward += float(reward[0])
        if is_done[0]:
            break
    logger.info(f"Episode finished after {len(trajectory_states)} steps.")
    logger.info(f"Episode fitness: {total_reward:.2f}")
    trajectory = jax.tree.map(np.asarray, trajectory_states)
    trajectory_file = os.path.join(LOG_DIR, "trajectory.pkl")
    with open(trajectory_file, "wb") as f:
        pickle.dump(
            {
                "trajectory": trajectory,
                "reward": total_reward,
                "dt": test_task._base_env.dt,
            },
            f,
        )
    logger.info(f"  Saved to: {trajectory_file}")
