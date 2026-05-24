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

    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--ctrl-dt", type=float, default=0.01)
    parser.add_argument(
        "--hparam",
        action="append",
        default=[
            "max_nds=100",
            "max_cs=10000",
            "feedforward=False",
            "default_output_activation=True",
        ],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--generations", type=int, default=700)
    parser.add_argument("--max-rollout-steps", type=int, default=250)
    parser.add_argument("--rollout-repeats", type=int, default=12)
    parser.add_argument(
        "--normalize-obs", action=argparse.BooleanOptionalAction, default=False
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


MUJOCO_ENV = "PandaPickCube"
INPUT_NAMES = [
    "Joint 1 θ",
    "Joint 2 θ",
    "Joint 3 θ",
    "Joint 4 θ",
    "Joint 5 θ",
    "Joint 6 θ",
    "Joint 7 θ",
    "Gripper L θ",
    "Gripper R θ",
    "Cube x",
    "Cube y",
    "Cube z",
    "Cube qw",
    "Cube qx",
    "Cube qy",
    "Cube qz",
    "Joint 1 ω",
    "Joint 2 ω",
    "Joint 3 ω",
    "Joint 4 ω",
    "Joint 5 ω",
    "Joint 6 ω",
    "Joint 7 ω",
    "Gripper L ω",
    "Gripper R ω",
    "Cube vx",
    "Cube vy",
    "Cube vz",
    "Cube wx",
    "Cube wy",
    "Cube wz",
    "Gripper x",
    "Gripper y",
    "Gripper z",
    "Gripper rotation r10",
    "Gripper rotation r11",
    "Gripper rotation r12",
    "Gripper rotation r20",
    "Gripper rotation r21",
    "Gripper rotation r22",
    "Cube rotation r10",
    "Cube rotation r11",
    "Cube rotation r12",
    "Cube rotation r20",
    "Cube rotation r21",
    "Cube rotation r22",
    "Cube to gripper dx",
    "Cube to gripper dy",
    "Cube to gripper dz",
    "Target to cube dx",
    "Target to cube dy",
    "Target to cube dz",
    "Target-cube rotation r00",
    "Target-cube rotation r01",
    "Target-cube rotation r02",
    "Target-cube rotation r10",
    "Target-cube rotation r11",
    "Target-cube rotation r12",
    "Joint 1 ctrl error",
    "Joint 2 ctrl error",
    "Joint 3 ctrl error",
    "Joint 4 ctrl error",
    "Joint 5 ctrl error",
    "Joint 6 ctrl error",
    "Joint 7 ctrl error",
    "Gripper L ctrl error",
]
OUTPUT_NAMES = [
    "Joint 1 ctrl delta",
    "Joint 2 ctrl delta",
    "Joint 3 ctrl delta",
    "Joint 4 ctrl delta",
    "Joint 5 ctrl delta",
    "Joint 6 ctrl delta",
    "Joint 7 ctrl delta",
    "Gripper ctrl delta",
]
LOG_DIR = f"log/mujoco/{MUJOCO_ENV}"
LOGGER_NAME = MUJOCO_ENV

LOG_INTERVAL = 20
VIZ_SERVER_PORT = 50051


if __name__ == "__main__":
    config = parse_args()
    key = jax.random.PRNGKey(config.seed)
    train_task = MujocoTask(
        env_name=MUJOCO_ENV,
        key=key,
        max_steps=config.max_rollout_steps,
        test=False,
        sim_dt=config.sim_dt,
        ctrl_dt=config.ctrl_dt,
    )
    test_task = MujocoTask(
        env_name=MUJOCO_ENV,
        key=key,
        max_steps=config.max_rollout_steps,
        test=True,
        sim_dt=config.sim_dt,
        ctrl_dt=config.ctrl_dt,
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
