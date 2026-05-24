# Like metamaze, after reaching the goal it teleports the agent to a new position,
# while keeping the goal fixed

import argparse
import os
import jax

from tasks.gymnax import GymnaxTask
from .setup import experiment, load_best_policy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hparam",
        action="append",
        default=[
            "feedforward_only=False",
            "max_nds=25",
            "max_cs=625",
            # "default_output_activation=True",
            # "add_one_c_prob=0.3",
            # "reenable_one_c_prob=0.1",
            # "add_one_nd_prob=0.03",
            # "perturb_wght_stdev=0.1",
            # "perturb_bias_stdev=0.1",
            "feedforward=False",
            "pop_size=1000",
            "max_nds=40",
            "max_cs=1600",
        ],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--generations", type=int, default=3000)
    parser.add_argument("--max-rollout-steps", type=int, default=500)
    parser.add_argument("--rollout-repeats", type=int, default=12)
    parser.add_argument(
        "--normalize-obs", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-rollouts", type=int, default=100)
    parser.add_argument("--test-interval", type=int, default=50)
    parser.add_argument(
        "--visualize-training",
        action="store_true",
        help="Communicate with a grpc server.",
    )
    parser.add_argument("--neat-checkpoint-path", type=str, default=None)
    parser.add_argument("--obs-normalizer-checkpoint-path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    config, _ = parser.parse_known_args()
    return config


GYMNAX_ENV = "PointRobot-misc"
INPUT_NAMES = ["x", "y", "last_r", "last_ax", "last_ay", "time"]
OUTPUT_NAMES = ["ax", "ay"]
LOG_DIR = "log/gymnax/PointRobot"
LOGGER_NAME = "PointRobot"

LOG_INTERVAL = 20
VIZ_SERVER_PORT = 50051


if __name__ == "__main__":
    config = parse_args()
    env_param_overrides = {
        "max_steps_in_episode": config.max_rollout_steps,
        "dense_reward": True,
    }
    train_task = GymnaxTask(
        # This task doesn't return done until max_steps_in_episode
        env_name=GYMNAX_ENV,
        max_steps=config.max_rollout_steps,
        test=False,
        env_param_overrides=env_param_overrides,
    )
    test_task = GymnaxTask(
        env_name=GYMNAX_ENV,
        max_steps=config.max_rollout_steps,
        test=True,
        env_param_overrides=env_param_overrides,
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

    logger.info(f"Generating episodes of the record fitness individual...")
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params, best_obs_params = load_best_policy(LOG_DIR, logger)
    best_params = best_params[None, :]
    key = jax.random.PRNGKey(0)[None, :]

    def frame(state):
        state = jax.tree.map(lambda x: x[0], state.state)
        return test_task.render(state)

    screens = []
    for ep in range(5):
        key = jax.random.PRNGKey(ep)[None, :]
        task_state = task_reset_fn(key)
        policy_state = policy_reset_fn(task_state)
        screens.append(frame(task_state))
        for _ in range(config.max_rollout_steps):
            obs = trainer._obs_normalizer.normalize_obs(task_state.obs, best_obs_params)
            task_state = task_state.replace(obs=obs)
            action, policy_state = action_fn(task_state, best_params, policy_state)
            task_state, reward, is_done = step_fn(task_state, action)
            screens.append(frame(task_state))
            if bool(is_done[0]):
                break

    gif_file = os.path.join(LOG_DIR, f"{LOGGER_NAME}.gif")
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=80, loop=0
    )
    logger.info(f"GIF saved to {gif_file}.")
