import argparse
import os
import jax
import jax.numpy as jnp

from tasks.gaussian import Gaussian
from .setup import experiment, load_best_policy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hparam",
        action="append",
        default=[
            "feedforward=True",
            "default_output_activation=True",
            "max_nds=20",
            "max_cs=400",
            "add_one_c_prob=0.1",
            "reenable_one_c_prob=0.03",
            "add_one_nd_prob=0.01",
            "perturb_wght_stdev=0.01",
            "perturb_bias_stdev=0.01",
        ],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--max-rollout-steps", type=int, default=2000)
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


INPUT_NAMES = ["x", "y"]
OUTPUT_NAMES = ["pred"]
LOG_DIR = "log/gaussian"
LOGGER_NAME = "Gaussian"

LOG_INTERVAL = 20
VIZ_SERVER_PORT = 50051


if __name__ == "__main__":
    config = parse_args()
    train_task = Gaussian(test=False, max_steps=config.max_rollout_steps)
    test_task = Gaussian(test=True, max_steps=config.max_rollout_steps)
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
    action_fn = jax.jit(policy.get_actions)
    best_params, best_obs_params = load_best_policy(LOG_DIR, logger)
    best_params = best_params[None, :]

    n_points = config.max_rollout_steps
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_points)
    task_state = task_reset_fn(keys)
    best_params = jnp.repeat(best_params, repeats=n_points, axis=0)
    policy_state = policy_reset_fn(task_state)
    obs = trainer._obs_normalizer.normalize_obs(task_state.obs, best_obs_params)
    task_state = task_state.replace(obs=obs)
    action, policy_state = action_fn(task_state, best_params, policy_state)
    img = Gaussian.render(task_state, action)
    png_file = os.path.join(LOG_DIR, f"{LOGGER_NAME}.png")
    img.save(png_file)
    logger.info("PNG saved to {}.".format(png_file))
