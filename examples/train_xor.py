import argparse

from tasks.xor import XOR
from .setup import experiment


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hparam",
        action="append",
        default=[
            "pop_size=50",
            "max_nds=5",
            "max_cs=25",
            "feedforward=True",
        ],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--max-rollout-steps", type=int, default=20)
    parser.add_argument("--rollout-repeats", type=int, default=12)
    parser.add_argument(
        "--normalize-obs", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-rollouts", type=int, default=100)
    parser.add_argument("--test-interval", type=int, default=25)
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
LOG_DIR = "log/xor"
LOGGER_NAME = "XOR"

LOG_INTERVAL = 20
VIZ_SERVER_PORT = 50051


if __name__ == "__main__":
    config = parse_args()
    train_task = XOR(test=False, max_steps=config.max_rollout_steps)
    test_task = XOR(test=True, max_steps=config.max_rollout_steps)
    n_inputs = train_task.obs_shape[0]
    n_outputs = train_task.act_shape[0]

    experiment(
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
