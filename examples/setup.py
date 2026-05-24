import ast
import os
import shutil
import jax.numpy as jnp
import numpy as np
from evojax import Trainer
from evojax import util

from src.hyperparams import Hyperparams
from src.solver import NeatAlgorithm
from src.policy import NeatPolicyCTRNN, NeatPolicyFeedforward


def experiment(
    config,
    train_task,
    test_task,
    n_inputs,
    n_outputs,
    input_names,
    output_names,
    log_dir,
    logger_name,
    log_interval,
    viz_port,
):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(name=logger_name, log_dir=log_dir, debug=config.debug)
    logger.info(logger_name)
    logger.info("=" * 30)

    hparams_dict = parse_hparams(config.hparam)
    hparams = Hyperparams(
        n_inputs=n_inputs, n_outputs=n_outputs, seed=config.seed, **hparams_dict
    )
    viz_client = None
    if config.visualize_training:
        from src.extra_modules.training_visualizer.client import (
            TrainingVisualizerClient,
        )

        viz_client = TrainingVisualizerClient(
            hparams=hparams,
            rollout_repeats=config.rollout_repeats,
            max_rollout_steps=config.max_rollout_steps,
            task_name=logger_name,
            input_names=input_names,
            output_names=output_names,
            feedforward=hparams.feedforward,
            normalize_obs=config.normalize_obs,
            server_port=viz_port,
        )
    solver = NeatAlgorithm(
        hparams=hparams,
        logger=logger,
        neat_checkpoint_path=config.neat_checkpoint_path,
        viz_client=viz_client,
        log_dir=log_dir,
    )
    if hparams.feedforward:
        policy = NeatPolicyFeedforward(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            max_nds=hparams.max_nds,
            logger=logger,
        )
    else:
        policy = NeatPolicyCTRNN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            max_nds=hparams.max_nds,
            ctrnn_integration_steps=hparams.ctrnn_integration_steps,
            logger=logger,
        )
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.generations,
        log_interval=log_interval,
        test_interval=config.test_interval,
        n_repeats=config.rollout_repeats,
        n_evaluations=config.test_rollouts,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
        normalize_obs=config.normalize_obs,
    )
    if config.obs_normalizer_checkpoint_path is not None:
        with np.load(config.obs_normalizer_checkpoint_path) as checkpoint:
            obs_params = jnp.asarray(checkpoint["obs_params"])
        trainer.sim_mgr.obs_params = obs_params

    trainer.run(demo_mode=False)

    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    return trainer, policy, logger


def parse_hparams(items):
    return dict(parse_hparam(item) for item in items)


def parse_hparam(item):
    key, _, raw = item.partition("=")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    return key.strip(), value


def load_best_policy(log_dir, logger=None):
    checkpoint_file = os.path.join(log_dir, "best.npz")
    with np.load(checkpoint_file) as checkpoint:
        params = jnp.asarray(checkpoint["params"])
        obs_params = jnp.asarray(checkpoint["obs_params"])
    if logger is not None:
        logger.info(f"Loaded best policy from {checkpoint_file}.")
    return params, obs_params
