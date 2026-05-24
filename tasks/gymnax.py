# A wrapper of gymnax environments into EvoJAX's VectorizedTask template.

import jax
import jax.numpy as jnp
import gymnax
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from PIL import Image


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: Any
    steps: jnp.int32
    key: jnp.ndarray


class GymnaxTask(VectorizedTask):
    def __init__(
        self,
        env_name: str,
        max_steps: int = 1000,
        test: bool = False,
        restart_on_done_mid_rollout: bool = False,
        env_param_overrides: dict | None = None,
    ):
        self.max_steps = max_steps
        self.test = test
        self.restart_on_done_mid_rollout = restart_on_done_mid_rollout

        self.base_env, self.env_params = gymnax.make(env_name)
        if env_param_overrides is not None:
            self.env_params = self.env_params.replace(**env_param_overrides)
        obs_space = self.base_env.observation_space(self.env_params)
        self.obs_shape = (int(jnp.prod(jnp.array(obs_space.shape))),)
        act_space = self.base_env.action_space(self.env_params)
        if hasattr(act_space, "n"):
            self.is_discrete_act_space = True
            self.act_shape = (int(act_space.n),)
        else:
            self.is_discrete_act_space = False
            self.act_shape = (int(jnp.prod(jnp.array(act_space.shape))),)

        def reset_fn(key):
            key, k1 = jax.random.split(key)
            obs, state = self.base_env.reset(k1, self.env_params)
            return State(
                obs=obs.ravel().astype(jnp.float32),
                state=state,
                steps=jnp.zeros((), dtype=jnp.int32),
                key=key,
            )

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            key, k1 = jax.random.split(state.key)
            if self.is_discrete_act_space:
                action = jnp.argmax(action).astype(jnp.int32)
            obs, current_state, reward, is_done, _ = self.base_env.step(
                k1, state.state, action, self.env_params
            )
            steps = state.steps + 1
            if (
                self.restart_on_done_mid_rollout
            ):  # gymnax restarts internally when it returns done
                is_done = steps >= self.max_steps
            else:
                is_done = jnp.logical_or(is_done, steps >= self.max_steps)
            steps = jnp.where(is_done, jnp.zeros((), jnp.int32), steps)
            return (
                State(
                    obs=obs.ravel().astype(jnp.float32),
                    state=current_state,
                    steps=steps,
                    key=key,
                ),
                reward.astype(jnp.float32),
                is_done,
            )

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)

    def render(self, env_state) -> Image.Image:
        rendered = self.base_env.render(env_state, self.env_params)
        if isinstance(rendered, tuple):
            fig = rendered[0]
        elif hasattr(rendered, "canvas"):
            fig = rendered
        elif hasattr(rendered, "figure"):
            fig = rendered.figure
        else:
            raise TypeError(
                f"Unsupported render output from {self.base_env.name}: "
                f"{type(rendered).__name__}"
            )
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)
        return img
