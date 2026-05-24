import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
import matplotlib.pyplot as plt
from PIL import Image


SIGMA_SQUARED = 1.0
XY_BOUND = 2.0


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


class Gaussian(VectorizedTask):
    """
    f(x,y) = exp(-((x^2 + y^2) / sigma**2))
    The function peaks at (0,0) with value 1.0.
    """

    def __init__(self, max_steps: int = 100, test: bool = False):
        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([2])
        self.act_shape = tuple([1])

        def reset_fn(key):
            key, k1 = jax.random.split(key)
            obs = jax.random.uniform(k1, shape=(2,), minval=-XY_BOUND, maxval=XY_BOUND)
            return State(obs=obs, steps=jnp.zeros((), dtype=int), key=key)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            key, k1 = jax.random.split(state.key, 2)
            target = jnp.exp(-jnp.sum(state.obs**2) / SIGMA_SQUARED)
            reward = -jnp.mean(jnp.square(action - target))
            next_obs = jax.random.uniform(
                k1, shape=(2,), minval=-XY_BOUND, maxval=XY_BOUND
            )
            steps = state.steps + 1
            is_done = steps >= self.max_steps
            return (State(obs=next_obs, steps=steps, key=key), reward, is_done)

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)

    @staticmethod
    def render(state, action) -> Image.Image:
        xs = ys = np.linspace(-XY_BOUND, XY_BOUND, 200)
        xv, yv = np.meshgrid(xs, ys)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        f = np.exp(-(xv**2 + yv**2) / SIGMA_SQUARED)
        im = axes[0].imshow(
            f, extent=(-XY_BOUND, XY_BOUND, -XY_BOUND, XY_BOUND), vmin=0.0, vmax=1.0
        )
        axes[0].set_title("Target: exp(-(x²+y²)/σ²)")

        obs = np.asarray(state.obs)
        actions = np.asarray(action).reshape(-1)
        axes[1].scatter(obs[:, 0], obs[:, 1], c=actions, vmin=0.0, vmax=1.0, s=30)
        axes[1].set(
            xlim=(-XY_BOUND, XY_BOUND), ylim=(-XY_BOUND, XY_BOUND), aspect="equal"
        )

        fig.colorbar(im, ax=axes)
        for ax in axes:
            ax.set(xticks=[], yticks=[])

        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)
        return img
