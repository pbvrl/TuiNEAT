import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from PIL import Image, ImageDraw


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


class Circle(VectorizedTask):
    """
    x^2 + y^2 <= 1  ->  1
    x^2 + y^2 >  1  ->  0
    """

    def __init__(self, max_steps: int = 100, test: bool = False):
        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([2])
        self.act_shape = tuple([1])

        def reset_fn(key):
            key, k1 = jax.random.split(key)
            obs = jax.random.uniform(k1, shape=(2,), minval=-1.5, maxval=1.5)
            return State(obs=obs, steps=jnp.zeros((), dtype=int), key=key)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            key, k1 = jax.random.split(state.key, 2)
            inside_circle = jnp.sum(state.obs**2) <= 1.0
            reward = jnp.squeeze(
                ((jax.nn.sigmoid(action) > 0.5) == inside_circle).astype(jnp.float32)
            )
            current_obs = jax.random.uniform(k1, shape=(2,), minval=-1.5, maxval=1.5)
            steps = state.steps + 1
            is_done = steps >= self.max_steps
            return State(obs=current_obs, steps=steps, key=key), reward, is_done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)

    @staticmethod
    def render(state, action) -> Image.Image:
        img_size = 900
        img = Image.new("RGB", (img_size, img_size), color="white")
        draw = ImageDraw.Draw(img)

        radius = 5
        scale = img_size // 5
        for i in range(len(state.obs)):
            obs = state.obs[i]
            action_inside = jax.nn.sigmoid(action[i]) > 0.5
            obs_inside = (obs[0] ** 2 + obs[1] ** 2) <= 1.0
            wrong = action_inside ^ obs_inside

            x = int(obs[0] * scale + img_size / 2)
            y = int(-obs[1] * scale + img_size / 2)  # (0,0) in PIL is the top left
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill="red" if wrong else "green",
            )

        radius = scale
        center = (img_size // 2, img_size // 2)
        draw.ellipse(
            [
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] + radius),
            ],
            outline="black",
            width=2,
        )

        draw.text((10, 10), "Green: correct", fill="black")
        draw.text((10, 30), "Red: wrong", fill="black")
        return img
