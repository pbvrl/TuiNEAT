import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from PIL import Image, ImageDraw


N_TURNS = 2.0
NOISE = 0.1


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    target: jnp.int32
    steps: jnp.ndarray
    key: jnp.ndarray


def spiral(k1, k2):
    theta = jax.random.uniform(k1, shape=(), minval=0, maxval=2 * jnp.pi * N_TURNS)
    r = theta / (2 * jnp.pi * N_TURNS)

    dr_dtheta = 1.0 / (2 * jnp.pi * N_TURNS)
    dx_dtheta = dr_dtheta * jnp.cos(theta) - r * jnp.sin(theta)
    dy_dtheta = dr_dtheta * jnp.sin(theta) + r * jnp.cos(theta)
    tangent_length = jnp.sqrt(dx_dtheta**2 + dy_dtheta**2)
    tangent_unit_x = dx_dtheta / tangent_length
    tangent_unit_y = dy_dtheta / tangent_length
    normal_x = -tangent_unit_y
    normal_y = tangent_unit_x

    noise = jax.random.uniform(k2, shape=(), minval=-NOISE, maxval=NOISE)
    x = r * jnp.cos(theta) + normal_x * noise
    y = r * jnp.sin(theta) + normal_y * noise
    return x, y


class Spiral(VectorizedTask):
    """
    Classify a point as belonging to one of two intertwined spirals.
    Spiral 1  ->  0
    Spiral 2  ->  1
    """

    def __init__(self, max_steps: int = 100, test: bool = False):
        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([2])
        self.act_shape = tuple([1])

        def reset_fn(key):
            key, k1, k2, k3 = jax.random.split(key, 4)
            x, y = spiral(k1, k2)
            target = jax.random.randint(k3, shape=(), minval=0, maxval=2)
            x = jnp.where(target == 0, x, -x)  # second spiral: flip the coordinates
            y = jnp.where(target == 0, y, -y)
            return State(
                obs=jnp.array([x, y]),
                target=target,
                steps=jnp.zeros((), dtype=jnp.int32),
                key=key,
            )

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            key, k1, k2, k3 = jax.random.split(state.key, 4)
            reward = jnp.squeeze(
                (
                    (jax.nn.sigmoid(action.astype(jnp.float32)) > 0.5).astype(jnp.int32)
                    == state.target
                ).astype(jnp.float32)
            )
            x, y = spiral(k1, k2)
            current_target = jax.random.randint(k3, shape=(), minval=0, maxval=2)
            x = jnp.where(current_target == 0, x, -x)
            y = jnp.where(current_target == 0, y, -y)
            steps = state.steps + 1
            return (
                State(
                    obs=jnp.array([x, y]), target=current_target, steps=steps, key=key
                ),
                reward,
                steps >= self.max_steps,
            )

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)

    @staticmethod
    def render(state, action) -> Image.Image:
        img_size = 900
        scale = (img_size - 50) // 2
        img = Image.new("RGB", (img_size, img_size), color="white")
        draw = ImageDraw.Draw(img)

        def spiral_polygon(flip: bool = False):
            thetas = np.linspace(0, 2 * np.pi * N_TURNS, 200)
            inner_points = []
            outer_points = []
            color = "blue"

            for theta in thetas:
                r = theta / (2 * np.pi * N_TURNS)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                if flip:
                    x = -x
                    y = -y
                    color = "orange"

                dr_dtheta = 1 / (2 * np.pi * N_TURNS)
                dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
                dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)
                tangent_length = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
                dx_dtheta /= tangent_length
                dy_dtheta /= tangent_length
                normal_x = -dy_dtheta
                normal_y = dx_dtheta

                inner_x = x - normal_x * NOISE
                inner_y = y - normal_y * NOISE
                outer_x = x + normal_x * NOISE
                outer_y = y + normal_y * NOISE

                # (0,0) in PIL is the top left
                inner_x = int((img_size // 2) + inner_x * scale)
                inner_y = int((img_size // 2) + -inner_y * scale)
                outer_x = int((img_size // 2) + outer_x * scale)
                outer_y = int((img_size // 2) + -outer_y * scale)
                inner_points.append((inner_x, inner_y))
                outer_points.append((outer_x, outer_y))

            # Create polygon by joining inner and outer points; Need to reverse one set to form a closed shape
            spiral_polygon = inner_points + outer_points[::-1]

            draw.polygon(spiral_polygon, fill=color, outline=None)

        spiral_polygon()
        spiral_polygon(flip=True)

        # Points
        radius = 5
        for i in range(len(state.obs)):
            obs = state.obs[i]
            is_spiral_2 = state.target[i] > 0.5
            pred_spiral_2 = jax.nn.sigmoid(action[i, 0]) > 0.5
            is_correct = is_spiral_2 == pred_spiral_2
            x = int((img_size // 2) + obs[0] * scale)
            y = int((img_size // 2) - obs[1] * scale)
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill="green" if is_correct else "red",
            )

        draw.text((10, 10), "Green: correct", fill="black")
        draw.text((10, 30), "Red: wrong", fill="black")
        return img
