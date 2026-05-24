import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    steps: jnp.int32
    key: jnp.ndarray


class XOR(VectorizedTask):
    """
    [0, 0]  ->  0
    [0, 1]  ->  1
    [1, 0]  ->  1
    [1, 1]  ->  0
    """

    def __init__(self, max_steps: int = 4, test: bool = False):
        self.max_steps = max_steps
        self.test = test
        self.obs_shape = tuple([2])
        self.act_shape = tuple([1])
        self.possible_obs = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

        def reset_fn(key):
            key, k1 = jax.random.split(key)
            obs_idx = jax.random.randint(k1, (), 0, 4)
            next_obs = self.possible_obs[obs_idx]
            return State(obs=next_obs, steps=jnp.zeros((), dtype=int), key=key)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            action = jnp.where(jax.nn.sigmoid(action) > 0.5, 1.0, 0.0)
            target = jnp.where(jnp.sum(state.obs) == 1.0, 1.0, 0.0)
            reward = jnp.squeeze((action == target).astype(jnp.float32))
            key, k1 = jax.random.split(state.key)
            obs_idx = jax.random.randint(k1, (), 0, 4)
            current_obs = self.possible_obs[obs_idx]
            steps = state.steps + 1
            is_done = steps >= self.max_steps
            return State(obs=current_obs, steps=steps, key=key), reward, is_done

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)
