# A wrapper of MuJoCoPlayground environments into EvoJAX's VectorizedTask template.

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask
from mujoco_playground import registry
from mujoco_playground._src.mjx_env import State as MujocoState


MUJOCO_ENVS = [
    "AcrobotSwingup",
    "AcrobotSwingupSparse",
    "AlohaHandOver",
    "AlohaSinglePegInsertion",
    "BallInCup",
    "BarkourJoystick",
    "CartpoleBalance",
    "CartpoleBalanceSparse",
    "CartpoleSwingup",
    "CartpoleSwingupSparse",
    "CheetahRun",
    "FingerSpin",
    "FingerTurnEasy",
    "FingerTurnHard",
    "FishSwim",
    "H1InplaceGaitTracking",
    "H1JoystickGaitTracking",
    "HopperHop",
    "HopperStand",
    "HumanoidRun",
    "HumanoidStand",
    "HumanoidWalk",
    "Op3Joystick",
    "PandaOpenCabinet",
    "PandaPickCube",
    "PandaPickCubeCartesian",
    "PandaPickCubeOrientation",
    "PandaRobotiqPushCube",
    "PendulumSwingup",
    "PointMass",
    "ReacherEasy",
    "ReacherHard",
    "SpotGetup",
    "SpotJoystickGaitTracking",
    "SwimmerSwimmer6",
    "WalkerRun",
    "WalkerStand",
    "WalkerWalk",
]


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    state: MujocoState
    steps: jnp.int32
    key: jnp.ndarray


class MujocoTask(VectorizedTask):
    def __init__(
        self,
        env_name: str,
        key: jnp.ndarray,
        max_steps: int = 1000,
        test: bool = False,
        sim_dt: float | None = None,
        ctrl_dt: float | None = None,
    ):
        self.max_steps = max_steps
        self.test = test

        config_overrides = {}
        if sim_dt is not None:
            config_overrides["sim_dt"] = sim_dt
        if ctrl_dt is not None:
            config_overrides["ctrl_dt"] = ctrl_dt
        self._base_env = registry.load(
            env_name, config_overrides=config_overrides or None
        )
        state = self._base_env.reset(key)
        self.obs_shape = (state.obs.shape[0],)
        self.act_shape = (self._base_env.action_size,)
        is_actuator_ctrl_range_limited = jnp.asarray(
            self._base_env.mjx_model.actuator_ctrllimited, dtype=bool
        )
        actuator_min = jnp.asarray(self._base_env.mjx_model.actuator_ctrlrange[:, 0])
        actuator_max = jnp.asarray(self._base_env.mjx_model.actuator_ctrlrange[:, 1])

        def reset_fn(key):
            key, k1 = jax.random.split(key)
            state = self._base_env.reset(k1)
            return State(
                obs=state.obs,
                state=state,
                steps=jnp.zeros((), dtype=jnp.int32),
                key=key,
            )

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state, action):
            action = jnp.where(
                is_actuator_ctrl_range_limited,
                jnp.clip(action, actuator_min, actuator_max),
                action,
            )
            current_state = self._base_env.step(state.state, action)
            steps = state.steps + 1
            is_done = jnp.logical_or(current_state.done > 0.5, steps >= max_steps)
            reward = current_state.reward
            steps = jnp.where(is_done, jnp.zeros((), jnp.int32), steps)
            return (
                State(
                    obs=current_state.obs,
                    state=current_state,
                    steps=steps,
                    key=state.key,
                ),
                reward,
                is_done,
            )

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key):
        return self._reset_fn(key)

    def step(self, state, action):
        return self._step_fn(state, action)
