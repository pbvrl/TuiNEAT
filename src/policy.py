import logging
from typing import Tuple
import jax
import jax.numpy as jnp
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState


ACTIVATIONS = [
    lambda x: jax.nn.identity(x),
    lambda x: jnp.tanh(x),
    lambda x: jax.nn.mish(x),
    lambda x: jnp.sin(x),
    lambda x: jnp.abs(x),
]


class NeatPolicyFeedforward(PolicyNetwork):
    def __init__(
        self, n_inputs: int, n_outputs: int, max_nds: int, logger: logging.Logger
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_nds = max_nds
        self._logger = logger

        @jax.jit
        def forward_fn(prms, obs):
            """
            Args:
                prms     see data definition
                obs      ( n_inputs ) float
            Returns:
                logits   ( n_outputs ) float
            """
            W = prms[:, : self.max_nds]
            node_info = prms[:, self.max_nds :]
            aggs = node_info[:, 0]
            bias = node_info[:, 1]
            acts = node_info[:, 2]
            is_indegree_positive = node_info[:, 3]
            node_indices = jnp.arange(self.n_inputs, self.max_nds)
            nds_values = jnp.zeros(self.max_nds).at[: self.n_inputs].set(obs)

            def process_node(carry, i):
                nds_values = carry
                incoming = W[i] * nds_values
                agg_val = jax.lax.cond(
                    aggs[i] == 2.0,
                    lambda inc: inc[jnp.argmax(jnp.abs(inc))] + bias[i],
                    lambda inc: jnp.sum(inc) + bias[i],
                    incoming,
                )
                act_val = jax.lax.switch(
                    acts[i].astype(jnp.int32), ACTIVATIONS, agg_val
                )
                act_val = jnp.where(is_indegree_positive[i] > 0, act_val, 0.0)
                nds_values = nds_values.at[i].set(act_val)
                return nds_values, None

            nds_values, _ = jax.lax.scan(process_node, nds_values, node_indices)
            logits = nds_values[self.max_nds - self.n_outputs :]
            return logits

        self._forward_fn = jax.vmap(forward_fn)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, _: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        logits = self._forward_fn(params, t_states.obs)
        return logits, _


class NeatPolicyCTRNN(PolicyNetwork):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        max_nds: int,
        ctrnn_integration_steps: int,
        logger: logging.Logger,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_nds = max_nds
        self.ctrnn_integration_steps = ctrnn_integration_steps
        self._logger = logger

        @jax.jit
        def forward_fn(prms, obs, prev_nds_values):
            """
            Args:
                prms           see data definition
                obs            ( n_inputs ) float
                prev_nds_values   ( max_nds ) float
            Returns:
                logits         ( n_outputs ) float
                prev_nds_values   ( max_nds ) float
            """
            W = prms[:, : self.max_nds]
            node_info = prms[:, self.max_nds :]
            aggs = node_info[:, 0]
            bias = node_info[:, 1]
            acts = node_info[:, 2]
            is_indegree_positive = node_info[:, 3]
            alpha = node_info[:, 4]
            node_indices = jnp.arange(self.n_inputs, self.max_nds)

            nds_values = prev_nds_values.at[: self.n_inputs].set(obs)

            def integration_step(prev_values, _):
                new_values = jnp.zeros(self.max_nds).at[: self.n_inputs].set(obs)

                def process_node(new_values, i):
                    # incoming = W[i] * new_values  # asynchronous Gauss-seidel
                    incoming = W[i] * prev_values  # synchronous forward-Euler
                    agg_val = jax.lax.cond(
                        aggs[i] == 2.0,
                        lambda inc: inc[jnp.argmax(jnp.abs(inc))] + bias[i],
                        lambda inc: jnp.sum(inc) + bias[i],
                        incoming,
                    )
                    act_val = jax.lax.switch(
                        acts[i].astype(jnp.int32), ACTIVATIONS, agg_val
                    )
                    act_val = jnp.where(is_indegree_positive[i] > 0, act_val, 0.0)
                    leaked = (1.0 - alpha[i]) * prev_values[i] + alpha[i] * act_val
                    new_values = new_values.at[i].set(leaked)
                    return new_values, None

                new_values, _ = jax.lax.scan(process_node, new_values, node_indices)
                return new_values, None

            nds_values, _ = jax.lax.scan(
                integration_step, nds_values, None, length=self.ctrnn_integration_steps
            )
            logits = nds_values[self.max_nds - self.n_outputs :]
            return logits, nds_values

        self._forward_fn = jax.vmap(forward_fn)

    def reset(self, t_states: TaskState) -> PolicyState:
        batch_size = t_states.obs.shape[0]
        prev_nds_values = jnp.zeros((batch_size, self.max_nds))
        return PolicyState(keys=prev_nds_values)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        prev_nds_values = p_states.keys
        logits, nds_values = self._forward_fn(params, t_states.obs, prev_nds_values)
        return logits, PolicyState(keys=nds_values)
