import os
from typing import TYPE_CHECKING
from logging import Logger
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from evojax.algo.base import NEAlgorithm

from .hyperparams import Hyperparams
from .data import (
    NeatAlgoData,
    prepare_params,
    prepare_params_feedforward,
)
from .topological_sorting import compute_dags_toposorts
from .algo.evaluation import (
    scale_positive_fitnesses,
    offset_fitnesses,
)
from .algo.crossover import apply_crossover_rate, perform_crossover
from .algo.mutation import (
    add_conns,
    add_conns_without_creating_cycles,
    add_nodes,
    change_weights,
    change_activations,
    change_aggregations,
    change_bias,
    change_ctrnn_alphas,
    disable_conns,
    reenable_conns,
    erase_conns,
    erase_nodes,
)
from .algo.speciation import (
    update_stagnation_tally,
    increase_stagnation_limit,
    cull_stagnant_species,
    calculate_adj_fitnesses,
    calculate_quotas,
    ruleout_species_laggards,
    rank_select_parents,
    get_species_frontrunners,
    get_reps,
    restore_species_frontrunners,
    erase_vacant_species,
    elect_reps,
    adjust_thresholds,
    speciate,
)

if TYPE_CHECKING:
    from .extra_modules.training_visualizer.client import TrainingVisualizerClient


CHECKPOINT_INTERVAL = 100


@jax.jit
def neat(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    data = scale_positive_fitnesses(data, hparams)
    data = offset_fitnesses(data, hparams)

    data = update_stagnation_tally(data)
    data = increase_stagnation_limit(data, hparams)
    data = cull_stagnant_species(data, hparams)
    data = calculate_adj_fitnesses(data)
    data = ruleout_species_laggards(data, hparams)
    data = calculate_quotas(data, hparams)
    data = rank_select_parents(data, hparams)
    (top_conns, top_conns_w, top_nodes, top_nodes_b, top_nodes_a) = (
        get_species_frontrunners(data, hparams)
    )
    reps_conns, reps_conns_w, reps_nodes, reps_nodes_b, reps_nodes_a = get_reps(data)

    data = apply_crossover_rate(data, hparams)
    data = perform_crossover(data, hparams)

    if hparams.feedforward:
        data = add_conns_without_creating_cycles(data, hparams)
    else:
        data = add_conns(data, hparams)
    data = add_nodes(data, hparams)
    data = reenable_conns(data, hparams)
    data = change_weights(data, hparams)
    data = change_aggregations(data, hparams)
    data = change_activations(data, hparams)
    data = change_bias(data, hparams)
    if not hparams.feedforward:
        data = change_ctrnn_alphas(data, hparams)
    data = disable_conns(data, hparams)
    data = erase_conns(data, hparams)
    data = erase_nodes(data, hparams)

    data = restore_species_frontrunners(
        data, hparams, top_conns, top_conns_w, top_nodes, top_nodes_b, top_nodes_a
    )
    data, dists = speciate(
        data, hparams, reps_conns, reps_conns_w, reps_nodes, reps_nodes_b, reps_nodes_a
    )
    data = erase_vacant_species(data)
    if hparams.dynamic_thresholds:
        data = adjust_thresholds(data, hparams, dists)
    data = elect_reps(data, hparams, dists)
    return data


@jax.jit
def ask(data: NeatAlgoData, hparams: Hyperparams):
    if hparams.feedforward:
        data = compute_dags_toposorts(data)
        params = prepare_params_feedforward(data)
    else:
        params = prepare_params(data)
    params = jnp.concatenate([params.weights, params.nodes], axis=-1)
    return data, params


class NeatAlgorithm(NEAlgorithm):
    def __init__(
        self,
        hparams: Hyperparams,
        logger: Logger,
        log_dir: str,
        neat_checkpoint_path: str | None = None,
        viz_client: "TrainingVisualizerClient | None" = None,
    ):
        self.logger = logger
        self.data = NeatAlgoData.create(hparams)
        self.hparams = hparams
        self.viz_client = viz_client
        self.checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        self.log_dir = log_dir
        if neat_checkpoint_path:
            self.logger.info(f"Starting from NEAT checkpoint {neat_checkpoint_path}")
            self.data = self.get_checkpoint(neat_checkpoint_path)
        self.pop_size = hparams.pop_size  # FULFILLS EvoJAX

    def ask(self):
        self.data, params = ask(self.data, self.hparams)
        self.cached_params = params
        return params

    def tell(self, fitness) -> None:
        self.data = self.data.replace(fitness=self.data.fitness.at[:, 0].set(fitness))
        if (self.data.gen % CHECKPOINT_INTERVAL == 0) and (self.data.gen > 0):
            self.save_checkpoint()
        if self.viz_client:
            self.hparams = self.viz_client.sync(self.data)
        max_fitness = jnp.max(fitness)
        best_current_prms = self.cached_params[jnp.argmax(fitness)]
        self.data = jax.lax.cond(
            max_fitness > self.data.best_f,
            lambda d: d.replace(best_f=max_fitness, best_prms=best_current_prms),
            lambda d: d,
            self.data,
        )
        self.data = neat(self.data, self.hparams)
        self.data = self.data.replace(gen=self.data.gen + 1)

    @property
    def best_params(self):  # FULFILLS EvoJAX
        return self.data.best_prms

    def get_checkpoint(self, path: str) -> NeatAlgoData:
        path = os.path.abspath(path)
        restored = self.checkpointer.restore(
            path,
            args=ocp.args.Composite(
                data=ocp.args.StandardRestore(
                    jax.tree.map(ocp.utils.to_shape_dtype_struct, self.data)
                ),
                hparams=ocp.args.StandardRestore(
                    jax.tree.map(ocp.utils.to_shape_dtype_struct, self.hparams)
                ),
            ),
        )
        assert self.hparams.pop_size == restored.hparams.pop_size
        assert self.hparams.max_cs == restored.hparams.max_cs
        assert self.hparams.max_nds == restored.hparams.max_nds
        assert self.hparams.max_species == restored.hparams.max_species
        assert self.hparams.n_inputs == restored.hparams.n_inputs
        assert self.hparams.n_outputs == restored.hparams.n_outputs
        return restored.data

    def save_checkpoint(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpointer.save(
            os.path.abspath(os.path.join(self.log_dir, f"gen_{self.data.gen}.pkl")),
            args=ocp.args.Composite(
                data=ocp.args.StandardSave(self.data),
                hparams=ocp.args.StandardSave(self.hparams),
            ),
            force=True,
        )
