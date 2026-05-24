import jax
import jax.numpy as jnp
import jax.random as random
from flax import struct

from .hyperparams import Hyperparams


@struct.dataclass
class NeatAlgoData(struct.PyTreeNode):  # https://github.com/google/flax/issues/4312
    """State kept from one generation to another, and some other variables defined here for convenience."""

    nodes: jnp.ndarray  # ( pop_size, max_nds, 4 ) int
    """
    [:, :, 0] = Id { 0: empty, >0: id }
    [:, :, 1] = Type { 0: empty, 1: input, 2: output, 3: hidden }
    [:, :, 2] = Aggregation function { -1: input node, 0: empty, 1: sum, 2: maxabs }
    [:, :, 3] = Activation function { -1: input node, 0: empty / identity, 1: tanh, 2: mish, 3: sin, 4: abs }
    """
    nodes_b: jnp.ndarray  # ( pop_size, max_nds ) float
    """
    [:] = Bias { 0.0: empty / input / value 0.0, (-inf, inf): value }
    """
    nodes_a: jnp.ndarray  # ( pop_size, max_nds ) float
    """
    CTRNN networks dt/tau. Only for hparams.feedforward = False
    [:, :] = Alpha { 0.0: feedforward only / value,  [hparams.ctrnn_min_alpha, hparams.ctrnn_max_alpha]: value }
    """
    conns: jnp.ndarray  # ( pop_size, max_cs, 4 ) int
    """
    [:, :, 0] = In node id { 0: empty, >0: id }
    [:, :, 1] = Out node id { 0: empty, >0: id }
    [:, :, 2] = Innovation number { 0: empty, >0: number }
    [:, :, 3] = Enabled { 0: empty or false, 1: true }
    """
    conns_w: jnp.ndarray  # ( pop_size, max_cs ) float
    """
    [:, :] = Connection weight { 0.0: empty / weight 0.0, (-inf, inf): weight }
    """
    toposorted_idxs: jnp.ndarray  # ( pop_size, max_nds ) int
    """
    Indexes of the nodes in one topological sort of each network. Only for hparams.feedforward = True
    [j, i] =  { -1: algo allows recurrence, [0, max_nds): index of the i'th sorted node in nodes[j] }
    """
    fitness: jnp.ndarray  # ( pop_size, 4 ) float
    """
    [:, 0] = Fitness { -inf: first generation placeholder, >(-inf): fitness }
    [:, 1] = Fitness with positive values scaled { -inf: first generation placeholder, >(-inf): fitness }
    [:, 2] = Fitness scaled and offset away from negatives { 0.0: first generation placeholder / fitness, >0: fitness }
    [:, 3] = Adjusted fitness { -inf: species stagnated / first generation placeholder, >=0.0: fitness }
    """
    ind: jnp.ndarray  # ( pop_size, 2 ) int
    """
    [:, 0] = Species the individual belongs to { -1: not yet assigned / species stagnated, [0, max_species): index }
    [:, 1] = Potential for parenthood { -1: stagnated / did not survive culling, 0: eligible }
    """
    parents: jnp.ndarray  # ( max_species, pop_size, 2 ) int
    """
    [:, :, 0] = Most fit parent { -1: not yet assigned / beyond quota, [0, pop_size-1]: index }
    [:, :, 1] = Less fit parent { -1: not yet assigned / beyond quota, [0, pop_size-1]: index }
    """
    species: jnp.ndarray  # ( max_species, 6 ) int
    """
    [:, 0] = Generation of its record fitness { -1: empty / erased by stagnation, >=0: gen }
    [:, 1] = Species id { -1: empty / erased by stagnation, >=1: id }
    [:, 2] = Quotas { -1: empty / erased by stagnation, 0: post-crossover and pre-speciation / value, [1, pop_size]: value }
    [:, 3] = Member count { -1: empty / erased by stagnation, 0: post-crossover and pre-speciation / count, [1, pop_size]: count }
    [:, 4] = Generation it originated { -1: empty / erased by stagnation, >=0: gen }
    [:, 5] = Representative { -1: empty / erased by stagnation / post-crossover and pre-speciation, [0, pop_size): index }
    """
    thresholds: jnp.ndarray  # ( max_species, ) float
    """
     Per-species compatibility threshold. Only for hparams.dynamic_thresholds = True
    [:] = { -1.0: empty, >0: threshold }
    """
    species_best_f: jnp.ndarray  # ( max_species, ) float
    """
    [:] = Record fitness among individuals from each species, across generations { -inf: empty or not yet evaluated, >(-inf): fitness }
    """
    rankings: jnp.ndarray  # ( max_species, pop_size ) int
    """
    [:, i] = Indexes of each species members in the population index, sorted by fitness { -1: beyond member count, [0, pop_size): index of i'th ranked member }
    """
    modified_stagnation_age: jnp.ndarray  # int
    """Stagnation age increased as generations go on."""
    next_inn: jnp.ndarray  # int
    next_nd_id: jnp.ndarray  # int
    next_sp_id: jnp.ndarray  # int
    best_prms: jnp.ndarray  # ( max_nds, max_nds + 5 ) float
    """Policy params of the individual that scored the record fitness across generations.
    FULFILLS EvoJAX."""
    best_f: jnp.ndarray  # float
    "Fitness of best_prms."
    gen: jnp.ndarray  # int
    key: jnp.ndarray  # PRNG key

    @classmethod
    def create(cls, hparams: Hyperparams):
        """Generate the initial algorithm data."""
        pop_size = hparams.pop_size
        n_inputs = hparams.n_inputs
        n_outputs = hparams.n_outputs
        max_cs = hparams.max_cs
        seed = hparams.seed
        initial_cs_percentage = hparams.percentage_of_possible_initial_connections
        replace_wght_stdev = hparams.replace_wght_stdev
        max_nds = hparams.max_nds
        is_feedforward = hparams.feedforward
        max_species = hparams.max_species
        base_compat_threshold = hparams.base_compat_threshold
        base_stagnation_age = hparams.base_stagnation_age

        initial_alpha = 0.5 * (hparams.ctrnn_min_alpha + hparams.ctrnn_max_alpha)
        if is_feedforward:
            initial_alpha = 0.0

        def init_nodes():
            n_init_nodes = n_inputs + n_outputs
            nodes = jnp.zeros((pop_size, max_nds, 4), dtype=jnp.int32)
            nodes = nodes.at[:, :n_init_nodes, 0].set(jnp.arange(n_init_nodes) + 1)
            nodes = nodes.at[:, :n_inputs, 1].set(1)
            nodes = nodes.at[:, n_inputs:n_init_nodes, 1].set(2)
            nodes = nodes.at[:, :n_inputs, 2].set(-1)
            nodes = nodes.at[:, n_inputs:n_init_nodes, 2].set(1)
            nodes = nodes.at[:, :n_inputs, 3].set(-1)
            nodes = nodes.at[:, n_inputs:n_init_nodes, 3].set(1)
            nodes_b = jnp.zeros((pop_size, max_nds), dtype=jnp.float32)
            nodes_a = jnp.zeros((pop_size, max_nds), dtype=jnp.float32)
            nodes_a = nodes_a.at[:, :n_init_nodes].set(initial_alpha)
            next_nd_id = n_init_nodes + 1
            toposorted_idxs = jnp.where(
                is_feedforward,
                jnp.tile(jnp.arange(max_nds, dtype=jnp.int32), (pop_size, 1)),
                -jnp.ones((pop_size, max_nds), dtype=jnp.int32),
            )
            return nodes, nodes_b, nodes_a, next_nd_id, toposorted_idxs

        def init_conns(key):
            conns = jnp.zeros((pop_size, max_cs, 4), dtype=jnp.int32)
            conns_w = jnp.zeros((pop_size, max_cs), dtype=jnp.float32)
            n_possible_cs = n_inputs * n_outputs
            n_init_cs = max(1, int(n_possible_cs * initial_cs_percentage))
            in_ids = jnp.repeat(jnp.arange(1, n_inputs + 1), n_outputs)
            out_ids = jnp.tile(
                jnp.arange(n_inputs + 1, n_inputs + n_outputs + 1), n_inputs
            )
            innovation_nums = jnp.arange(1, n_possible_cs + 1)
            key, k1, k2 = random.split(key, 3)
            scores = random.uniform(k1, (pop_size, n_possible_cs))
            selected_idxs = jnp.sort(jnp.argsort(scores, axis=1)[:, :n_init_cs], axis=1)
            selected_in = jnp.take(in_ids, selected_idxs)
            selected_out = jnp.take(out_ids, selected_idxs)
            selected_inn = jnp.take(innovation_nums, selected_idxs)
            conns = conns.at[:, :n_init_cs, 0].set(selected_in)
            conns = conns.at[:, :n_init_cs, 1].set(selected_out)
            conns = conns.at[:, :n_init_cs, 2].set(selected_inn)
            conns = conns.at[:, :n_init_cs, 3].set(1)
            weights = random.normal(k2, (pop_size, n_init_cs)) * replace_wght_stdev
            conns_w = conns_w.at[:, :n_init_cs].set(weights)
            next_inn = n_possible_cs + 1
            return conns, conns_w, next_inn, key

        def init_species():
            ind = jnp.zeros((pop_size, 2), dtype=jnp.int32)
            species = -jnp.ones((max_species, 6), dtype=jnp.int32)
            species = species.at[0, :].set(jnp.array([0, 1, pop_size, pop_size, 0, 0]))
            species_best_f = jnp.full((max_species,), -jnp.inf, dtype=jnp.float32)
            thresholds = jnp.full((max_species,), -1.0, dtype=jnp.float32)
            thresholds = thresholds.at[0].set(base_compat_threshold)
            rankings = (
                jnp.full((max_species, pop_size), -1, dtype=jnp.int32)
                .at[0]
                .set(jnp.arange(pop_size, dtype=jnp.int32))
            )
            parents = -jnp.ones((max_species, pop_size, 2), dtype=jnp.int32)
            return (ind, species, thresholds, species_best_f, rankings, parents)

        key = random.PRNGKey(seed)
        nodes, nodes_b, nodes_a, next_nd_id, toposorted_idxs = init_nodes()
        conns, conns_w, next_inn, key = init_conns(key)
        (ind, species, thresholds, species_best_f, rankings, parents) = init_species()
        return cls(
            nodes,
            nodes_b,
            nodes_a,
            conns,
            conns_w,
            toposorted_idxs,
            jnp.full((pop_size, 4), -jnp.inf, dtype=jnp.float32).at[:, 2:].set(0.0),
            ind,
            parents,
            species,
            thresholds,
            species_best_f,
            rankings,
            jnp.int32(base_stagnation_age),
            jnp.int32(next_inn),
            jnp.int32(next_nd_id),
            jnp.int32(2),
            jnp.zeros((max_nds, max_nds + 5)),
            jnp.float32(-jnp.inf),
            jnp.int32(0),
            key,
        )


@struct.dataclass
class PolicyPrms:
    "Parameters that set the policy. forward_fn(prms, obs_i) = action_i"

    weights: jnp.ndarray  # ( max_nds, max_nds ) float
    """
    [out_idx, in_idx] = Weight { 0.0: disabled / weight, (-inf, inf): weight }
    With the nodes sorted to acomodate forward_fn:
    1. Topologically sorted; applies to the feedforward case.
    2. Sorted so that:
        inputs at [0, n_inputs)
        hidden / empty nodes in the middle (keeping topological sort, if applicable)
        outputs at [max_nds - n_outputs, max_nds)
    3. Within inputs/outputs, sorted by ids
    """
    nodes: jnp.ndarray  # ( max_nds, 5 ) float
    """
    [ :, 0] = Aggregation { 0.0: identity, 1.0: sum, 2.0: maxabs }
    [ :, 0] = Aggregation { 0.0: identity, 1.0: sum, 2.0: maxabs }
    [ :, 1] = Bias { (-inf, inf): value }
    [ :, 2] = Activation { 0.0: identity, 1.0: tanh, 2.0: mish, 3.0: sin, 4.0: abs }
    [ :, 3] = Has incoming enabled connections { 0.0: no, 1.0: yes }
    [ :, 4] = CTRNNs alpha coefficient { 0.0: value / hparams.feedforward=True, [hparams.ctrnn_min_alpha, hparams.ctrnn_max_alpha]: value }
    Same sorting as for weights
    """


@struct.dataclass
class Prms:
    prms: jnp.ndarray  # ( max_nds, max_nds + 5 ) float
    """
    Same but concatenated into a single array. FULFILLS EvoJAX.
    """


def sort_inputs_outputs(nds):
    type_key = jnp.where(nds[:, 1] == 1, 0, jnp.where(nds[:, 1] == 2, 2, 1))
    stride = jnp.maximum(jnp.max(nds[:, 0]), nds.shape[0]) + 1
    within_type_sort = jnp.where(type_key != 1, nds[:, 0], jnp.arange(nds.shape[0]))
    return jnp.argsort(type_key * stride + within_type_sort, stable=True)


def prepare_params(data: NeatAlgoData):
    def body(cs, cs_w, nds, nds_b, nds_a):
        sorted_idxs = sort_inputs_outputs(nds)
        nds = jnp.take(nds, sorted_idxs, axis=0)
        nds_b = jnp.take(nds_b, sorted_idxs, axis=0)
        nds_a = jnp.take(nds_a, sorted_idxs, axis=0)
        return prepare_prms(cs, cs_w, nds, nds_b, nds_a)

    return jax.vmap(body)(
        data.conns, data.conns_w, data.nodes, data.nodes_b, data.nodes_a
    )


def prepare_params_feedforward(data: NeatAlgoData):
    def body(cs, cs_w, nds, nds_b, nds_a, toposrtd_idxs):
        nds = jnp.take(nds, toposrtd_idxs, axis=0)
        nds_b = jnp.take(nds_b, toposrtd_idxs, axis=0)
        nds_a = jnp.take(nds_a, toposrtd_idxs, axis=0)
        sorted_idxs = sort_inputs_outputs(nds)
        nds = jnp.take(nds, sorted_idxs, axis=0)
        nds_b = jnp.take(nds_b, sorted_idxs, axis=0)
        nds_a = jnp.take(nds_a, sorted_idxs, axis=0)
        return prepare_prms(cs, cs_w, nds, nds_b, nds_a)

    return jax.vmap(body)(
        data.conns,
        data.conns_w,
        data.nodes,
        data.nodes_b,
        data.nodes_a,
        data.toposorted_idxs,
    )


def prepare_prms(cs, cs_w, sorted_nds, sorted_nds_b, sorted_nds_a) -> PolicyPrms:
    max_nds = sorted_nds.shape[0]
    ssort_idxs = jnp.argsort(sorted_nds[:, 0])
    ssort_ids = sorted_nds[:, 0][ssort_idxs]
    in_idxs = ssort_idxs[
        jnp.clip(jnp.searchsorted(ssort_ids, cs[:, 0]), 0, max_nds - 1)
    ]
    out_idxs = ssort_idxs[
        jnp.clip(jnp.searchsorted(ssort_ids, cs[:, 1]), 0, max_nds - 1)
    ]
    is_enabled = cs[:, 3] == 1
    enabled_weights = jnp.where(is_enabled, cs_w, 0.0)
    weights = jnp.zeros((max_nds, max_nds)).at[out_idxs, in_idxs].add(enabled_weights)

    aggs = jnp.where(sorted_nds[:, 2] == -1, 0, sorted_nds[:, 2]).astype(jnp.float32)
    bias = sorted_nds_b.astype(jnp.float32)
    acts = jnp.where(sorted_nds[:, 3] == -1, 0, sorted_nds[:, 3]).astype(jnp.float32)
    is_indegree_positive = (
        jnp.zeros(max_nds).at[out_idxs].add(is_enabled.astype(jnp.float32)) > 0
    ).astype(jnp.float32)
    alpha = sorted_nds_a.astype(jnp.float32)
    nodes = jnp.stack([aggs, bias, acts, is_indegree_positive, alpha], axis=1)
    return PolicyPrms(weights=weights, nodes=nodes)
