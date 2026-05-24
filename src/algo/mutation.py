import math
import jax
import jax.numpy as jnp

from ..hyperparams import Hyperparams
from ..data import NeatAlgoData


def add_conns(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """For each network in the population, add one connection gene to its genome, over a probability threshold.

    Args:
        data.nodes
        data.conns
        data.conns_w
        data.key
        data.next_inn
        hparams.pop_size
        hparams.add_one_c_prob
        hparams.replace_wght_stdev
    Returns:
        data.conns
        data.conns_w
        data.key
        data.next_inn
    """
    pop_size = hparams.pop_size
    max_nds = hparams.max_nds

    key, k1, *mut_keys = jax.random.split(data.key, pop_size + 2)
    mut_keys = jnp.array(mut_keys)
    is_mutating = jax.random.bernoulli(k1, hparams.add_one_c_prob, shape=(pop_size,))
    below_max_cs = jnp.any(data.conns[:, :, 0] == 0, axis=1)
    is_mutating = is_mutating & below_max_cs

    nds_idxs = jnp.arange(max_nds)
    in_idxs, out_idxs = jnp.dstack(jnp.meshgrid(nds_idxs, nds_idxs)).reshape(-1, 2).T

    def find_candidate_cs(args):
        """For one network, find all valid non-existing connections."""
        cs, nds = args

        in_ids, out_ids = nds[in_idxs, 0], nds[out_idxs, 0]
        in_types, out_types = nds[in_idxs, 1], nds[out_idxs, 1]
        is_candidate = ((in_types == 1) | (in_types == 3)) & (
            (out_types == 2) | (out_types == 3)
        )

        c_in_idxs = jnp.argmax(cs[:, 0, None] == nds[None, :, 0], axis=1)
        c_out_idxs = jnp.argmax(cs[:, 1, None] == nds[None, :, 0], axis=1)
        exists = jnp.zeros((max_nds, max_nds), dtype=jnp.bool_)
        exists = exists.at[c_in_idxs, c_out_idxs].set(cs[:, 0] != 0)
        is_existing = exists[in_idxs, out_idxs]

        is_candidate = is_candidate & ~is_existing

        candidate_cs = jnp.where(
            is_candidate[:, None],
            jnp.stack([in_ids, out_ids], axis=1),
            jnp.zeros((max_nds**2, 2), dtype=jnp.int32),
        )
        return candidate_cs, jnp.any(is_candidate)

    candidate_conns, is_candidate_available = jax.vmap(
        jax.lax.cond, in_axes=(0, 0, None, 0, None)
    )(
        is_mutating,
        (data.conns, data.nodes),
        find_candidate_cs,
        None,
        lambda _: (jnp.zeros((max_nds**2, 2), dtype=jnp.int32), False),
    )
    is_mutating = is_mutating & is_candidate_available

    def add_c(args):
        """For one network, add a new connection.

        Leave the innovation numbers unset."""
        key, remaining_cs, cs, cs_w = args
        k1, k2 = jax.random.split(key)
        weight = (
            jax.random.uniform(k2, minval=-1, maxval=1) * hparams.replace_wght_stdev
        )
        c = jax.random.choice(
            k1,
            remaining_cs,
            p=~(jnp.any(remaining_cs == 0, axis=1, keepdims=True)[:, 0]),
        )
        c_idx = jnp.argmax(jnp.all(cs == 0, axis=1))
        cs = cs.at[c_idx].set(jnp.array([c[0], c[1], 0, 1], dtype=jnp.int32))
        cs_w = cs_w.at[c_idx].set(weight)
        return cs, cs_w, c_idx

    conns, conns_w, conn_idxs = jax.vmap(jax.lax.cond, in_axes=(0, 0, None, 0, None))(
        is_mutating,
        (mut_keys, candidate_conns, data.conns, data.conns_w),
        add_c,
        (data.conns, data.conns_w),
        lambda x: (x[0], x[1], 0),
    )

    def assign_innovs(carry, input):
        """Assign innovation numbers,
        with reuse for connections that have originated more than once within the current generation."""
        next_inn, seen_conns, i = carry
        is_mutating, c = input
        is_seen = jnp.where(
            is_mutating, jnp.any(jnp.all(c[:2] == seen_conns[:, :2], axis=1)), False
        )
        seen_conns = jnp.where(
            (is_mutating & ~is_seen),
            seen_conns.at[i].set(jnp.array([c[0], c[1], next_inn], dtype=jnp.int32)),
            seen_conns,
        )
        inn = jnp.where(
            is_mutating,
            jnp.where(
                is_seen,
                seen_conns[jnp.argmax(jnp.all(c[:2] == seen_conns[:, :2], axis=1)), 2],
                next_inn,
            ),
            c[2],
        )
        return (
            jnp.where((is_mutating & ~is_seen), next_inn + 1, next_inn),
            seen_conns,
            i + 1,
        ), inn

    idxs = jnp.arange(pop_size)
    (next_inn, _, _), innovs = jax.lax.scan(
        assign_innovs,
        (data.next_inn, jnp.zeros((pop_size, 3), dtype=jnp.int32), 0),
        (is_mutating, conns[idxs, conn_idxs, :]),
    )

    conns = conns.at[jnp.arange(len(conns)), conn_idxs, 2].set(innovs)

    return data.replace(conns=conns, conns_w=conns_w, next_inn=next_inn, key=key)


def add_conns_without_creating_cycles(
    data: NeatAlgoData, hparams: Hyperparams
) -> NeatAlgoData:
    """For each network in the population, add one connection gene to its genome, over a probability threshold.
    Without creating cycles.

    Args:
        data.nodes
        data.conns
        data.conns_w
        data.key
        data.next_inn
        hparams.pop_size
        hparams.add_one_c_prob
    Returns:
        data.conns
        data.conns_w
        data.key
        data.next_inn
    """
    pop_size = hparams.pop_size
    max_nds = hparams.max_nds

    key, k1, *mut_keys = jax.random.split(data.key, pop_size + 2)
    mut_keys = jnp.array(mut_keys)
    is_mutating = jax.random.bernoulli(k1, hparams.add_one_c_prob, shape=(pop_size,))
    below_max_cs = jnp.any(data.conns[:, :, 0] == 0, axis=1)
    is_mutating = is_mutating & below_max_cs

    nds_idxs = jnp.arange(max_nds)
    in_idxs, out_idxs = jnp.dstack(jnp.meshgrid(nds_idxs, nds_idxs)).reshape(-1, 2).T
    is_self_conn = in_idxs == out_idxs

    def find_candidate_cs(args):
        """For one network, find all valid non-existing connections."""
        cs, nds = args

        in_ids, out_ids = nds[in_idxs, 0], nds[out_idxs, 0]
        in_types, out_types = nds[in_idxs, 1], nds[out_idxs, 1]
        is_candidate = ((in_types == 1) | (in_types == 3)) & (
            (out_types == 2) | (out_types == 3)
        )

        c_in_idxs = jnp.argmax(cs[:, 0, None] == nds[None, :, 0], axis=1)
        c_out_idxs = jnp.argmax(cs[:, 1, None] == nds[None, :, 0], axis=1)
        exists = jnp.zeros((max_nds, max_nds), dtype=jnp.bool_)
        exists = exists.at[c_in_idxs, c_out_idxs].set(cs[:, 0] != 0)
        is_existing = exists[in_idxs, out_idxs]

        def square_reachability(reach, _):
            return reach | ((reach @ reach) > 0).astype(jnp.int32), None

        adj = jnp.zeros((max_nds, max_nds), dtype=jnp.int32)
        adj = adj.at[c_in_idxs, c_out_idxs].set((cs[:, 0] != 0).astype(jnp.int32))
        longest_path = max_nds - 1
        n_iters = math.ceil(math.log2(longest_path))
        reachability, _ = jax.lax.scan(square_reachability, adj, None, length=n_iters)
        is_cycle_creating = (
            (reachability[out_idxs, in_idxs] == 1) & (in_types != 1) & (out_types != 2)
        )

        is_candidate = is_candidate & ~is_existing & ~is_cycle_creating & ~is_self_conn

        candidate_cs = jnp.where(
            is_candidate[:, None],
            jnp.stack([in_ids, out_ids], axis=1),
            jnp.zeros((max_nds**2, 2), dtype=jnp.int32),
        )
        return candidate_cs, jnp.any(is_candidate)

    candidate_conns, is_candidate_available = jax.vmap(
        jax.lax.cond, in_axes=(0, 0, None, 0, None)
    )(
        is_mutating,
        (data.conns, data.nodes),
        find_candidate_cs,
        None,
        lambda _: (jnp.zeros((max_nds**2, 2), dtype=jnp.int32), False),
    )
    is_mutating = is_mutating & is_candidate_available

    def add_c(args):
        """For one network, add a new connection.

        Leave the innovation numbers unset."""
        key, remaining_cs, cs, cs_w = args
        k1, k2 = jax.random.split(key)
        weight = (
            jax.random.uniform(k2, minval=-1, maxval=1) * hparams.replace_wght_stdev
        )
        c = jax.random.choice(
            k1,
            remaining_cs,
            p=~(jnp.any(remaining_cs == 0, axis=1, keepdims=True)[:, 0]),
        )
        c_idx = jnp.argmax(jnp.all(cs == 0, axis=1))
        cs = cs.at[c_idx].set(jnp.array([c[0], c[1], 0, 1], dtype=jnp.int32))
        cs_w = cs_w.at[c_idx].set(weight)
        return cs, cs_w, c_idx

    conns, conns_w, conn_idxs = jax.vmap(jax.lax.cond, in_axes=(0, 0, None, 0, None))(
        is_mutating,
        (mut_keys, candidate_conns, data.conns, data.conns_w),
        add_c,
        (data.conns, data.conns_w),
        lambda x: (x[0], x[1], 0),
    )

    def assign_innovs(carry, input):
        """Assign innovation numbers,
        with reuse for connections that have originated more than once within the current generation."""
        next_inn, seen_conns, i = carry
        is_mutating, c = input
        is_seen = jnp.where(
            is_mutating, jnp.any(jnp.all(c[:2] == seen_conns[:, :2], axis=1)), False
        )
        seen_conns = jnp.where(
            (is_mutating & ~is_seen),
            seen_conns.at[i].set(jnp.array([c[0], c[1], next_inn], dtype=jnp.int32)),
            seen_conns,
        )
        inn = jnp.where(
            is_mutating,
            jnp.where(
                is_seen,
                seen_conns[jnp.argmax(jnp.all(c[:2] == seen_conns[:, :2], axis=1)), 2],
                next_inn,
            ),
            c[2],
        )
        return (
            jnp.where((is_mutating & ~is_seen), next_inn + 1, next_inn),
            seen_conns,
            i + 1,
        ), inn

    idxs = jnp.arange(pop_size)
    (next_inn, _, _), innovs = jax.lax.scan(
        assign_innovs,
        (data.next_inn, jnp.zeros((pop_size, 3), dtype=jnp.int32), 0),
        (is_mutating, conns[idxs, conn_idxs, :]),
    )

    conns = conns.at[jnp.arange(len(conns)), conn_idxs, 2].set(innovs)

    return data.replace(conns=conns, conns_w=conns_w, next_inn=next_inn, key=key)


def add_nodes(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """For each network in the population, split an enabled connection into two and add one node,
    over a probability threshold.

    If the number of hidden nodes is small, bias nodes towards appearing from older connections.
    This mitigates the chaining effect, where from one connection continually stem more:

    0 --- 0     to     0 - 0 - 0 - 0 - 0 - 0 - 0
    0 -- /             0 -------------------- /

    Args:
        data.key
        data.conns
        data.nodes
        data.nodes_b
        data.nodes_a
        data.next_nd_id
        data.next_inn
        hparams.pop_size
        hparams.max_cs
        hparams.add_one_nd_prob
    Returns:
        data.key
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
        data.next_nd_id
        data.next_inn
    """

    pop_size = hparams.pop_size
    max_cs = hparams.max_cs

    initial_alpha = (hparams.ctrnn_min_alpha + hparams.ctrnn_max_alpha) * 0.5
    if hparams.feedforward:
        initial_alpha = 0.0

    key, k1, *mut_keys = jax.random.split(data.key, pop_size + 2)
    mut_keys = jnp.array(mut_keys)
    is_mutating = jax.random.bernoulli(k1, hparams.add_one_nd_prob, shape=(pop_size,))
    is_below_cs_cap = jnp.sum(jnp.all(data.conns[:, :] == 0, axis=2), axis=1) >= 2
    is_any_c_enabled = jnp.any(data.conns[:, :, 3] == 1, axis=1)
    is_below_nds_cap = jnp.any(data.nodes[:, :, 0] == 0, axis=1)
    is_mutating = is_mutating & is_below_cs_cap & is_below_nds_cap & is_any_c_enabled

    small_threshold = 7
    is_small = jnp.sum(data.nodes[:, :, 1] == 3, axis=1) < small_threshold

    def add_nd(args):
        """For one network, split a connection into two and add a node.

        Leave the node ids and innovation numbers unset."""
        cs, cs_w, nds, nds_b, nds_a, key, is_small = args
        is_enabled = (cs[:, 3] == 1).astype(jnp.float32)
        prob = jnp.where(
            is_small, (1 / jnp.log2(3 + cs[:, 2])) * is_enabled, is_enabled
        )
        split_c_idx = jax.random.choice(key, jnp.arange(max_cs), p=prob)
        split_in_id = cs[split_c_idx, 0]
        split_out_id = cs[split_c_idx, 1]
        split_w = cs_w[split_c_idx]
        if hparams.erase_split_c:
            cs = cs.at[split_c_idx].set(jnp.zeros(4, dtype=jnp.int32))
            cs_w = cs_w.at[split_c_idx].set(0.0)
        else:
            cs = cs.at[split_c_idx, 3].set(0)
        c1_idx = jnp.argmax(jnp.all(cs == 0, axis=1))
        cs = cs.at[c1_idx].set(jnp.array([split_in_id, 0, 0, 1], dtype=jnp.int32))
        c2_idx = jnp.argmax(jnp.all(cs == 0, axis=1))
        cs = cs.at[c2_idx].set(jnp.array([0, split_out_id, 0, 1], dtype=jnp.int32))
        cs_w = cs_w.at[jnp.array([c1_idx, c2_idx])].set(jnp.array([1.0, split_w]))
        nd_idx = jnp.argmax(jnp.all(nds == 0, axis=1))
        nds = nds.at[nd_idx].set(jnp.array([0, 3, 1, 1], dtype=jnp.int32))
        nds_b = nds_b.at[nd_idx].set(0.0)
        nds_a = nds_a.at[nd_idx].set(initial_alpha)
        return (
            cs,
            cs_w,
            split_in_id,
            split_out_id,
            c1_idx,
            c2_idx,
            nds,
            nds_b,
            nds_a,
            nd_idx,
        )

    (
        conns,
        conns_w,
        split_in_ids,
        split_out_ids,
        conn1_idxs,
        conn2_idxs,
        nodes,
        nodes_b,
        nodes_a,
        node_idxs,
    ) = jax.vmap(jax.lax.cond, in_axes=(0, 0, None, 0, None))(
        is_mutating,
        (
            data.conns,
            data.conns_w,
            data.nodes,
            data.nodes_b,
            data.nodes_a,
            mut_keys,
            is_small,
        ),
        add_nd,
        (data.conns, data.conns_w, data.nodes, data.nodes_b, data.nodes_a),
        lambda x: (x[0], x[1], 0, 0, 0, 0, x[2], x[3], x[4], 0),
    )

    def assign_node_ids_and_innovs(carry, input):
        """Assign node IDs and innovation numbers,
        with reuse for splits that have originated more than once within the current generation."""
        next_nd_id, next_inn, seen_splits, i = carry
        is_mutating, split_conn = input
        is_seen = jnp.where(
            is_mutating,
            jnp.any(jnp.all(split_conn == seen_splits[:, :2], axis=1)),
            False,
        )
        seen_splits = jnp.where(
            (is_mutating & ~is_seen),
            seen_splits.at[i].set(
                jnp.concatenate(
                    [
                        split_conn,
                        jnp.array(
                            [next_nd_id, next_inn, next_inn + 1], dtype=jnp.int32
                        ),
                    ]
                )
            ),
            seen_splits,
        )
        seen_idx = jnp.argmax(jnp.all(split_conn == seen_splits[:, :2], axis=1))
        node_id = jnp.where(
            is_mutating, jnp.where(is_seen, seen_splits[seen_idx, 2], next_nd_id), 0
        )
        inn1 = jnp.where(
            is_mutating,
            jnp.where(is_seen, seen_splits[seen_idx, 3], next_inn),
            split_conn[0],
        )
        inn2 = jnp.where(
            is_mutating,
            jnp.where(is_seen, seen_splits[seen_idx, 4], next_inn + 1),
            split_conn[1],
        )
        return (
            jnp.where((is_mutating & ~is_seen), next_nd_id + 1, next_nd_id),
            jnp.where((is_mutating & ~is_seen), next_inn + 2, next_inn),
            seen_splits,
            i + 1,
        ), (node_id, inn1, inn2)

    idxs = jnp.arange(pop_size)
    split_conns = jnp.column_stack((split_in_ids, split_out_ids))
    (next_nd_id, next_inn, _, _), (node_ids, innovs1, innovs2) = jax.lax.scan(
        assign_node_ids_and_innovs,
        (
            data.next_nd_id,
            data.next_inn,
            jnp.zeros((pop_size, 5), dtype=jnp.int32),
            0,
        ),
        (is_mutating, split_conns),
    )

    is_mutating = is_mutating[:, None, None]
    nodes = jnp.where(is_mutating, nodes.at[idxs, node_idxs, 0].set(node_ids), nodes)
    conns = jnp.where(
        is_mutating,
        conns.at[idxs, conn1_idxs, 1]
        .set(node_ids)
        .at[idxs, conn2_idxs, 0]
        .set(node_ids),
        conns,
    )
    conns = jnp.where(
        is_mutating,
        conns.at[idxs, conn1_idxs, 2].set(innovs1).at[idxs, conn2_idxs, 2].set(innovs2),
        conns,
    )

    return data.replace(
        conns=conns,
        conns_w=conns_w,
        nodes=nodes,
        nodes_b=nodes_b,
        nodes_a=nodes_a,
        next_inn=next_inn,
        next_nd_id=next_nd_id,
        key=key,
    )


def change_weights(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Modify the weights of each existing connection of each network,
    over a probability threshold.

    Args:
        data.key
        data.conns
        data.conns_w
        hparams.perturb_wght_stdev
        hparams.perturb_wght_prob
        hparams.replace_wght_prob
    Returns:
        data.key
        data.conns_w
    """
    key, k1, k2, k3 = jax.random.split(data.key, 4)
    new_weights = jax.random.normal(k1, shape=data.conns_w.shape)
    is_replacing = jax.random.bernoulli(
        k2, hparams.replace_wght_prob, shape=data.conns_w.shape
    )
    is_perturbing = jax.random.bernoulli(
        k3, hparams.perturb_wght_prob, shape=data.conns_w.shape
    )
    is_enabled = data.conns[:, :, 3] == 1
    is_replacing = is_replacing & is_enabled
    is_perturbing = is_perturbing & is_enabled
    mutated_weights = jnp.where(
        is_perturbing,
        data.conns_w + new_weights * hparams.perturb_wght_stdev,
        data.conns_w,
    )
    mutated_weights = jnp.where(
        is_replacing, new_weights * hparams.replace_wght_stdev, mutated_weights
    )
    mutated_weights = jnp.clip(mutated_weights, hparams.min_wght, hparams.max_wght)
    return data.replace(conns_w=mutated_weights, key=key)


def change_aggregations(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Change the aggregation function of each non-input node of each network,
    over a probability threshold.

    Args:
        data.key
        data.nodes
        hparams.change_agg_prob
    Returns:
        data.key
        data.nodes
    """
    key, k1, k2, k3 = jax.random.split(data.key, 4)
    aggs = data.nodes[:, :, 2]
    is_valid = data.nodes[:, :, 1] > 1
    is_mutating = jax.random.bernoulli(k1, hparams.change_agg_prob, shape=aggs.shape)
    is_mutating = is_mutating & is_valid
    min_agg = 1
    max_agg = 2
    new_aggs = jax.random.randint(k2, aggs.shape, min_agg, max_agg + 1)
    offset = jax.random.randint(k3, aggs.shape, 1, max_agg - min_agg + 1)
    offset_new_aggs = (
        ((new_aggs - min_agg) + offset) % (max_agg - min_agg + 1)
    ) + min_agg
    new_aggs = jnp.where(new_aggs == aggs, offset_new_aggs, new_aggs)
    nodes = data.nodes.at[:, :, 2].set(jnp.where(is_mutating, new_aggs, aggs))
    return data.replace(nodes=nodes, key=key)


def change_activations(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Change the activation function of each non-input node of each network,
    over a probability threshold.
    Fix outputs to the default based on hyperparams.

    Args:
        data.key
        data.nodes
        hparams.change_act_prob
        hparams.default_output_activation
    Returns:
        data.key
        data.nodes
    """
    fix_outputs = hparams.default_output_activation

    key, k1, k2, k3 = jax.random.split(data.key, 4)
    acts = data.nodes[:, :, 3]
    is_output = data.nodes[:, :, 1] == 2
    is_hidden = data.nodes[:, :, 1] == 3
    is_mutating = jax.random.bernoulli(k1, hparams.change_act_prob, shape=acts.shape)
    is_mutating = is_mutating & (is_hidden | (is_output & jnp.logical_not(fix_outputs)))
    min_act = 0
    max_act = 4
    new_acts = jax.random.randint(k2, acts.shape, min_act, max_act + 1)
    offset = jax.random.randint(k3, acts.shape, 1, max_act + 1)
    offset_new_acts = (
        ((new_acts - min_act) + offset) % (max_act - min_act + 1)
    ) + min_act
    new_acts = jnp.where(new_acts == acts, offset_new_acts, new_acts)

    default_act = 1
    acts = jnp.where(is_mutating, new_acts, acts)
    acts = jnp.where(is_output & fix_outputs, default_act, acts)
    nodes = data.nodes.at[:, :, 3].set(acts)
    return data.replace(nodes=nodes, key=key)


def change_bias(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Change the bias of each non-input node of each network,
    over a probability threshold.

    Args:
        data.key
        data.nodes
        data.nodes_b
        hparams.perturb_bias_stdev
        hparams.perturb_bias_prob
        hparams.replace_bias_prob
        hparams.replace_bias_stdev
    Returns:
        data.key
        data.nodes_b
    """
    key, k1, k2, k3 = jax.random.split(data.key, 4)
    new_biases = jax.random.normal(k1, shape=data.nodes_b.shape)
    is_replacing = jax.random.bernoulli(
        k2, hparams.replace_bias_prob, shape=data.nodes_b.shape
    )
    is_perturbing = jax.random.bernoulli(
        k3, hparams.perturb_bias_prob, shape=data.nodes_b.shape
    )
    is_valid = data.nodes[:, :, 1] > 1
    is_replacing = is_replacing & is_valid
    is_perturbing = is_perturbing & is_valid
    mutated_biases = jnp.where(
        is_perturbing,
        data.nodes_b + new_biases * hparams.perturb_bias_stdev,
        data.nodes_b,
    )
    mutated_biases = jnp.where(
        is_replacing, new_biases * hparams.replace_bias_stdev, mutated_biases
    )
    mutated_biases = jnp.clip(mutated_biases, hparams.min_bias, hparams.max_bias)
    return data.replace(nodes_b=mutated_biases, key=key)


def change_ctrnn_alphas(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Change the CTRNN leak coefficient alpha of each non-input node of each network,
    over a probability threshold.

    Perturbation adds gaussian noise scaled by perturb_alpha_stdev.
    Replacement maps the same gaussian draw into [ctrnn_min_alpha, ctrnn_max_alpha]
    by centering on the midpoint and scaling by the half-range.

    Args:
        data.key
        data.nodes
        data.nodes_a
        hparams.perturb_alpha_stdev
        hparams.perturb_alpha_prob
        hparams.replace_alpha_prob
        hparams.ctrnn_min_alpha
        hparams.ctrnn_max_alpha
    Returns:
        data.key
        data.nodes_a
    """
    key, k1, k2, k3 = jax.random.split(data.key, 4)
    new_alphas = jax.random.normal(k1, shape=data.nodes_a.shape)
    is_replacing = jax.random.bernoulli(
        k2, hparams.replace_alpha_prob, shape=data.nodes_a.shape
    )
    is_perturbing = jax.random.bernoulli(
        k3, hparams.perturb_alpha_prob, shape=data.nodes_a.shape
    )
    is_valid = data.nodes[:, :, 1] > 1
    is_replacing = is_replacing & is_valid
    is_perturbing = is_perturbing & is_valid
    midpoint = 0.5 * (hparams.ctrnn_min_alpha + hparams.ctrnn_max_alpha)
    half_range = 0.5 * (hparams.ctrnn_max_alpha - hparams.ctrnn_min_alpha)
    mutated_alphas = jnp.where(
        is_perturbing,
        data.nodes_a + new_alphas * hparams.perturb_alpha_stdev,
        data.nodes_a,
    )
    mutated_alphas = jnp.where(
        is_replacing, midpoint + new_alphas * half_range, mutated_alphas
    )
    mutated_alphas = jnp.clip(
        mutated_alphas, hparams.ctrnn_min_alpha, hparams.ctrnn_max_alpha
    )
    return data.replace(nodes_a=mutated_alphas, key=key)


def disable_conns(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Disable each existing enabled connection of each network,
    over a probability threshold.

    Avoid disabling connections whose in nodes have no other enabled outgoing connections.

    Args:
        data.key
        data.conns
        hparams.pop_size
        hparams.max_cs
        hparams.disable_c_prob
    Returns:
        data.key
        data.conns
    """
    pop_size = hparams.pop_size
    max_cs = hparams.max_cs
    max_nds = hparams.max_nds

    key, k1 = jax.random.split(data.key)

    def count_outgoing_cs(nds, cs):
        is_enabled = (cs[:, 3] == 1).astype(jnp.int32)
        in_nds_order_in_nds = jnp.argmax(cs[:, 0, None] == nds[None, :, 0], axis=1)
        nds_outgoing_counts = jax.ops.segment_sum(
            is_enabled, in_nds_order_in_nds, max_nds
        )
        cs_outgoing_counts_from_own_in_nd = jnp.take(
            nds_outgoing_counts, in_nds_order_in_nds
        )
        return cs_outgoing_counts_from_own_in_nd

    outgoing_counts_from_in_node = jax.vmap(count_outgoing_cs)(data.nodes, data.conns)

    is_disabling = jax.random.bernoulli(k1, hparams.disable_c_prob, (pop_size, max_cs))
    is_enabled = data.conns[:, :, 3] == 1
    is_disabling = is_disabling & is_enabled & (outgoing_counts_from_in_node > 1)
    conns = data.conns.at[:, :, 3].set(jnp.where(is_disabling, 0, data.conns[:, :, 3]))
    return data.replace(conns=conns, key=key)


def reenable_conns(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """For each network, reenable one disabled connection over a probability threshold.

    Args:
        data.key
        data.conns
        hparams.pop_size
        hparams.max_cs
        hparams.reenable_one_c_prob
    Returns:
        data.key
        data.conns
    """
    pop_size = hparams.pop_size
    max_cs = hparams.max_cs

    key, k1, *mut_keys = jax.random.split(data.key, pop_size + 2)
    mut_keys = jnp.array(mut_keys)
    is_mutating = jax.random.bernoulli(
        k1, hparams.reenable_one_c_prob, shape=(pop_size,)
    )
    is_disabled = (data.conns[:, :, 0] != 0) & (data.conns[:, :, 3] == 0)
    is_any_c_disabled = jnp.any(is_disabled, axis=1)
    is_mutating = is_mutating & is_any_c_disabled

    def reenable_c(args):
        key, cs, is_disabled = args
        c_idx = jax.random.choice(
            key, jnp.arange(max_cs), p=is_disabled.astype(jnp.float32)
        )
        cs = cs.at[c_idx, 3].set(1)
        return cs

    conns = jax.vmap(jax.lax.cond, in_axes=(0, 0, None, 0, None))(
        is_mutating,
        (mut_keys, data.conns, is_disabled),
        reenable_c,
        data.conns,
        lambda x: x,
    )
    return data.replace(conns=conns, key=key)


def erase_conns(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """For each network in the population, erase one connection gene from its genome,
    over a probability threshold.

    Args:
        data.key
        data.conns
        data.conns_w
        hparams.pop_size
        hparams.max_cs
        hparams.erase_c_prob
    Returns:
        data.key
        data.conns
        data.conns_w
    """
    pop_size = hparams.pop_size
    max_cs = hparams.max_cs

    key, k1, *mut_keys = jax.random.split(data.key, pop_size + 2)
    mut_keys = jnp.array(mut_keys)
    is_mutating = jax.random.bernoulli(k1, hparams.erase_c_prob, shape=(pop_size,))
    is_any_c_existing = jnp.any(data.conns[:, :, 0] != 0, axis=1)
    is_mutating = is_mutating & is_any_c_existing

    def erase_c(args):
        key, cs, cs_w = args
        is_existing = (cs[:, 0] != 0).astype(jnp.float32)
        c_idx = jax.random.choice(key, jnp.arange(max_cs), p=is_existing)
        cs = cs.at[c_idx].set(jnp.array([0, 0, 0, 0], dtype=jnp.int32))
        cs_w = cs_w.at[c_idx].set(0.0)
        return cs, cs_w

    conns, conns_w = jax.vmap(jax.lax.cond, in_axes=(0, 0, None, 0, None))(
        is_mutating,
        (mut_keys, data.conns, data.conns_w),
        erase_c,
        (data.conns, data.conns_w),
        lambda x: (x[0], x[1]),
    )

    return data.replace(conns=conns, conns_w=conns_w, key=key)


def erase_nodes(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """For each network in the population, erase one hidden node that has no connections,
    over a probability threshold.

    Args:
        data.key
        data.nodes
        data.nodes_b
        data.nodes_a
        data.conns
        hparams.pop_size
        hparams.max_nds
        hparams.erase_disjointed_nd_prob
    Returns:
        data.key
        data.nodes
        data.nodes_b
        data.nodes_a
    """
    pop_size = hparams.pop_size
    max_nds = hparams.max_nds

    key, k1, *mut_keys = jax.random.split(data.key, pop_size + 2)
    mut_keys = jnp.array(mut_keys)
    is_mutating = jax.random.bernoulli(
        k1, hparams.erase_disjointed_nd_prob, shape=(pop_size,)
    )

    def find_erasable_nodes(nds, cs):
        """For one network, find all hidden nodes with no connections."""
        is_existing_nd = nds[:, 0] != 0
        is_hidden = nds[:, 1] == 3
        nds_ids = nds[:, 0]
        existing_cs = cs[:, 0] != 0
        is_conn_in = jnp.any(
            (nds_ids[:, None] == cs[None, :, 0]) & existing_cs[None, :], axis=1
        )
        is_conn_out = jnp.any(
            (nds_ids[:, None] == cs[None, :, 1]) & existing_cs[None, :], axis=1
        )
        is_connected = is_conn_in | is_conn_out
        is_erasable = is_existing_nd & is_hidden & ~is_connected
        return is_erasable

    is_candidate = jax.vmap(find_erasable_nodes)(data.nodes, data.conns)
    is_candidate_available = jnp.any(is_candidate, axis=1)
    is_mutating = is_mutating & is_candidate_available

    def erase_nd(args):
        key, nds, nds_b, nds_a, is_erasable = args
        nd_idx = jax.random.choice(
            key, jnp.arange(max_nds), p=is_erasable.astype(jnp.float32)
        )
        nds = nds.at[nd_idx].set(jnp.array([0, 0, 0, 0], dtype=jnp.int32))
        nds_b = nds_b.at[nd_idx].set(0.0)
        nds_a = nds_a.at[nd_idx].set(0.0)
        return nds, nds_b, nds_a

    nodes, nodes_b, nodes_a = jax.vmap(jax.lax.cond, in_axes=(0, 0, None, 0, None))(
        is_mutating,
        (mut_keys, data.nodes, data.nodes_b, data.nodes_a, is_candidate),
        erase_nd,
        (data.nodes, data.nodes_b, data.nodes_a, is_candidate),
        lambda x: (x[0], x[1], x[2]),
    )

    return data.replace(nodes=nodes, nodes_b=nodes_b, nodes_a=nodes_a, key=key)
