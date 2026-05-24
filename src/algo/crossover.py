import jax
import jax.numpy as jnp

from ..hyperparams import Hyperparams
from ..data import NeatAlgoData


def apply_crossover_rate(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """For each species,
    replace ((1 - crossover_rate) * quota) offspring with their top individuals.

    Args:
        data.parents
        data.rankings
        data.species
        hparams.pop_size
        hparams.crossover_rate
    Returns:
        data.parents
    """
    pop_size = hparams.pop_size
    crossover_rate = hparams.crossover_rate

    quotas = data.species[:, 2]
    non_crossover_count = jnp.floor(
        (1.0 - crossover_rate) * quotas.astype(jnp.float32)
    ).astype(jnp.int32)
    non_crossover_count = jnp.minimum(non_crossover_count, data.species[:, 3])
    parents = jnp.where(
        (jnp.arange(pop_size)[None, :] < non_crossover_count[:, None])[..., None],
        jnp.stack([data.rankings, data.rankings], axis=-1),
        data.parents,
    )
    return data.replace(parents=parents)


def perform_crossover(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Produce the next generation from the selected parents using crossover.

    Args:
        data.parents
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
        data.key
        data.species
        hparams.pop_size
        hparams.max_cs
        hparams.max_nds
        hparams.enabled_recessiveness_prob
        hparams.avg_wghts_prob
    Returns:
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
        data.key
        data.species
    """
    pop_size = hparams.pop_size
    max_cs = hparams.max_cs
    max_nds = hparams.max_nds

    counts = jnp.sum(data.parents[..., 0] != -1, axis=1)
    cumsum = jnp.cumsum(counts)
    inds_sp_idx = jnp.searchsorted(cumsum, jnp.arange(pop_size), side="right")
    offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), cumsum[:-1]])
    idx_within_sp = jnp.arange(pop_size) - offsets[inds_sp_idx]
    parents = data.parents[inds_sp_idx, idx_within_sp]

    key, k1, k2, k3 = jax.random.split(data.key, 4)
    conns0 = data.conns[parents[:, 0]]
    conns1 = data.conns[parents[:, 1]]
    conns_w0 = data.conns_w[parents[:, 0]]
    conns_w1 = data.conns_w[parents[:, 1]]
    is_conns_from1 = jax.random.choice(k1, jnp.array([True, False]), conns0.shape[:2])
    is_averaging = jax.random.bernoulli(k2, hparams.avg_wghts_prob, conns0.shape[:2])
    is_disabling = jax.random.bernoulli(
        k3, hparams.enabled_recessiveness_prob, conns0.shape[:2]
    )

    def crossover_prts_cs(
        cs0_ids, cs1_ids, cs0, cs1, cs_w0, cs_w1, is_cs_from1, is_avrging, is_dsbling
    ):
        is_empty0 = cs0_ids == 0
        is_empty1 = cs1_ids == 0
        ssort_ids0 = jnp.where(~is_empty0, cs0_ids, jnp.iinfo(jnp.int32).max)
        ssort_ids1 = jnp.where(~is_empty1, cs1_ids, jnp.iinfo(jnp.int32).max)
        ssort_idxs0 = jnp.argsort(ssort_ids0)
        ssort_idxs1 = jnp.argsort(ssort_ids1)
        ssort_ids0 = ssort_ids0[ssort_idxs0]
        ssort_ids1 = ssort_ids1[ssort_idxs1]
        cs0 = cs0[ssort_idxs0]
        cs1 = cs1[ssort_idxs1]
        cs_w0 = cs_w0[ssort_idxs0]
        cs_w1 = cs_w1[ssort_idxs1]
        cs0_ids = cs0_ids[ssort_idxs0]
        cs1_ids = cs1_ids[ssort_idxs1]
        is_cs_from1 = is_cs_from1[ssort_idxs0]
        is_avrging = is_avrging[ssort_idxs0]
        is_dsbling = is_dsbling[ssort_idxs0]
        is_empty0 = is_empty0[ssort_idxs0]
        idxs1 = jnp.clip(jnp.searchsorted(ssort_ids1, ssort_ids0), 0, max_cs - 1)
        is_mtchng = (cs1_ids[idxs1] == cs0_ids) & ~is_empty0
        cs1 = cs1[idxs1]
        cs_w1 = cs_w1[idxs1]
        is_cs_from1 = is_cs_from1 & is_mtchng
        is_avrging = is_avrging & is_mtchng
        is_dsbling = is_dsbling & is_mtchng
        cs = jnp.where(is_cs_from1[:, None], cs1, cs0)
        cs_w = jnp.where(is_cs_from1, cs_w1, cs_w0)
        cs_w = jnp.where(is_avrging, (cs_w0 + cs_w1) / 2, cs_w)
        is_dsbling = is_dsbling & ((cs0[:, 3] == 0) | (cs1[:, 3] == 0))
        cs = cs.at[:, 3].set(jnp.where(is_dsbling, 0, cs[:, 3]))
        return cs, cs_w, is_cs_from1

    conns, conns_w, is_conns_from1 = jax.vmap(crossover_prts_cs)(
        conns0[:, :, 2],
        conns1[:, :, 2],
        conns0,
        conns1,
        conns_w0,
        conns_w1,
        is_conns_from1,
        is_averaging,
        is_disabling,
    )

    nodes0 = data.nodes[parents[:, 0], :, :]
    nodes1 = data.nodes[parents[:, 1], :, :]
    nodes_b0 = data.nodes_b[parents[:, 0]]
    nodes_b1 = data.nodes_b[parents[:, 1]]
    nodes_a0 = data.nodes_a[parents[:, 0]]
    nodes_a1 = data.nodes_a[parents[:, 1]]

    def derive_child_nds(cs, is_cs_from1, nds0, nds1, nds_b0, nds_b1, nds_a0, nds_a1):
        in_ids, out_ids, is_empty = cs[:, 0], cs[:, 1], cs[:, 2] == 0
        # matching innovation numbers therefore matching node ids
        nds_ids0 = nds0[:, 0]
        in_nds_idxs = jnp.argmax(in_ids[:, None] == nds_ids0[None, :], axis=1)
        out_nds_idxs = jnp.argmax(out_ids[:, None] == nds_ids0[None, :], axis=1)
        is_cs_from0 = (~is_cs_from1 & ~is_empty).astype(jnp.int32)
        p1_counts = jnp.zeros(max_nds, dtype=jnp.int32)
        p1_counts = p1_counts.at[in_nds_idxs].add(is_cs_from1)
        p1_counts = p1_counts.at[out_nds_idxs].add(is_cs_from1)
        p0_counts = jnp.zeros(max_nds, dtype=jnp.int32)
        p0_counts = p0_counts.at[in_nds_idxs].add(is_cs_from0)
        p0_counts = p0_counts.at[out_nds_idxs].add(is_cs_from0)
        is_nds_from1 = p1_counts > p0_counts
        nds_ids1 = nds1[:, 0]
        ssort_ids = jnp.where(nds_ids1 != 0, nds_ids1, jnp.iinfo(jnp.int32).max)
        ssort_idxs = jnp.argsort(ssort_ids)
        ssort_ids = ssort_ids[ssort_idxs]
        nds_idxs1 = ssort_idxs[
            jnp.clip(jnp.searchsorted(ssort_ids, nds_ids0), 0, max_nds - 1)
        ]
        nodes = jnp.where(is_nds_from1[:, None], nds1[nds_idxs1], nds0)
        nodes_b = jnp.where(is_nds_from1, nds_b1[nds_idxs1], nds_b0)
        nodes_a = jnp.where(is_nds_from1, nds_a1[nds_idxs1], nds_a0)
        return nodes, nodes_b, nodes_a

    nodes, nodes_b, nodes_a = jax.vmap(derive_child_nds)(
        conns, is_conns_from1, nodes0, nodes1, nodes_b0, nodes_b1, nodes_a0, nodes_a1
    )

    # Reset species
    is_empty_sp = data.species[:, 0] == -1
    member_counts = jnp.where(is_empty_sp, data.species[:, 3], 0)
    species = data.species.at[:, 3].set(member_counts).at[:, 5].set(-1)
    ind = jnp.column_stack(
        [
            jnp.full(data.ind.shape[0], -1, dtype=data.ind.dtype),
            jnp.zeros(data.ind.shape[0], dtype=data.ind.dtype),
        ]
    )

    return data.replace(
        conns=conns,
        conns_w=conns_w,
        nodes=nodes,
        nodes_b=nodes_b,
        nodes_a=nodes_a,
        key=key,
        species=species,
        ind=ind,
    )
