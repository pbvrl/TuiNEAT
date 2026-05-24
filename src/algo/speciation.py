import jax
import jax.numpy as jnp

from ..hyperparams import Hyperparams
from ..data import NeatAlgoData


def update_stagnation_tally(data: NeatAlgoData) -> NeatAlgoData:
    """Keep track of species stagnation.

    Args:
        data.ind
        data.fitness
        data.gen
        data.species
        data.species_best_f
        hparams.max_species (inferred)
    Returns:
        data.species
        data.species_best_f
    """

    def update_sp_trackrecord(idx, sp, sp_best_f):
        """For one species, update its record fitness and when it was last reached."""
        best_f_current_gen = jnp.max(
            jnp.where(data.ind[:, 0] == idx, data.fitness[:, 0], -jnp.inf)
        )
        is_improving = (best_f_current_gen > sp_best_f) & (sp[0] != -1)
        sp = sp.at[0].set(jnp.where(is_improving, data.gen, sp[0]))
        sp_best_f = jnp.where(is_improving, best_f_current_gen, sp_best_f)
        return sp, sp_best_f

    species, species_best_f = jax.vmap(update_sp_trackrecord)(
        jnp.arange(data.species.shape[0]), data.species, data.species_best_f
    )
    return data.replace(species=species, species_best_f=species_best_f)


def increase_stagnation_limit(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Increase the stagnation cutoff based on the current generation.

    Args:
        data.gen
        hparams.base_stagnation_age
        hparams.stagnation_age_powerlaw_growth
    Returns:
        data.modified_stagnation_age
    """
    stagnation_age = jnp.float32(hparams.base_stagnation_age) * jnp.power(
        (data.gen + 100) / 100, hparams.stagnation_age_powerlaw_growth
    )
    return data.replace(modified_stagnation_age=jnp.int32(stagnation_age))


def cull_stagnant_species(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Cull species that haven't surpassed their best fitness in n generations.
    If all species were to be culled, keep one.
    Exempt top species.

    Args:
        data.ind
        data.fitness
        data.gen
        data.species
        data.species_best_f
        data.modified_stagnation_age
        hparams.max_species
        hparams.n_stagnation_exempt_top_species
    Returns:
        data.species
        data.species_best_f
        data.ind
    """
    max_species = hparams.max_species
    n_exempt_species = hparams.n_stagnation_exempt_top_species

    cutoff_age = data.modified_stagnation_age
    records = jnp.where(data.species[:, 0] != -1, data.species_best_f, -jnp.inf)
    species_rank = jnp.argsort(records, descending=True)
    species_rank = (
        jnp.empty_like(species_rank).at[species_rank].set(jnp.arange(max_species))
    )
    is_exempt = species_rank < n_exempt_species

    def cull_stagnant_sp(sp_idx):
        is_exempt_sp = is_exempt[sp_idx]
        sp = data.species[sp_idx]
        sp_best_f = data.species_best_f[sp_idx]
        is_stagnant = jnp.where(
            sp[0] != -1, ((data.gen - sp[0]) >= cutoff_age) & ~is_exempt_sp, False
        )
        sp = jnp.where(
            is_stagnant, jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32), sp
        )
        sp_best_f = jnp.where(is_stagnant, -jnp.inf, sp_best_f)
        return sp, sp_best_f

    species, species_best_f = jax.vmap(cull_stagnant_sp)(jnp.arange(max_species))

    is_empty_sp = jnp.where(species[:, 0] == -1, True, False)
    is_from_empty_sp = jnp.where(
        data.ind[:, 0] >= 0, is_empty_sp[data.ind[:, 0]], False
    )
    ind = jnp.where(is_from_empty_sp[:, None], -1, data.ind)

    # Handle the case where all are erased; possible with n_exempt_species = 0
    all_culled = jnp.all(species[:, 0] == -1)
    top_sp_idx = jnp.argmax(data.species_best_f)
    species = jnp.where(
        all_culled, species.at[top_sp_idx].set(data.species[top_sp_idx]), species
    )
    species_best_f = jnp.where(
        all_culled,
        species_best_f.at[top_sp_idx].set(data.species_best_f[top_sp_idx]),
        species_best_f,
    )
    ind = jnp.where(
        all_culled, jnp.where(data.ind[:, 0][:, None] == top_sp_idx, data.ind, -1), ind
    )

    return data.replace(species=species, species_best_f=species_best_f, ind=ind)


def calculate_adj_fitnesses(data: NeatAlgoData) -> NeatAlgoData:
    """Calculate the adjusted fitness of each individual.

    Args:
        data.fitness
        data.species
        data.ind
    Returns:
        data.fitness
    """

    is_stagnant_species = data.ind[:, 0] == -1
    denom = jnp.where(~is_stagnant_species, data.species[data.ind[:, 0], 3], 1).astype(
        jnp.float32
    )
    adj_fitnesses = jnp.where(
        ~is_stagnant_species, data.fitness[:, 2] / denom, -jnp.inf
    )
    return data.replace(fitness=data.fitness.at[:, 3].set(adj_fitnesses))


def ruleout_species_laggards(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Exclude the lowest performers from each species from parenthood.

    Args:
        data.ind
        data.fitness
        hparams.pop_size
        hparams.max_species
        hparams.intraspecies_parenthood_eligibility
    Returns:
        data.ind
        data.rankings
    """
    pop_size = hparams.pop_size
    max_species = hparams.max_species
    surv_thresh = hparams.intraspecies_parenthood_eligibility

    def extract_sp_ranking(sp_idx):
        is_from_sp = data.ind[:, 0] == sp_idx
        fitnesses = jnp.where(is_from_sp, data.fitness[:, 0], -jnp.inf)
        ranking = jnp.argsort(fitnesses, descending=True)
        member_count = jnp.sum(is_from_sp).astype(jnp.int32)
        return jnp.where(jnp.arange(pop_size) < member_count, ranking, -1)

    rankings = jax.vmap(extract_sp_ranking)(jnp.arange(max_species))

    def ruleout_sp_laggards(ranking):
        member_count = jnp.sum(ranking >= 0).astype(jnp.int32)
        include_count = jnp.ceil(member_count.astype(jnp.float32) * surv_thresh).astype(
            jnp.int32
        )
        eligible_idxs = jnp.where(jnp.arange(pop_size) < include_count, ranking, -1)
        sp_parenthood_status = (
            -jnp.ones((pop_size + 1), dtype=jnp.int32).at[eligible_idxs].set(0)
        )[:-1]
        return sp_parenthood_status

    species_parenthood_status = jax.vmap(ruleout_sp_laggards)(rankings)
    parenthood_status = jnp.where(
        data.ind[:, 0] >= 0,
        species_parenthood_status[data.ind[:, 0], jnp.arange(pop_size)],
        -1,
    )
    return data.replace(ind=data.ind.at[:, 1].set(parenthood_status), rankings=rankings)


def calculate_quotas(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Calculate the amount of offspring for each species to produce.
    Consider only the eligible parents' adjusted fitness.

    Args:
        data.species
        data.ind
        data.fitness
        hparams.pop_size
        hparams.max_species
        hparams.species_fitness_powerscaling
    """
    pop_size = hparams.pop_size
    max_species = hparams.max_species
    species_fitness_powerscaling = hparams.species_fitness_powerscaling

    is_empty_species = data.species[:, 0] == -1

    def adj_f_sum(sp_idx):
        is_from_sp = data.ind[:, 0] == sp_idx
        is_eligible = is_from_sp & (data.ind[:, 1] == 0)
        eligible_count = jnp.sum(is_eligible)
        adj_f_sum = jnp.where(
            is_empty_species[sp_idx],
            0.0,
            jnp.sum(jnp.where(is_eligible, data.fitness[:, 2], 0.0)),
        )
        return adj_f_sum, eligible_count

    adj_f_sums, eligible_counts = jax.vmap(adj_f_sum)(jnp.arange(max_species))
    denom = jnp.maximum(eligible_counts, 1).astype(jnp.float32)
    adj_f_avgs = jnp.where(eligible_counts > 0, adj_f_sums / denom, 0.0)
    total_adj_f = jnp.sum(adj_f_avgs)
    f_normalized = jnp.where(
        total_adj_f > 0,
        adj_f_avgs / total_adj_f,
        jnp.where(
            ~is_empty_species, 1.0 / jnp.maximum(jnp.sum(~is_empty_species), 1), 0.0
        ),
    )
    f_powerscaled = f_normalized**species_fitness_powerscaling
    f_powerscaled = f_powerscaled / jnp.sum(f_powerscaled)
    quotas = jnp.floor(f_powerscaled * pop_size).astype(jnp.int32)

    rounding_error = pop_size - jnp.sum(quotas)
    top_species = jnp.argmax(jnp.where(~is_empty_species, adj_f_avgs, -jnp.inf))
    quotas = quotas.at[top_species].set(quotas[top_species] + rounding_error)
    quotas = jnp.where(is_empty_species, -1, quotas)
    species = data.species.at[:, 2].set(quotas)
    return data.replace(species=species)


def rank_select_parents(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """Rank select the next pairs of parents.

    Args:
        data.ind
        data.fitness
        data.species
        data.rankings
        data.key
        hparams.pop_size
        hparams.max_species
        hparams.interspecies_mating_ratio
    Returns:
        data.parents
        data.key
    """
    pop_size = hparams.pop_size
    max_species = hparams.max_species
    intersp_ratio = hparams.interspecies_mating_ratio

    key, *species_keys = jax.random.split(data.key, max_species + 1)
    species_keys = jnp.array(species_keys)
    is_surv = data.ind[:, 1] != -1
    interspecies_ranking = jnp.argsort(
        jnp.where(is_surv, data.fitness[:, 0], -jnp.inf), descending=True
    )

    def select_from_sp(sp_idx, sp_key):
        k1, k2, k3 = jax.random.split(sp_key, 3)

        def rank_select(k, ranking):
            is_valid = (ranking >= 0) & is_surv[ranking]
            surv_count = jnp.sum(is_valid).astype(jnp.int32)
            ranks = jnp.where(
                jnp.arange(pop_size) < surv_count, surv_count - jnp.arange(pop_size), 0
            )
            denom = jnp.sum(ranks)
            prob = jnp.where(denom > 0, ranks / denom, 1.0 / pop_size)
            return jax.random.choice(k, ranking, (pop_size,), True, prob)

        intrasp_parents0 = rank_select(k1, data.rankings[sp_idx])
        intrasp_parents1 = rank_select(k2, data.rankings[sp_idx])
        intersp_parents1 = rank_select(k3, interspecies_ranking)
        return intrasp_parents0, intrasp_parents1, intersp_parents1

    intra0, intra1, inter1 = jax.vmap(select_from_sp)(
        jnp.arange(max_species), species_keys
    )

    quotas = data.species[:, 2]
    intra_quotas = jnp.round(quotas.astype(jnp.float32) * (1.0 - intersp_ratio)).astype(
        jnp.int32
    )
    is_intra = jnp.arange(pop_size)[None, :] < intra_quotas[:, None]
    parents0 = intra0
    parents1 = jnp.where(is_intra, intra1, inter1)
    parents = jnp.stack([parents0, parents1], axis=-1)

    fitness_p0 = data.fitness[parents[..., 0], 0]
    fitness_p1 = data.fitness[parents[..., 1], 0]
    is_swapping = fitness_p0 < fitness_p1
    parents = jnp.where(is_swapping[..., None], parents[..., ::-1], parents)
    is_within_quota = jnp.arange(pop_size)[None, :] < quotas[:, None]
    parents = jnp.where(is_within_quota[..., None], parents, -1)
    return data.replace(parents=parents, key=key)


def get_species_frontrunners(data: NeatAlgoData, hparams: Hyperparams):
    """Get a snapshot of the highest fitness individuals from each species.

    Args:
        data.ind
        data.species
        data.rankings
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
        hparams.n_intraspecies_culling_exempt_members
    Returns:
        top_conns, top_conns_w, top_nodes, top_nodes_b, top_nodes_a
    """
    n_highperformers = hparams.intraspecies_unchanged_frontrunners
    top_idxs = data.rankings[:, :n_highperformers]
    is_empty_sp = (data.species[:, 0] == -1)[:, None]
    is_empty = (top_idxs < 0) | is_empty_sp
    top_conns = jnp.where(
        ~is_empty[..., None, None], data.conns[top_idxs], jnp.zeros_like(data.conns[0])
    )
    top_conns_w = jnp.where(
        ~is_empty[..., None], data.conns_w[top_idxs], jnp.zeros_like(data.conns_w[0])
    )
    top_nodes = jnp.where(
        ~is_empty[..., None, None], data.nodes[top_idxs], jnp.zeros_like(data.nodes[0])
    )
    top_nodes_b = jnp.where(
        ~is_empty[..., None], data.nodes_b[top_idxs], jnp.zeros_like(data.nodes_b[0])
    )
    top_nodes_a = jnp.where(
        ~is_empty[..., None], data.nodes_a[top_idxs], jnp.zeros_like(data.nodes_a[0])
    )
    return top_conns, top_conns_w, top_nodes, top_nodes_b, top_nodes_a


def get_reps(data: NeatAlgoData):
    """Get a snapshot of the current representatives.

    Args:
        data.species
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
    Returns:
        reps_conns, reps_conns_w, reps_nodes, reps_nodes_b, reps_nodes_a
    """
    rep_idxs = data.species[:, 5]
    is_empty_sp = data.species[:, 0] == -1
    rep_idxs = jnp.where(is_empty_sp, 0, rep_idxs)
    reps_conns = jnp.where(
        is_empty_sp[:, None, None], jnp.zeros_like(data.conns[0]), data.conns[rep_idxs]
    )
    reps_conns_w = jnp.where(
        is_empty_sp[:, None], jnp.zeros_like(data.conns_w[0]), data.conns_w[rep_idxs]
    )
    reps_nodes = jnp.where(
        is_empty_sp[:, None, None], jnp.zeros_like(data.nodes[0]), data.nodes[rep_idxs]
    )
    reps_nodes_b = jnp.where(
        is_empty_sp[:, None], jnp.zeros_like(data.nodes_b[0]), data.nodes_b[rep_idxs]
    )
    reps_nodes_a = jnp.where(
        is_empty_sp[:, None], jnp.zeros_like(data.nodes_a[0]), data.nodes_a[rep_idxs]
    )
    return reps_conns, reps_conns_w, reps_nodes, reps_nodes_b, reps_nodes_a


def restore_species_frontrunners(
    data: NeatAlgoData,
    hparams: Hyperparams,
    top_conns: jnp.ndarray,
    top_conns_w: jnp.ndarray,
    top_nodes: jnp.ndarray,
    top_nodes_b: jnp.ndarray,
    top_nodes_a: jnp.ndarray,
) -> NeatAlgoData:
    """Overwrite species offspring with their previous generation top performers.
    For n_highperformers >= quota >=1, keep one offspring.

    Args:
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
        data.species
        top_conns, top_conns_w, top_nodes, top_nodes_b, top_nodes_a
        hparams.max_species
        hparams.n_intraspecies_culling_exempt_members
    Returns:
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
    """
    max_species = hparams.max_species
    n_highperformers = hparams.intraspecies_unchanged_frontrunners

    quotas = data.species[:, 2]
    member_counts = data.species[:, 3]
    is_empty_sp = data.species[:, 0] == -1
    quotas = jnp.where(is_empty_sp, 0, jnp.maximum(quotas, 0))
    species_offsets = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(quotas[:-1])]
    )

    def restore_sp_frontrunners(carry, sp_idx):
        cs, cs_w, nds, nds_b, nds_a = carry
        sp_offset = species_offsets[sp_idx]

        def restore_frontrunner(carry, rank):
            conns, conns_w, nodes, nodes_b, nodes_a = carry
            ind_idx = sp_offset + rank
            is_eligible = (member_counts[sp_idx] >= rank + 1) & (
                quotas[sp_idx] >= rank + 2
            )
            conns = conns.at[ind_idx].set(
                jnp.where(is_eligible, top_conns[sp_idx, rank], conns[ind_idx])
            )
            conns_w = conns_w.at[ind_idx].set(
                jnp.where(is_eligible, top_conns_w[sp_idx, rank], conns_w[ind_idx])
            )
            nodes = nodes.at[ind_idx].set(
                jnp.where(is_eligible, top_nodes[sp_idx, rank], nodes[ind_idx])
            )
            nodes_b = nodes_b.at[ind_idx].set(
                jnp.where(is_eligible, top_nodes_b[sp_idx, rank], nodes_b[ind_idx])
            )
            nodes_a = nodes_a.at[ind_idx].set(
                jnp.where(is_eligible, top_nodes_a[sp_idx, rank], nodes_a[ind_idx])
            )
            return (conns, conns_w, nodes, nodes_b, nodes_a), None

        (cs, cs_w, nds, nds_b, nds_a), _ = jax.lax.scan(
            restore_frontrunner,
            (cs, cs_w, nds, nds_b, nds_a),
            jnp.arange(n_highperformers),
        )
        return (cs, cs_w, nds, nds_b, nds_a), None

    (conns, conns_w, nodes, nodes_b, nodes_a), _ = jax.lax.scan(
        restore_sp_frontrunners,
        (data.conns, data.conns_w, data.nodes, data.nodes_b, data.nodes_a),
        jnp.arange(max_species),
    )
    return data.replace(
        conns=conns, conns_w=conns_w, nodes=nodes, nodes_b=nodes_b, nodes_a=nodes_a
    )


def measure_distance(
    cs0, w0, nds0, nds_b0, nds_a0, cs1, w1, nds1, nds_b1, nds_a1, hparams: Hyperparams
) -> jnp.ndarray:
    """Calculate the compatibility distance between two networks."""

    max_cs = hparams.max_cs

    def sort_by_innovation(cs, w):
        sort_key = jnp.where(cs[:, 2] > 0, cs[:, 2], jnp.iinfo(jnp.int32).max)
        order = jnp.argsort(sort_key)
        return cs[order], w[order]

    cs0, w0 = sort_by_innovation(cs0, w0)
    cs1, w1 = sort_by_innovation(cs1, w1)

    def cs_distance(inns0, enabled0, w0, inns1, enabled1, w1):
        is_empty0 = inns0 == 0
        is_empty1 = inns1 == 0
        n_genes0 = jnp.sum(~is_empty0, dtype=jnp.float32)
        n_genes1 = jnp.sum(~is_empty1, dtype=jnp.float32)
        max_inn0 = jnp.max(inns0)
        max_inn1 = jnp.max(inns1)
        inns0 = jnp.where(~is_empty0, inns0, jnp.iinfo(jnp.int32).max)
        inns1 = jnp.where(~is_empty1, inns1, jnp.iinfo(jnp.int32).max)
        idxs1 = jnp.searchsorted(inns1, inns0)
        idxs1 = jnp.minimum(idxs1, max_cs - 1)
        is_mtchng = (inns1[idxs1] == inns0) & ~is_empty0
        n_mtchng = jnp.sum(is_mtchng, dtype=jnp.float32)
        denom = jnp.maximum(1.0, n_mtchng)
        w_diff = (
            jnp.sum(jnp.abs(w0 - w1[idxs1]) * is_mtchng.astype(jnp.float32)) / denom
        )
        n_enabled_diff = jnp.sum(
            (enabled0 != enabled1[idxs1]).astype(jnp.float32)
            * is_mtchng.astype(jnp.float32)
        )
        n_excess0 = jnp.sum((inns0 > max_inn1) & ~is_empty0, dtype=jnp.float32)
        n_excess1 = jnp.sum((inns1 > max_inn0) & ~is_empty1, dtype=jnp.float32)
        n_excess = n_excess0 + n_excess1
        n_disjoint = n_genes0 + n_genes1 - 2 * n_mtchng - n_excess
        n = jnp.maximum(n_genes0, n_genes1).astype(jnp.float32)
        n = jnp.where(n > 20.0, n, 1.0)
        return (
            (hparams.compat_excess_coeff * n_excess / n)
            + (hparams.compat_disjoint_coeff * n_disjoint / n)
            + (hparams.compat_wght_coeff * w_diff)
            + (hparams.compat_enabled_coeff * n_enabled_diff)
        )

    def nds_distance(ids0, agg0, act0, b0, a0, ids1, agg1, act1, b1, a1):
        match_matrix = (ids0[:, None] == ids1[None, :]) & (ids0[:, None] != 0)
        is_mtchng = match_matrix.any(axis=1)
        idxs1 = jnp.argmax(match_matrix, axis=1)
        n_mtchng = jnp.sum(is_mtchng, dtype=jnp.float32)
        denom = jnp.maximum(1.0, n_mtchng)
        is_mtchng_f = is_mtchng.astype(jnp.float32)
        agg_diff = (
            jnp.sum((agg0 != agg1[idxs1]).astype(jnp.float32) * is_mtchng_f) / denom
        )
        act_diff = (
            jnp.sum((act0 != act1[idxs1]).astype(jnp.float32) * is_mtchng_f) / denom
        )
        b_diff = (
            jnp.sum(jnp.abs(b0 - b1[idxs1]).astype(jnp.float32) * is_mtchng_f) / denom
        )
        a_diff = (
            jnp.sum(jnp.abs(a0 - a1[idxs1]).astype(jnp.float32) * is_mtchng_f) / denom
        )
        return (
            (hparams.compat_agg_coeff * agg_diff)
            + (hparams.compat_act_coeff * act_diff)
            + (hparams.compat_bias_coeff * b_diff)
            + (hparams.compat_alpha_coeff * a_diff)
        )

    cs_dist = cs_distance(cs0[:, 2], cs0[:, 3], w0, cs1[:, 2], cs1[:, 3], w1)
    nds_dist = nds_distance(
        nds0[:, 0],
        nds0[:, 2],
        nds0[:, 3],
        nds_b0,
        nds_a0,
        nds1[:, 0],
        nds1[:, 2],
        nds1[:, 3],
        nds_b1,
        nds_a1,
    )
    return cs_dist + nds_dist


def speciate(
    data: NeatAlgoData,
    hparams: Hyperparams,
    reps_conns: jnp.ndarray,
    reps_conns_w: jnp.ndarray,
    reps_nodes: jnp.ndarray,
    reps_nodes_b: jnp.ndarray,
    reps_nodes_a: jnp.ndarray,
):
    """Iteratively assign individuals to species or make them unto new ones.
    Proceed from higher to decreasing fitness individuals.

    Args:
        data.conns
        data.conns_w
        data.nodes
        data.nodes_b
        data.nodes_a
        data.species
        data.species_best_f
        data.thresholds
        data.next_sp_id
        data.ind
        data.fitness
        reps_conns, reps_conns_w, reps_nodes, reps_nodes_b, reps_nodes_a
        hparams.max_cs (callee)
        hparams.compat_excess_coeff (callee)
        hparams.compat_disjoint_coeff (callee)
        hparams.compat_wght_coeff (callee)
        hparams.compat_enabled_coeff (callee)
        hparams.compat_agg_coeff (callee)
        hparams.compat_act_coeff (callee)
        hparams.compat_bias_coeff (callee)
        hparams.compat_alpha_coeff (callee)
        hparams.base_compat_threshold
    Returns:
        data.ind
        data.species
        data.species_best_f
        data.thresholds
        data.next_sp_id
        dists
    """
    base_threshold = hparams.base_compat_threshold

    ind_idxs = jnp.argsort(data.fitness[:, 0], descending=True)
    new_reps = jnp.full((hparams.max_species,), -1, dtype=jnp.int32)

    def place_individual(carry, ind_idx):
        (ind, species, species_best_f, thresholds, next_sp_id, new_reps) = carry
        is_empty_sp = species[:, 0] == -1
        distances = jax.vmap(
            measure_distance,
            in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0, None),
        )(
            data.conns[ind_idx],
            data.conns_w[ind_idx],
            data.nodes[ind_idx],
            data.nodes_b[ind_idx],
            data.nodes_a[ind_idx],
            jnp.where((new_reps >= 0)[:, None, None], data.conns[new_reps], reps_conns),
            jnp.where((new_reps >= 0)[:, None], data.conns_w[new_reps], reps_conns_w),
            jnp.where((new_reps >= 0)[:, None, None], data.nodes[new_reps], reps_nodes),
            jnp.where((new_reps >= 0)[:, None], data.nodes_b[new_reps], reps_nodes_b),
            jnp.where((new_reps >= 0)[:, None], data.nodes_a[new_reps], reps_nodes_a),
            hparams,
        )
        distances = jnp.where(is_empty_sp, jnp.inf, distances)

        is_close_enough = jnp.any(distances <= thresholds)
        is_below_species_cap = jnp.any(species[:, 0] == -1)

        branch = (
            0 * jnp.int32(is_close_enough)
            + (1 * jnp.int32(~is_close_enough & is_below_species_cap))
            + (2 * jnp.int32(~is_close_enough & ~is_below_species_cap))
        )

        def assign_to_closest_within_thresholds(args):
            (ind, species, species_best_f, thresholds, next_sp_id, new_reps) = args
            distances_within_thresholds = jnp.where(
                distances <= thresholds, distances, jnp.inf
            )
            sp_idx = jnp.argmin(distances_within_thresholds)
            ind = ind.at[ind_idx, 0].set(sp_idx)
            species = species.at[sp_idx, 3].add(1)
            return (ind, species, species_best_f, thresholds, next_sp_id, new_reps)

        def create_new(args):
            (ind, species, species_best_f, thresholds, next_sp_id, new_reps) = args
            sp_idx = jnp.argmax(species[:, 0] == -1)
            ind = ind.at[ind_idx, 0].set(sp_idx)
            species = species.at[sp_idx, :].set(
                jnp.array(
                    [data.gen, next_sp_id, 0, 1, data.gen, ind_idx], dtype=jnp.int32
                )
            )
            species_best_f = species_best_f.at[sp_idx].set(-jnp.inf)
            thresholds = thresholds.at[sp_idx].set(base_threshold)
            new_reps = new_reps.at[sp_idx].set(ind_idx)
            next_sp_id = next_sp_id + 1
            return (ind, species, species_best_f, thresholds, next_sp_id, new_reps)

        def assign_to_closest(args):
            (ind, species, species_best_f, thresholds, next_sp_id, new_reps) = args
            sp_idx = jnp.argmin(distances)
            ind = ind.at[ind_idx, 0].set(sp_idx)
            species = species.at[sp_idx, 3].add(1)
            return (ind, species, species_best_f, thresholds, next_sp_id, new_reps)

        args = (ind, species, species_best_f, thresholds, next_sp_id, new_reps)
        return jax.lax.switch(
            branch,
            [assign_to_closest_within_thresholds, create_new, assign_to_closest],
            args,
        ), None

    ((ind, species, species_best_f, thresholds, next_sp_id, new_reps), _) = (
        jax.lax.scan(
            place_individual,
            (
                data.ind,
                data.species,
                data.species_best_f,
                data.thresholds,
                data.next_sp_id,
                new_reps,
            ),
            ind_idxs,
        )
    )

    is_new_species = (new_reps >= 0)[:, None]
    dists = jax.vmap(
        jax.vmap(
            measure_distance,
            in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0, None),
        ),
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None),
    )(
        jnp.where(is_new_species[:, None], data.conns[new_reps], reps_conns),
        jnp.where(is_new_species, data.conns_w[new_reps], reps_conns_w),
        jnp.where(is_new_species[:, None], data.nodes[new_reps], reps_nodes),
        jnp.where(is_new_species, data.nodes_b[new_reps], reps_nodes_b),
        jnp.where(is_new_species, data.nodes_a[new_reps], reps_nodes_a),
        data.conns,
        data.conns_w,
        data.nodes,
        data.nodes_b,
        data.nodes_a,
        hparams,
    )
    is_from_sp = ind[:, 0][None, :] == jnp.arange(hparams.max_species)[:, None]
    dists = jnp.where(is_from_sp, dists, -1.0)

    data = data.replace(
        ind=ind,
        species=species,
        next_sp_id=next_sp_id,
        species_best_f=species_best_f,
        thresholds=thresholds,
    )
    return data, dists


def erase_vacant_species(data: NeatAlgoData) -> NeatAlgoData:
    """Erase species that received no individuals during speciation."""
    is_vacant = data.species[:, 3] == 0
    empty_sp = jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32)
    species = jnp.where(is_vacant[:, None], empty_sp, data.species)
    species_best_f = jnp.where(is_vacant, -jnp.inf, data.species_best_f)
    thresholds = jnp.where(is_vacant, -1.0, data.thresholds)
    return data.replace(
        species=species, species_best_f=species_best_f, thresholds=thresholds
    )


def elect_reps(data: NeatAlgoData, hparams: Hyperparams, dists: jnp.ndarray):
    """Elect each species' representative as the member closest to its current representative.

    Args:
        data.ind
        data.species
        dists
        hparams.max_species
    Returns:
        data.species
    """
    max_species = hparams.max_species

    def pick_rep(sp_idx):
        is_from_sp = data.ind[:, 0] == sp_idx
        dsts = jnp.where(is_from_sp, dists[sp_idx], jnp.inf)
        is_empty_sp = data.species[sp_idx, 0] == -1
        return jnp.where(is_empty_sp, -1, jnp.argmin(dsts))

    reps = jax.vmap(pick_rep)(jnp.arange(max_species))
    species = data.species.at[:, 5].set(reps)
    return data.replace(species=species)


def adjust_thresholds(
    data: NeatAlgoData, hparams: Hyperparams, dists: jnp.ndarray
) -> NeatAlgoData:
    """Adjust the per-species compatibility threshold based on the average distance to a representative.

    Args:
        data.ind
        data.species
        data.thresholds
        data.gen
        dists
        hparams.max_species
        hparams.base_compat_threshold
        hparams.dynamic_threshold_ratio
        hparams.dynamic_threshold_learning_rate
    Returns:
        data.thresholds
    """
    max_species = hparams.max_species
    base_threshold = hparams.base_compat_threshold
    target_ratio = hparams.dynamic_threshold_ratio
    lr = hparams.dynamic_threshold_learning_rate

    is_empty_species = data.species[:, 0] == -1

    def adjust_threshold(sp_idx):
        is_from_sp = data.ind[:, 0] == sp_idx
        member_count = data.species[sp_idx, 3].astype(jnp.float32)
        sum_dist = jnp.sum(jnp.where(is_from_sp, dists[sp_idx], 0.0))
        gens_since_improving_best_f = data.gen - data.species[sp_idx, 0]
        is_updatable = gens_since_improving_best_f >= 5
        avg_dist = sum_dist / jnp.maximum(member_count, 1.0)
        old = data.thresholds[sp_idx]
        new = old + lr * (avg_dist - target_ratio * old)
        new = jnp.clip(new, 0.0, base_threshold)
        new = jnp.where(is_updatable, new, old)
        return jnp.where(is_empty_species[sp_idx], -1.0, new)

    thresholds = jax.vmap(adjust_threshold)(jnp.arange(max_species))
    return data.replace(thresholds=thresholds)
