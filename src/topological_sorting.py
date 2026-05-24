import jax
import jax.numpy as jnp

from .data import NeatAlgoData


def compute_dags_toposorts(data: NeatAlgoData) -> NeatAlgoData:
    """Find one possible topological sort for each network, assuming no cycles.

    Args:
        data.conns
        data.nodes
        hparams.max_nds (inferred)
    Returns:
        data.toposorted_idxs
    """
    toposorted_idxs = jax.vmap(compute_dag_toposort)(data.conns, data.nodes)
    return data.replace(toposorted_idxs=toposorted_idxs)


def compute_dag_toposort(cs, nds) -> jnp.ndarray:
    adj_mtrx = adjacency_matrix(cs, nds)
    toposrtd_idxs = kahns_algorithm(adj_mtrx)
    toposrtd_idxs = unpad_empty_indexes(toposrtd_idxs)
    return toposrtd_idxs


def adjacency_matrix(cs, nds):
    """Adjacency matrix of a network padded with -1s at rows/columns corresponding to empty node slots.
    For a network i, the row, column at index j correspond to the node whose id is at nodes[i, j, 0]"""
    in_ids = cs[:, 0]
    out_ids = cs[:, 1]
    ssort_idxs = jnp.argsort(nds[:, 0])
    ssort_ids = nds[:, 0][ssort_idxs]
    in_idxs = ssort_idxs[
        jnp.clip(jnp.searchsorted(ssort_ids, in_ids), 0, nds.shape[0] - 1)
    ]
    out_idxs = ssort_idxs[
        jnp.clip(jnp.searchsorted(ssort_ids, out_ids), 0, nds.shape[0] - 1)
    ]
    existing_conns = in_ids != 0
    adj_mtrx = (
        jnp.zeros((nds.shape[0], nds.shape[0]), dtype=jnp.int32)
        .at[in_idxs, out_idxs]
        .set(existing_conns.astype(jnp.int32))
    )
    is_empty = nds[:, 0] == 0
    adj_mtrx = jnp.where(is_empty[:, None] | is_empty[None, :], -1, adj_mtrx)
    return adj_mtrx


def kahns_algorithm(adj_mtrx):
    """Kahn's Topological Sorting Algorithm on the adjacency matrix of a network; assumes no cycles.

    Example input:
    [[  0  1 -1  0 -1 ]
     [  0  0 -1  1 -1 ]
     [ -1 -1 -1 -1 -1 ]
     [  0  0 -1  0 -1 ]
     [ -1 -1 -1 -1 -1 ]]
    Example output:
     [  0  1  3 -1 -1 ]
    """
    max_n = adj_mtrx.shape[0]
    nds_idxs = -jnp.ones((max_n), dtype=jnp.int32)
    indegrees = jnp.sum(adj_mtrx == 1, axis=0)
    indegrees = jnp.where(jnp.all(adj_mtrx == -1, axis=0), -1, indegrees)

    def body(carry, _):
        nds_idxs, indegrees, iter = carry
        next_nd = jnp.where(indegrees == 0, jnp.arange(max_n), -1).max()
        nds_idxs = nds_idxs.at[iter].set(next_nd)
        next_nd_successors = jnp.where(next_nd >= 0, adj_mtrx[next_nd], 0).astype(
            jnp.bool_
        )
        indegrees = jnp.where(next_nd_successors, indegrees - 1, indegrees)
        return (nds_idxs, indegrees.at[next_nd].set(-1), iter + 1), None

    nds_idxs, _, _ = jax.lax.scan(body, (nds_idxs, indegrees, 0), None, max_n)[0]
    return nds_idxs


def unpad_empty_indexes(arr):
    """Add the missing indexes corresponding to empty node slots.

    Example input:
    [  0  1  3  -1 -1 ]
    Example output:
    [  0  1  3   2  4 ]
    """
    missing_indexes = jnp.sort(
        jnp.setdiff1d(jnp.arange(arr.shape[0]), arr, size=arr.shape[0], fill_value=-1)
    )
    return jnp.where(arr == -1, missing_indexes, arr)
