import jax.numpy as jnp

from ..hyperparams import Hyperparams
from ..data import NeatAlgoData


def scale_positive_fitnesses(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """f(i) = f(i) * positive_fitnesses_scaling  if f(i) > 0
              f(i)                               otherwise

    Args:
        data.fitness
        hparams.positive_fitnesses_scaling
    Returns:
        data.fitness
    """
    scale = hparams.positive_fitnesses_scaling
    scaled_fitness = jnp.where(
        data.fitness[:, 0] > 0, data.fitness[:, 0] * scale, data.fitness[:, 0]
    )
    return data.replace(fitness=data.fitness.at[:, 1].set(scaled_fitness))


def offset_fitnesses(data: NeatAlgoData, hparams: Hyperparams) -> NeatAlgoData:
    """f = f - min(f) + fitness_maxabs_offset*max(abs(f))

    Args:
        data.fitness
        hparams.fitness_maxabs_offset
    Returns:
        data.fitness
    """
    offset_prctg = hparams.fitness_maxabs_offset
    scaled_fitness = data.fitness[:, 1]
    is_finite = jnp.isfinite(scaled_fitness)
    min_fitness = jnp.where(
        jnp.any(is_finite),
        jnp.min(jnp.where(is_finite, scaled_fitness, jnp.inf)),
        -jnp.inf,
    )
    absmax_fitness = jnp.where(
        jnp.any(is_finite),
        jnp.max(jnp.where(is_finite, jnp.abs(scaled_fitness), 0.0)),
        0.0,
    )
    offset = jnp.where(is_finite, -min_fitness + (absmax_fitness * offset_prctg), 0.0)
    offset_fitness = scaled_fitness + offset
    return data.replace(fitness=data.fitness.at[:, 2].set(offset_fitness))
