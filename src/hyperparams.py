from flax import struct


@struct.dataclass
class Hyperparams:
    seed: int = 1
    pop_size: int = struct.field(default=500, pytree_node=False)
    n_inputs: int = struct.field(default=12, pytree_node=False)
    n_outputs: int = struct.field(default=3, pytree_node=False)
    max_nds: int = struct.field(default=50, pytree_node=False)
    max_cs: int = struct.field(default=2500, pytree_node=False)
    max_species: int = struct.field(default=7, pytree_node=False)
    feedforward: bool = struct.field(default=False, pytree_node=False)
    ctrnn_integration_steps: int = struct.field(default=10, pytree_node=False)
    default_output_activation: bool = False
    percentage_of_possible_initial_connections: float = 0.2
    fitness_maxabs_offset: float = 0.1
    positive_fitnesses_scaling: float = 1.0
    add_one_c_prob: float = 0.3
    add_one_nd_prob: float = 0.03
    erase_split_c: bool = struct.field(default=False, pytree_node=False)
    change_act_prob: float = 0.02
    change_agg_prob: float = 0.01
    perturb_wght_stdev: float = 0.3
    replace_wght_stdev: float = 5.0
    perturb_wght_prob: float = 0.7
    replace_wght_prob: float = 0.03
    max_wght: float = 99999.0
    min_wght: float = -99999.0
    perturb_bias_stdev: float = 0.3
    replace_bias_stdev: float = 5.0
    perturb_bias_prob: float = 0.7
    replace_bias_prob: float = 0.03
    max_bias: float = 99999.0
    min_bias: float = -99999.0
    perturb_alpha_stdev: float = 0.03
    perturb_alpha_prob: float = 0.7
    replace_alpha_prob: float = 0.03
    ctrnn_min_alpha: float = struct.field(default=0.05, pytree_node=False)
    ctrnn_max_alpha: float = struct.field(default=0.3, pytree_node=False)
    disable_c_prob: float = 0.001
    reenable_one_c_prob: float = 0.1
    erase_c_prob: float = 0.0
    erase_disjointed_nd_prob: float = 0.0
    crossover_rate: float = 0.70
    enabled_recessiveness_prob: float = 0.75
    avg_wghts_prob: float = 0.35
    intraspecies_parenthood_eligibility: float = 0.3
    intraspecies_unchanged_frontrunners: int = struct.field(
        default=5, pytree_node=False
    )
    interspecies_mating_ratio: float = 0.02
    species_fitness_powerscaling: float = 1.5
    compat_excess_coeff: float = 1.0
    compat_disjoint_coeff: float = 1.0
    compat_wght_coeff: float = 0.4
    compat_enabled_coeff: float = 1.0
    compat_agg_coeff: float = 0.5
    compat_act_coeff: float = 0.5
    compat_bias_coeff: float = 0.2
    compat_alpha_coeff: float = 0.2
    base_compat_threshold: float = 3.0
    dynamic_thresholds: bool = struct.field(default=False, pytree_node=False)
    dynamic_threshold_ratio: float = 0.8
    dynamic_threshold_learning_rate: float = 0.1
    base_stagnation_age: int = 25
    stagnation_age_powerlaw_growth: float = 0.0
    n_stagnation_exempt_top_species: int = 0

    def __post_init__(self):
        """Catch hyperparameter configurations that break the program."""
        assert self.max_nds > self.n_inputs + self.n_outputs, (
            "hparams.max_nds is not big enough to initialize the networks"
        )
        assert self.max_cs >= self.n_inputs * self.n_outputs, (
            "hparams.max_cs is not big enough to initialize all initial connections"
        )
        assert self.pop_size > self.max_species, (
            "hparams.pop_size must be bigger than hparams.max_species"
        )
        assert self.intraspecies_unchanged_frontrunners <= self.pop_size, (
            "hparams.intraspecies_unchanged_frontrunners must not be bigger than hparams.pop_size"
        )
        assert self.ctrnn_min_alpha >= 0.0, "hparams.ctrnn_max_alpha should be >= 0.0"
        assert self.ctrnn_max_alpha <= 1.0, (
            "hparams.ctrnn_max_alpha should be <= 1.0 for the recurrent net to have memory"
        )
