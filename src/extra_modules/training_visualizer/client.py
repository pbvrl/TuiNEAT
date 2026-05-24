import jax
import jax.numpy as jnp
import numpy as np
import grpc

from src.hyperparams import Hyperparams
from ...data import NeatAlgoData

import src.extra_modules.training_visualizer.grpc_api.schema_pb2 as pb
import src.extra_modules.training_visualizer.grpc_api.schema_pb2_grpc as grpc_api

_devices = jax.devices()
IS_GPU_ENABLED = any(device.platform == "gpu" for device in _devices)

N_SENT_NETWORKS_PER_GROUP = 3

VIZ_TUNABLE_HPARAMS = [
    ("addOneConnProb", "add_one_c_prob", float),
    ("addOneNodeProb", "add_one_nd_prob", float),
    ("changeActProb", "change_act_prob", float),
    ("changeAggProb", "change_agg_prob", float),
    ("defaultOutputActivation", "default_output_activation", bool),
    ("perturbWghtStdev", "perturb_wght_stdev", float),
    ("replaceWghtStdev", "replace_wght_stdev", float),
    ("perturbWghtProb", "perturb_wght_prob", float),
    ("replaceWghtProb", "replace_wght_prob", float),
    ("perturbBiasStdev", "perturb_bias_stdev", float),
    ("replaceBiasStdev", "replace_bias_stdev", float),
    ("perturbBiasProb", "perturb_bias_prob", float),
    ("replaceBiasProb", "replace_bias_prob", float),
    ("perturbAlphaStdev", "perturb_alpha_stdev", float),
    ("perturbAlphaProb", "perturb_alpha_prob", float),
    ("replaceAlphaProb", "replace_alpha_prob", float),
    ("disableCProb", "disable_c_prob", float),
    ("reenableOneCProb", "reenable_one_c_prob", float),
    ("eraseConnProb", "erase_c_prob", float),
    ("crossoverRate", "crossover_rate", float),
    ("enabledRecessivenessProb", "enabled_recessiveness_prob", float),
    ("avgWeightsProb", "avg_wghts_prob", float),
    ("intraspeciesParenthoodEligibility", "intraspecies_parenthood_eligibility", float),
    ("interspeciesMatingRatio", "interspecies_mating_ratio", float),
    ("speciesFitnessPowerscaling", "species_fitness_powerscaling", float),
    ("compatExcessCoeff", "compat_excess_coeff", float),
    ("compatDisjointCoeff", "compat_disjoint_coeff", float),
    ("compatWghtCoeff", "compat_wght_coeff", float),
    ("compatEnabledCoeff", "compat_enabled_coeff", float),
    ("compatAggCoeff", "compat_agg_coeff", float),
    ("compatActCoeff", "compat_act_coeff", float),
    ("compatBiasCoeff", "compat_bias_coeff", float),
    ("compatAlphaCoeff", "compat_alpha_coeff", float),
    ("baseCompatThreshold", "base_compat_threshold", float),
    ("dynamicThresholds", "dynamic_thresholds", bool),
    ("dynamicThresholdRatio", "dynamic_threshold_ratio", float),
    ("dynamicThresholdLearningRate", "dynamic_threshold_learning_rate", float),
    ("baseStagnationAge", "base_stagnation_age", int),
    ("stagnationAgePowerlawGrowth", "stagnation_age_powerlaw_growth", float),
    ("nStagnationExemptTopSpecies", "n_stagnation_exempt_top_species", int),
]


class TrainingVisualizerClient:
    def __init__(
        self,
        hparams: Hyperparams,
        rollout_repeats: int,
        max_rollout_steps: int,
        task_name: str,
        input_names: list[str],
        output_names: list[str],
        feedforward: bool = False,
        normalize_obs: bool = False,
        server_port: str = "50051",
    ):
        self.hparams = hparams
        self.rollout_repeats = rollout_repeats
        self.max_rollout_steps = max_rollout_steps
        self.task_name = task_name
        self.input_names = input_names
        self.output_names = output_names
        self.feedforward = feedforward
        self.normalize_obs = normalize_obs
        self.channel = grpc.insecure_channel(f"localhost:{server_port}")

    def sync(self, data: NeatAlgoData) -> Hyperparams:
        """Send visualization data and receive a request for hyperparams."""
        viz_data = self.prepare_data(data)
        stub = grpc_api.TrainingVisualizerStub(self.channel)
        response = stub.SyncVisualization(viz_data)
        self.hparams = self.hparams.replace(
            **{
                hp: cast(getattr(response.vizTunableHparams, viz))
                for viz, hp, cast in VIZ_TUNABLE_HPARAMS
            }
        )
        return self.hparams

    def prepare_data(self, data: NeatAlgoData) -> pb.TrainingData:
        """Format the data to the protobuffers schema."""
        hparams = self.hparams

        viz_tunable_hparams = pb.VizTunableHparams(
            **{viz: getattr(hparams, hp) for viz, hp, _ in VIZ_TUNABLE_HPARAMS}
        )
        viz_hparams = pb.VizHparams(
            vizTunableHparams=viz_tunable_hparams,
            popSize=hparams.pop_size,
            feedforward=self.feedforward,
            ctrnnIntegrationSteps=hparams.ctrnn_integration_steps,
            intraspeciesUnchangedFrontrunners=hparams.intraspecies_unchanged_frontrunners,
            normalizeObs=self.normalize_obs,
        )
        perf_related_params = pb.PerformanceRelatedParams(
            popSize=hparams.pop_size,
            gpuEnabled=IS_GPU_ENABLED,
            maxConns=hparams.max_cs,
            maxNodes=hparams.max_nds,
            rolloutRepeats=self.rollout_repeats,
            maxRolloutSteps=self.max_rollout_steps,
        )

        def sp_top_networks(species_idx):
            is_from_sp = data.ind[:, 0] == species_idx
            sp_fitnesses = jnp.where(is_from_sp, data.fitness[:, 0], -jnp.inf)
            top_idxs = jnp.argsort(-sp_fitnesses)[:N_SENT_NETWORKS_PER_GROUP]
            is_empty_sp = data.species[species_idx, 0] == -1
            is_empty = (sp_fitnesses[top_idxs] == -jnp.inf) | is_empty_sp
            return jnp.where(~is_empty, top_idxs, -1)

        top_idxs = jax.vmap(sp_top_networks)(jnp.arange(hparams.max_species))
        top_idxs = top_idxs.ravel()
        top_idxs = top_idxs[top_idxs != -1]

        min_fs = jax.ops.segment_min(
            data.fitness[:, 0], data.ind[:, 0], hparams.max_species
        )
        max_fs = jax.ops.segment_max(
            data.fitness[:, 0], data.ind[:, 0], hparams.max_species
        )

        toposorted_idxs = jnp.where(
            self.feedforward,
            data.toposorted_idxs,
            jnp.tile(jnp.arange(hparams.max_nds), (hparams.pop_size, 1)),
        )

        top_idxs = np.asarray(top_idxs)
        max_fs = np.asarray(max_fs)
        min_fs = np.asarray(min_fs)
        conns = np.asarray(data.conns)
        conns_w = np.asarray(data.conns_w)
        nodes = np.asarray(data.nodes)
        nodes_b = np.asarray(data.nodes_b)
        nodes_a = np.asarray(data.nodes_a)
        fitness = np.asarray(data.fitness)
        ind = np.asarray(data.ind)
        species = np.asarray(data.species)
        thresholds = np.asarray(data.thresholds)
        toposorted_idxs = np.asarray(toposorted_idxs)

        enabled_weights = conns_w[(conns[:, :, 0] != 0) & (conns[:, :, 3] == 1)]
        if len(enabled_weights) > 0:
            min_weight = float(np.min(enabled_weights))
            max_weight = float(np.max(enabled_weights))
        else:
            min_weight = 0.0
            max_weight = 0.0

        existing_biases = nodes_b[nodes[:, :, 0] != 0]
        min_bias = float(np.min(existing_biases))
        max_bias = float(np.max(existing_biases))

        non_input_mask = (nodes[:, :, 0] != 0) & (nodes[:, :, 1] != 1)
        if non_input_mask.any():
            existing_alphas = nodes_a[non_input_mask]
            ctrnn_min_alpha = float(np.min(existing_alphas))
            ctrnn_max_alpha = float(np.max(existing_alphas))
        else:
            ctrnn_min_alpha = 0.0
            ctrnn_max_alpha = 0.0

        top_conns = conns[top_idxs]
        top_conns_w = conns_w[top_idxs]
        top_nodes = nodes[top_idxs]
        top_nodes_b = nodes_b[top_idxs]
        top_nodes_a = nodes_a[top_idxs]
        top_ind_fitnesses = fitness[top_idxs, 0]
        top_toposorted_idxs = toposorted_idxs[top_idxs]

        ind_species_ids = species[ind[top_idxs, 0], 1]
        quotas = species[:, 2]
        fitnesses = fitness[:, 0]
        gen = int(data.gen)

        species_list = [
            pb.Species(
                id=int(species[idx, 1]),
                quota=int(quotas[idx]),
                memberCount=int(species[idx, 3]),
                minFitness=float(min_fs[idx]),
                maxFitness=float(max_fs[idx]),
                compatThreshold=float(thresholds[idx]),
                stagnation=int(gen - species[idx, 0]),
            )
            for idx in range(hparams.max_species)
        ]

        networks = []
        for i in range(len(top_idxs)):
            cs = top_conns[i]
            cs_w = top_conns_w[i]
            is_empty = cs[:, 0] == 0
            cs = cs[~is_empty]
            cs_w = cs_w[~is_empty]
            cs_list = [
                pb.Connection(
                    inId=int(c[0]), outId=int(c[1]), enabled=bool(c[3]), weight=float(w)
                )
                for c, w in zip(cs, cs_w)
            ]

            toposrtd_idxs = top_toposorted_idxs[i]
            nds = top_nodes[i][toposrtd_idxs]
            nds_b = top_nodes_b[i][toposrtd_idxs]
            nds_a = top_nodes_a[i][toposrtd_idxs]
            is_empty = nds[:, 0] == 0
            nds = nds[~is_empty]
            nds_b = nds_b[~is_empty]
            nds_a = nds_a[~is_empty]
            nds_list = [
                pb.Node(
                    id=int(nd[0]),
                    type=int(nd[1]),
                    aggregation=int(nd[2]),
                    activation=int(nd[3]),
                    bias=float(nd_b),
                    ctrnnAlpha=float(nd_a),
                )
                for nd, nd_b, nd_a in zip(nds, nds_b, nds_a)
            ]

            networks.append(
                pb.Network(
                    cs=cs_list,
                    toposrtdNds=nds_list,
                    fitness=float(top_ind_fitnesses[i]),
                    speciesId=int(ind_species_ids[i]),
                )
            )

        pop_avg_fitness = float(np.mean(fitnesses))
        modified_stagnation_age = int(np.asarray(data.modified_stagnation_age))

        return pb.TrainingData(
            gen=gen,
            networks=networks,
            species=species_list,
            popAvgFitness=pop_avg_fitness,
            modifiedStagnationAge=modified_stagnation_age,
            perfRelatedParams=perf_related_params,
            vizHparams=viz_hparams,
            minWeight=min_weight,
            maxWeight=max_weight,
            minBias=min_bias,
            maxBias=max_bias,
            ctrnnMinAlpha=ctrnn_min_alpha,
            ctrnnMaxAlpha=ctrnn_max_alpha,
            taskName=self.task_name,
            inputNames=self.input_names,
            outputNames=self.output_names,
        )
