
import os

import matplotlib.pyplot as plt
import numpy as np

from mixture_greedy.config import Config
from mixture_greedy.data.features import (
    _candidate_feature_npz_paths,
    _load_features_from_npz,
    _resolve_existing_path,
)
from mixture_greedy.rke_offline import RKEOfflineEvaluator
from mixture_greedy.rke_online import RKEOnlineEvaluator
from mixture_greedy.metrics.convergence import alpha_convergence
from mixture_greedy.metrics.distances import compute_pairwise_distance
from mixture_greedy.metrics.rke import (
    KernelUtils,
    _linear_coeff,
    compute_fid,
    compute_linear_term,
    compute_nearest_neighbour_distances,
    compute_reals_nearest_neighbour_distances,
    default_config,
    get_kth_value,
)

# Legacy scalar aliases intentionally capture the shared defaults at import time.
OGD = default_config.OGD
ALPHA_PRINT = default_config.ALPHA_PRINT
RESCALER = default_config.RESCALER
KID_NEW = default_config.KID_NEW
QUADRATIC_METRIC = default_config.QUADRATIC_METRIC
LINEAR_METRIC = default_config.LINEAR_METRIC
NUMBER_OF_SIMULATIONS = default_config.NUMBER_OF_SIMULATIONS
SIGMA = default_config.SIGMA
DELTA = default_config.DELTA
BETA = default_config.BETA
INITIAL_SAMPLE_COUNT = default_config.INITIAL_SAMPLE_COUNT
OFFLINE_CUTOFF = default_config.OFFLINE_CUTOFF
DATASET_REAL_CUTOFF = default_config.DATASET_REAL_CUTOFF
MINI_BATCH = default_config.MINI_BATCH
DELTA_L = default_config.DELTA_L
DELTA_G = default_config.DELTA_G
TOTAL_ROUNDS = default_config.TOTAL_ROUNDS
BLOCK_SIZE = default_config.BLOCK_SIZE
small_amount = default_config.small_amount
LAMBDA = default_config.LAMBDA
KNN = default_config.KNN
EPSILON = default_config.EPSILON
feature_extractor = default_config.feature_extractor
DEVICE = default_config.DEVICE
DEBUG = default_config.DEBUG
CHANGE_POINT = default_config.CHANGE_POINT
CHANGE_POINT_PATH = default_config.CHANGE_POINT_PATH


def _save_curve(values, *, ylabel, title, output_path, alpha=1.0):
    """Save one legacy diagnostic curve with the original plot styling."""
    plt.figure(figsize=(10, 5))
    plt.plot(values, alpha=alpha)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path)


def run_legacy_experiment():
    """Run the original RKE/KID experiment with its historical defaults."""
    global cluster_name, dataset_name, experiment, mode, model_names, with_quality

    model_names = [
        "FFHQ-Efficient VDVAE-dgm",
        "FFHQ-LDM",
        "FFHQ-LDM-dgm",
        "FFHQ-stylegan-0.5",
        "FFHQ-stylegan-0.7",
        "FFHQ-stylegan2-ada-0.5",
        "FFHQ-stylegan2-ada-0.7",
        "stylegan-xl-0.7",
    ]
    dataset_name = "ffhq"
    cluster_name = None
    mode = "mixture-ucb"
    experiment = ""
    with_quality = True

    print("MODE is -> ", mode)
    scores_of_simulations = {}
    offline_evaluator = RKEOfflineEvaluator(
        model_names, dataset_name, has_reference=with_quality
    )
    offline_evaluator.print_model_scores()
    print(offline_evaluator.optimal_alphas)
    true_kernel = KernelUtils.scaled_kernel(
        offline_evaluator.kernel, offline_evaluator._get_dataset_sizes()
    )
    true_alphas = offline_evaluator.optimal_alphas
    L_star = offline_evaluator.optimal_ans
    true_precision = np.array(list(offline_evaluator.linears.values()))

    for simulation in range(NUMBER_OF_SIMULATIONS):
        print("SIMULATION ROUND - ", simulation)
        online_evaluator = RKEOnlineEvaluator(
            model_names,
            dataset_name,
            offline_evaluator,
            mode,
            use_linear=with_quality,
        )
        online_evaluator.run_online_evaluation(num_rounds=TOTAL_ROUNDS)
        online_evaluator.report_sample_nums()
        scores_of_simulations[f"{simulation}"] = online_evaluator.scores

        instantaneous_regret = []
        for round_index, alpha_t in enumerate(online_evaluator.alpha_history):
            objective = float(alpha_t @ true_kernel @ alpha_t) - LAMBDA * float(
                alpha_t @ true_precision
            )
            instantaneous_regret.append(objective - L_star)
            if round_index in (2999, 5999):
                print(round_index, alpha_t)

        instantaneous_regret = np.array(instantaneous_regret)
        cumulative_regret = np.cumsum(instantaneous_regret)
        average_regret = cumulative_regret / (
            np.arange(len(cumulative_regret)) + 1
        )
        _, _, kl_divergence = alpha_convergence(
            online_evaluator.alpha_history,
            offline_evaluator.optimal_alphas,
        )

        _save_curve(
            instantaneous_regret,
            ylabel="instantaneous regret",
            title="Instantaneous RKE regret over time",
            output_path=f"{dataset_name}/inst_regret_{TOTAL_ROUNDS}.png",
            alpha=0.5,
        )
        _save_curve(
            average_regret,
            ylabel="average regret R_t / t",
            title=(
                "Average cumulative RKE regret with last one being "
                f"{average_regret[-1]}"
            ),
            output_path=f"{dataset_name}/avg_regret_{TOTAL_ROUNDS}.png",
        )
        _save_curve(
            kl_divergence,
            ylabel="KLD",
            title=f"KLD with last one being {kl_divergence[-1]}",
            output_path=f"{dataset_name}/kld_{TOTAL_ROUNDS}.png",
        )

    cluster_extension = f"_{cluster_name}" if cluster_name is not None else ""
    output_directory = f"./{experiment}/{dataset_name}{cluster_extension}/{SIGMA}"
    os.makedirs(output_directory, exist_ok=True)

    extension = ""
    if mode in ["one-arm-ucb", "mixture-ucb"]:
        extension += f"-{DELTA_L}"
        if with_quality:
            extension += f"-{DELTA_G}"
    ogd_extension = "-OGD" if OGD else ""
    np.savez(
        f"{output_directory}/{mode}{extension}{ogd_extension}.npz",
        **scores_of_simulations,
    )

    offline_evaluator.print_model_scores()
    print(offline_evaluator.optimal_alphas)


if __name__ == "__main__":
    run_legacy_experiment()
