import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from online_reproducible import RKEOfflineEvaluator, RKEOnlineEvaluator, Config


config = Config(
    OGD=False,
    ALPHA_PRINT=10000,
    RESCALER=1,
    KID_NEW=False,
    QUADRATIC_METRIC='kid',  # 'rke' or 'kid'
    LINEAR_METRIC='kid',
    NUMBER_OF_SIMULATIONS=5,
    SIGMA=30,
    DELTA=0.03,
    BETA=1,
    INITIAL_SAMPLE_COUNT=3,
    OFFLINE_CUTOFF=10000,
    DATASET_REAL_CUTOFF=5000,
    MINI_BATCH=1,
    DELTA_L=0.6,
    DELTA_G=0.4,
    TOTAL_ROUNDS=7000,
    BLOCK_SIZE=800,
    small_amount=1e-8,
    DYNAMIC_EPSILON = False,
    LAMBDA=0,
    KNN=5,
    EG_ETA=0.01,
    EG_SQRT_DECAY=False,
    EG_STEPS=10,
    ALPHA_FLOOR=0.0,
    feature_extractor='dino',
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    DEBUG=False,
)

EG_STEP_GRID = [1, 5, 10, 100]
EG_ETA_GRID = [0.001, 0.01, 0.1, 0.0001]

dataset = "FFHQ256"
if dataset == "FFHQ256":
    names = [ "efficient-vdvae", "insgen",  "ldm",  "stylegan-xl",  "stylenat"]
if dataset == "FAHQ":
    names = ["AFHQ-styleGan2-Ada", "stylegan-xl",  "ldm", "efficient-vdvae"]

elif dataset == "Imagenet256":
    names = ['imagenet256-DiT-XL-2-guided', 'Imagenet256-LDM', 'imagenet256-RQ-Transformer', 'imagenet256-StyleGAN-XL']
elif dataset == "lsun":
    names = ["iddpm_lsun_bedroom", "lsun_projected-gan-0.7",  "lsun-unleashing_transformers", "lsun_stylegan-0.7"]
elif dataset == "red_birds":
    names = ["kandinsky","pixart","sd3"]
elif dataset == "city":
    names = ["llama", "gemma", "qwen"]
elif dataset == "t2i_dog":
    names = ['poodle', 'bulldog', 'german_shepherd', 'golden_retriever', 'havanese']
elif dataset == "t2i_car":
    names = ["realistic", "surreal", "cartoony"]


def plot_results(path, modes_names, sigma, total_rounds, metric_name='scores'):
    """Plot the results from saved .npz files.
    
    Args:
        path: Directory containing the .npz files
        modes_names: List of method names
        sigma: Sigma value used in the experiment
        total_rounds: Total number of rounds
        metric_name: 'scores' for KID/RKE or 'fid' for FID
    """
    methods = [f'{name}_{sigma}' for name in modes_names]

    # Style map for different methods
    style = {
        f'Mixture-UCB_{sigma}':     dict(color='b', linestyle='-',  marker='s'),
        f'Mixture Oracle_{sigma}':  dict(color='r', linestyle='--', marker='s'),
        f'Mixture Greedy_{sigma}':  dict(color='g', linestyle='-',  marker='s'),
        f'Mixture Greedy EG_{sigma}':  dict(color='darkgreen', linestyle='-.', marker='D'),
        f'Mixture-UCB_dl0.2_{sigma}':  dict(color='navy', linestyle='-',  marker='s'),
        f'Mixture-UCB_dl0.6_{sigma}':  dict(color='royalblue', linestyle='-',  marker='s'),
        f'Mixture-UCB_dl0.9_{sigma}':  dict(color='cornflowerblue', linestyle='-',  marker='s'),
    }

    # Display names for legend (without sigma, and ts -> Thompson Sampling)
    display_names = {
        f'Mixture-UCB_{sigma}':     'Mixture UCB',
        f'Mixture Oracle_{sigma}':  'Mixture Oracle',
        f'One-Arm UCB_{sigma}':     'Vanilla UCB',
        f'ts_{sigma}':              'Mixture TS',
        f'Mixture Greedy_{sigma}':  'Mixture Greedy',
        f'Mixture Greedy EG_{sigma}':  'Mixture Greedy EG',
        f'One-Arm TS_{sigma}':     'Vanilla TS',
        f'Mixture-UCB_dl0.2_{sigma}':  r'$\delta_L = 0.2$',
        f'Mixture-UCB_dl0.6_{sigma}':  'Mixture UCB',
        f'Mixture-UCB_dl0.9_{sigma}':  r'$\delta_L = 0.9$',

    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for method in methods:
        fp = os.path.join(path, f'{method}_{metric_name}.npz')
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skipping.")
            continue
        z = np.load(fp)
        scores = z['arr_0']
        scores = np.asarray(scores).squeeze()

        print(f"Method: {method}, Total Rounds in Scores: {len(scores)}")  # Print the true number of rounds

        # FID is computed every 500 rounds, so no subsampling needed
        if metric_name == 'fid':
            # FID scores are already at 200-round intervals (0, 200, 400, ...)
            x_fid = np.arange(0, total_rounds, 200)
            y = scores[:len(x_fid)]
            x_plot = x_fid
        else:
            # Regular scores are computed every round, subsample them
            x = np.arange(100, 7000, 500)  # Dynamically adjust x based on scores length
            y = scores[100:7000:500]  # Ensure y matches x length
            x_plot = x

        st = style.get(method, dict())
        label = display_names.get(method, method)
        if method.startswith('Mixture Greedy EG steps_'):
            # Compact labels for sweep modes: Mixture Greedy EG steps_X_eta_Y
            suffix = method[len('Mixture Greedy EG steps_'):]
            if '_eta_' in suffix:
                step_part, eta_part = suffix.split('_eta_', 1)
                eta_human = eta_part.replace('p', '.')
                label = f'EG steps={step_part}, eta={eta_human}'
        ax.plot(
            x_plot, y,
            label=label,
            linewidth=3,
            markersize=9,
            **st
        )

    ax.set_xlabel("Step", fontsize=22)
    if metric_name == 'fid':
        ax.set_ylabel("FID", fontsize=22)
    else:
        ax.set_ylabel(config.QUADRATIC_METRIC.upper(), fontsize=22)
    ax.tick_params(axis='both', labelsize=18)

    # Ensure y-axis uses plain numbers (disable offset/scientific formatting)
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.get_major_formatter().set_scientific(False)

    handles, labels = ax.get_legend_handles_labels()
    # Keep the plotting area readable when many methods are shown.
    legend_cols = 1 if len(labels) <= 4 else 2 if len(labels) <= 10 else 3
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=legend_cols,
        fontsize=10,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    # Save the figure
    out_path = os.path.join(path, f'{metric_name}_{sigma}_3.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Plot saved to: {out_path}")
    plt.show()


def plot_alpha_l2_norm(path, modes_names, sigma, total_rounds):
    """Plot the L2 norm of (alpha - true_alphas) over rounds.
    
    Args:
        path: Directory containing the alpha_history.npz files
        modes_names: List of method names
        sigma: Sigma value used in the experiment
        total_rounds: Total number of rounds
    """
    methods = [f'{name}_{sigma}' for name in modes_names]
    
    # Parse true_alphas from one of the results files
    true_alphas = None
    for name in modes_names:
        results_file = os.path.join(path, f'results_{name}_{sigma}.txt')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                lines = f.readlines()
                # The last line contains the optimal alphas array
                for i, line in enumerate(lines):
                    if 'optimal alphas' in line.lower():
                        # Next line contains the array
                        alpha_str = lines[i + 1].strip()
                        # Parse the numpy array string, e.g., "[6.83747337e-01 2.97824913e-01 ...]"
                        alpha_str = alpha_str.strip('[]')
                        true_alphas = np.array([float(x) for x in alpha_str.split()])
                        break
            if true_alphas is not None:
                break
    
    if true_alphas is None:
        print("Warning: Could not find true alphas in results files, skipping alpha L2 norm plot.")
        return
    
    style = {
        f'Mixture-UCB_{sigma}':     dict(color='b', linestyle='-',  marker='s'),
        f'Mixture Oracle_{sigma}':  dict(color='r', linestyle='--', marker='s'),
        f'Mixture Greedy_{sigma}':  dict(color='g', linestyle='-',  marker='s'),
        f'Mixture Greedy EG_{sigma}':  dict(color='darkgreen', linestyle='-.', marker='D'),
        f'Mixture-UCB_dl0.2_{sigma}':  dict(color='navy', linestyle='-',  marker='s'),
        f'Mixture-UCB_dl0.6_{sigma}':  dict(color='royalblue', linestyle='-',  marker='s'),
        f'Mixture-UCB_dl0.9_{sigma}':  dict(color='cornflowerblue', linestyle='-',  marker='s'),
    }

    # Display names for legend (without sigma, and ts -> Thompson Sampling)
    display_names = {
        f'Mixture-UCB_{sigma}':     'Mixture UCB',
        f'Mixture Oracle_{sigma}':  'Mixture Oracle',
        f'One-Arm UCB_{sigma}':     'Vanilla UCB',
        f'ts_{sigma}':              'Mixture TS',
        f'Mixture Greedy_{sigma}':  r'$\delta_L = 0$',
        f'Mixture Greedy EG_{sigma}':  'Mixture Greedy EG',
        f'One-Arm TS_{sigma}':     'Vanilla TS',
        f'Mixture-UCB_dl0.2_{sigma}':  r'$\delta_L = 0.2$',
        f'Mixture-UCB_dl0.6_{sigma}':  r'$\delta_L = 0.6$',
        f'Mixture-UCB_dl0.9_{sigma}':  r'$\delta_L = 0.9$',

    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # x-window to plot (subsample for clarity)
    x = np.arange(100, 7000, 500)

    for method in methods:
        fp = os.path.join(path, f'{method}_alpha_history.npz')
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skipping.")
            continue
        
        z = np.load(fp)
        alpha_history = z['alpha_history']  # Shape: (total_rounds, num_arms)
        
        # Compute L2 norm of (alpha - true_alpha) for each round
        l2_norms = np.linalg.norm(alpha_history - true_alphas, axis=1, ord=1)  # L1 norm
        
        # Subsample for plotting
        y = l2_norms[100:4000:500]
        
        st = style.get(method, dict())
        label = display_names.get(method, method)
        ax.plot(
            x, y,
            label=label,
            linewidth=3,
            markersize=9,
            **st
        )

    ax.set_xlabel("Step", fontsize=22)
    ax.set_ylabel(r"$\|\alpha - \alpha^*\|_1$", fontsize=22)
    ax.tick_params(axis='both', labelsize=18)

    handles, labels = ax.get_legend_handles_labels()
    legend_cols = 1 if len(labels) <= 4 else 2 if len(labels) <= 10 else 3
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=legend_cols,
        fontsize=10,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    # Save the figure
    out_path = os.path.join(path, f'alpha_l1_norm_{sigma}.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Alpha L1 norm plot saved to: {out_path}")
    plt.show()


def get_results():
    global dataset, names

    dataset_name = dataset
    
    # Different DELTA_L values used for already-saved UCB baselines
    delta_l_values = [0.6]

    base_save_path = f'/research/d7/rshr/bahar/MAB/Mixture-Greedy/methods/cr/{dataset_name}_inception/{config.TOTAL_ROUNDS}/{config.QUADRATIC_METRIC}'
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)



    with_quality = False
    if config.QUADRATIC_METRIC == 'kid':
        with_quality = True

    offline_evaluator = RKEOfflineEvaluator(names, dataset_name, has_reference=with_quality, config=config)
    offline_evaluator.print_model_scores()
    true_alphas = offline_evaluator.optimal_alphas
    print("Optimal alphas: ", true_alphas, offline_evaluator.optimal_rke)
    true_kernel = offline_evaluator.true_kernel
    true_linears = offline_evaluator.true_linears

    # print(true_alphas)

    # ==================== Run only Mixture Greedy EG sweep ====================
    sweep_root = os.path.join(base_save_path, 'eg_sweep')
    os.makedirs(sweep_root, exist_ok=True)
    eg_mode_names = []

    for eg_steps in EG_STEP_GRID:
        for eg_eta in EG_ETA_GRID:
            config.EG_STEPS = eg_steps
            config.EG_ETA = eg_eta

            eta_str = f"{eg_eta:g}".replace('.', 'p')
            setting_tag = f"steps_{eg_steps}_eta_{eta_str}"
            mode_name = f"Mixture Greedy EG {setting_tag}"
            eg_mode_names.append(mode_name)

            save_path = os.path.join(sweep_root, setting_tag)
            os.makedirs(save_path, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Running {mode_name}")
            print(f"{'='*60}\n")

            mode = 'mixture-greedy-eg'
            scores_of_simulation = {}
            fid_scores_of_simulation = {}

            number_of_simulations = config.NUMBER_OF_SIMULATIONS
            for j in range(number_of_simulations):
                print(f"Simulation {j+1} / {config.NUMBER_OF_SIMULATIONS}")
                online_evaluator = RKEOnlineEvaluator(names, dataset_name, offline_evaluator, mode, use_linear=with_quality, config=config)
                online_evaluator.run_online_evaluation(num_rounds=config.TOTAL_ROUNDS)
                online_evaluator.report_sample_nums()
                scores_of_simulation[j] = online_evaluator.scores
                if len(online_evaluator.fid_scores) > 0:
                    fid_scores_of_simulation[j] = online_evaluator.fid_scores

            vals = [np.asarray(v) for v in scores_of_simulation.values()]
            mean_scores = np.mean(np.stack(vals), axis=0)

            if len(fid_scores_of_simulation) > 0:
                fid_vals = [np.asarray(v) for v in fid_scores_of_simulation.values()]
                mean_fid_scores = np.mean(np.stack(fid_vals), axis=0)
                np.savez(f'{save_path}/{mode_name}_{config.SIGMA}_fid.npz', mean_fid_scores)
                np.savez(f'{base_save_path}/{mode_name}_{config.SIGMA}_fid.npz', mean_fid_scores)

            alphas_of_simulation = online_evaluator.alpha_history

            with open(f"{save_path}/results_{mode_name}_{config.SIGMA}.txt", "w") as f:
                f.write(str(names)+ "\n" + str(online_evaluator.sample_indexes.values())+"\n ================ \n las alphas: \n"+str(alphas_of_simulation[-1])+ "\n ================ \n optimal alphas: \n"+ str(true_alphas))

            # Save under settings folder and base folder for easy comparison plotting.
            np.savez(f'{save_path}/{mode_name}_{config.SIGMA}_scores.npz', mean_scores)
            np.savez(f'{save_path}/{mode_name}_{config.SIGMA}_alpha_history.npz', alpha_history=alphas_of_simulation)
            np.savez(f'{base_save_path}/{mode_name}_{config.SIGMA}_scores.npz', mean_scores)
            np.savez(f'{base_save_path}/{mode_name}_{config.SIGMA}_alpha_history.npz', alpha_history=alphas_of_simulation)
            print(f"Alphas from {mode_name}: ", alphas_of_simulation[-1])

    return base_save_path, eg_mode_names, delta_l_values


if __name__ == "__main__":
    # Define sigma values to test
    sigma_values = [40]
    
    for sigma in sigma_values:
        print(f"\n{'='*60}")
        print(f"Plotting saved results with SIGMA = {sigma}")
        print(f"{'='*60}\n")

        config.SIGMA = sigma
        dataset_name = dataset
        base_path = f'/research/d7/rshr/bahar/MAB/Mixture-Greedy/methods/cr/{dataset_name}_inception/{config.TOTAL_ROUNDS}/{config.QUADRATIC_METRIC}'

        if not os.path.exists(base_path):
            print(f"No saved results directory found: {base_path}")
            continue

        suffix = f'_{sigma}_scores.npz'
        saved_modes = [
            fname[:-len(suffix)]
            for fname in os.listdir(base_path)
            if fname.endswith(suffix)
        ]

        if not saved_modes:
            print(f"No saved score files found for SIGMA={sigma} in {base_path}")
            continue

        # Focus the plot on: dl=0.6 baseline + EG variants with different steps.
        eta_for_step_comparison = 0.0001
        eta_str = f"{eta_for_step_comparison:g}".replace('.', 'p')

        baseline_modes = ['Mixture Oracle', 'Mixture Greedy', 'Mixture-UCB_dl0.6']
        eg_step_modes = [
            f'Mixture Greedy EG steps_{step}_eta_{eta_str}'
            for step in EG_STEP_GRID
        ]

        target_modes = baseline_modes + eg_step_modes
        modes_names = [m for m in target_modes if m in saved_modes]

        if not modes_names:
            print("No matching modes found for step comparison.")
            print(f"Expected eta tag: {eta_str}; available mode count: {len(saved_modes)}")
            continue

        print(f"Found {len(modes_names)} saved modes to plot.")
        total_rounds = config.TOTAL_ROUNDS
        plot_results(base_path, modes_names, sigma, total_rounds, metric_name='scores')
        
        print(f"\nCompleted plotting for SIGMA = {sigma}\n")
