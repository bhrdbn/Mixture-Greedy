import numpy as np
import torch
import os
import time

from mixture_greedy.config import ConfigVNE
from mixture_greedy.data.features import DEFAULT_DATA_ROOT
from mixture_greedy.metrics.vne import compute_mixture_vne

fixed_omega = None

def cov_rff2(x, feature_dim, std, presign_omeaga=None):
    global fixed_omega
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if fixed_omega is None:
        # Initialize fixed omega only once
        fixed_omega = torch.randn((x_dim, feature_dim), device=x.device, dtype=x.dtype) * (1 / std)

    omegas = fixed_omega if presign_omeaga is None else presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]

    return batched_rff


default_config = ConfigVNE()


class VNEOnlineEvaluator:
    """Online model-mixture evaluator that maximizes Von Neumann entropy."""

    def __init__(self, model_names, dataset_name, offline_evaluator=None,
                 mode='mixture-greedy', config=None, oracle_alphas=None):
        if config is None:
            config = default_config
        self.config = config
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.mode = mode
        self.model2idx = {name: i for i, name in enumerate(model_names)}
        self.OfflineEvaluator = offline_evaluator
        self.oracle_alphas = oracle_alphas
        self.number_of_arms = len(model_names)

        self.datasets = self._load_all_datasets()
        self.sample_indexes = {model: 0 for model in model_names}

        self.samples = {model: [] for model in model_names}
        self.all_features = []
        self.sample_to_arm = []

        self.current_round = 0
        self.scores = []
        self.vne_scores = []
        self.alpha_history = []

        self.alphas = np.ones(self.number_of_arms, dtype=np.float64) / self.number_of_arms

    def _load_dataset(self, model_name):
        print(f"--- Loading model: {model_name}")
        dataset_path = self._get_dataset_path(model_name)
        dataset = np.load(dataset_path)[f'{self.config.feature_extractor}_features']
        shuffled_indices = np.random.permutation(dataset.shape[0])
        dataset = dataset[shuffled_indices]
        return dataset
    
    def _load_all_datasets(self):
        return {model: self._load_dataset(model) for model in self.model_names}

    def _get_dataset_path(self, model_name):
        if self.dataset_name in ['imagenet', 'ffhq', 'ffhq256', 'cifar10', 'lsun', 'toy', 
                                 'ffhq_truncated', 'quality_ffhq', 'afhq_truncated', 'FFHQ', 
                                 'FFHQ256', 'Imagenet256']:
            return os.path.join(
                DEFAULT_DATA_ROOT,
                self.dataset_name,
                'features',
                'CLIP',
                f'{model_name}.npz',
            )
        elif self.dataset_name == 'FFHQ_truncated':
            return os.path.join(f'/research/d7/rshr/bahar/data/datasets/FFHQ/dino', f'{model_name}.npz')

        elif self.dataset_name == 't2i_cluster':
            return os.path.join('..', f'{model_name}/{cluster_name}.npz')
        elif self.dataset_name == 't2i_coco':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 't2t':
            return os.path.join('../', f'{model_name}_features.npz')
        elif self.dataset_name == 'imagewoof':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 'styles':
            return os.path.join('../', f'{model_name}.npz')
        elif self.dataset_name == 't2i_dog':
            return os.path.join('/research/d7/rshr/bahar/data/sdxl', f'{model_name}_features.npz')
        elif self.dataset_name == 't2i_birds':
            return os.path.join('/research/d7/rshr/bahar/MAB/red_cartoony_birds/embeddings', f'steps_{10}.npz')
        elif self.dataset_name == 'red_birds':
            return os.path.join('/research/d7/rshr/bahar/red_birds/embeddings', f'{model_name}.npz')
        elif self.dataset_name == 't2i_cars':
            return os.path.join('/research/d7/rshr/bahar/data/sdxl', f'{model_name}_cars.npz')
        elif self.dataset_name in ['city', 'celeb', 'city_s', 'city_us']:
            return os.path.join(f'/research/d7/rshr/bahar/data/text_data/{self.dataset_name}/', f'{model_name}_{self.dataset_name}_sbert_features.npz')

    def _get_samples(self, model_name, num_samples):
        """Get samples from a model's dataset."""
        dataset = self.datasets[model_name]
        n = len(dataset)

        if self.sample_indexes[model_name] + num_samples > n:
            # Sample randomly with replacement
            returning_sample = dataset[np.random.randint(0, n, size=num_samples)]
        else:
            # Sequential sampling
            start_idx = self.sample_indexes[model_name]
            returning_sample = dataset[start_idx:start_idx + num_samples]
        
        self.sample_indexes[model_name] += num_samples
        return returning_sample

    def _add_sample(self, model_name, feature):
        """Add a new sample."""
        self.samples[model_name].append(feature)
        self.all_features.append(feature)
        self.sample_to_arm.append(self.model2idx[model_name])

    def _compute_per_arm_vendi_scores(self):
        """Compute current per-arm Vendi scores from collected samples.

        Returns:
            List[float]: vendi score (exp(VNE)) per arm, or -inf if insufficient samples for that arm.
        """
        vendi_scores = []
        eps = self.config.VNE_EPS
        for model_name in self.model_names:
            rho = self._compute_arm_rho(model_name, eps=eps)
            if rho is None:
                vendi_scores.append(-np.inf)
                continue
            eigenvalues = torch.linalg.eigvalsh(rho)
            eigenvalues = torch.clamp(eigenvalues, min=eps)
            eigenvalues = eigenvalues / eigenvalues.sum()
            vne = -torch.sum(eigenvalues * torch.log(eigenvalues + eps)).item()
            vendi = np.exp(vne)
            vendi_scores.append(vendi)
        return vendi_scores

    def _select_one_arm_greedy(self):
        """Select arm index with highest Vendi score so far (one-arm greedy).

        Exploration schedule: during initial phase (same as initial_alphas counts)
        cycle through arms to collect samples. After that, pick arm with highest
        per-arm vendi score computed from current samples.
        """
        m = self.number_of_arms
        # Initial exploration: cycle through arms
        if self.current_round < self.config.INITIAL_SAMPLE_COUNT * m:
            return int(self.current_round % m)

        vendi_scores = self._compute_per_arm_vendi_scores()
        # If all are -inf (no arm has enough samples), fall back to uniform random
        if all([np.isneginf(v) for v in vendi_scores]):
            return int(self.current_round % m)

        best_idx = int(np.nanargmax(np.array(vendi_scores)))
        return best_idx

    def _select_one_arm_epsilon_greedy(self, epsilon_override=None):
        """Select arm using epsilon-greedy on per-arm Vendi scores.

        Args:
            epsilon_override: optional float to override config epsilon for this step
        Returns:
            int: index of selected arm
        """
        m = self.number_of_arms
        # Initial exploration: cycle through arms
        if self.current_round < self.config.INITIAL_SAMPLE_COUNT * m:
            return int(self.current_round % m)

        base_eps = self.config.EPSILON if epsilon_override is None else epsilon_override
        if self.config.EPSILON_DECAY and self.config.EPSILON_DECAY > 0:
            eps = base_eps * np.exp(-self.config.EPSILON_DECAY * self.current_round)
        else:
            eps = base_eps

        # With probability eps, choose uniformly at random
        if np.random.rand() < eps:
            return int(np.random.randint(0, m))

        # Otherwise choose best vendi arm (reuse greedy logic)
        vendi_scores = self._compute_per_arm_vendi_scores()
        if all([np.isneginf(v) for v in vendi_scores]):
            return int(self.current_round % m)
        best_idx = int(np.nanargmax(np.array(vendi_scores)))
        return best_idx
    
    def _compute_arm_rho(self, model_name, eps=1e-10):
        """Compute normalized density matrix rho_i for an arm using covariance.
        
        If kernel_type is 'rff', features are transformed using RFF before computing covariance.
        Otherwise, cosine normalization is applied.
        
        Args:
            model_name: Name of the model/arm
            eps: Numerical stability epsilon
        
        Returns:
            rho_i: Normalized density matrix (d x d)
        """
        features = self.samples[model_name]
        if len(features) < 2:
            return None

        X = torch.tensor(np.array(features), dtype=torch.float64)  # N x d

        if self.config.kernel_type == 'rff':
            X = cov_rff2(X, self.config.rff_n, self.config.rff_sigma)  # Apply RFF transformation
        elif self.config.kernel_type == 'cosine':
            # Normalize features to unit length (cosine similarity)
            X = X / (torch.norm(X, dim=1, keepdim=True) + eps)
        else:
            raise ValueError(f"Unsupported kernel_type: {self.config.kernel_type}")

        N = X.shape[0]

        # Covariance: Ci = (1/(N-1)) * X^T @ X
        Ci = (1.0 / (N - 1)) * (X.T @ X)  # d x d

        # Normalize: rho_i = Ci / Tr(Ci)
        trace_Ci = torch.trace(Ci)
        if trace_Ci < eps:
            return None

        rho_i = Ci / trace_Ci
        return rho_i
    
    def _get_all_rhos(self):
        """Get normalized density matrices for all arms.
        
        If kernel_type is 'rff', features are transformed using RFF before computing covariance.
        Otherwise, cosine normalization is applied.
        
        Returns:
            List of rho_i matrices (one per arm), or None if any arm has insufficient samples
        """
        rho_list = []
        for model_name in self.model_names:
            rho_i = self._compute_arm_rho(model_name)
            if rho_i is None:
                return None
            rho_list.append(rho_i)
        return rho_list

    def _get_arm_indices(self):
        """Get indices of samples belonging to each arm."""
        arm_indices = [[] for _ in range(self.number_of_arms)]
        for idx, arm in enumerate(self.sample_to_arm):
            arm_indices[arm].append(idx)
        return arm_indices

    def _get_sample_sizes(self):
        """Get number of samples for each arm."""
        return [len(self.samples[model]) for model in self.model_names]

    def report_sample_nums(self):
        for name in self.model_names:
            print(f"model: {name}, num: {len(self.samples[name])}")

    def initial_alphas(self, round_):
        """Get initial alpha values during exploration phase."""
        if self.mode == 'one-arm-oracle':
            model_idx = self.OfflineEvaluator.optimal_model
            alphas = np.zeros(self.number_of_arms)
            alphas[model_idx] = 1
        elif self.mode == 'mixture-oracle':
            if self.oracle_alphas is not None:
                alphas = self.oracle_alphas
            elif self.OfflineEvaluator is not None:
                alphas = self.OfflineEvaluator.optimal_alphas
            else:
                raise ValueError("mixture-oracle mode requires either oracle_alphas or OfflineEvaluator with optimal_alphas")
        elif self.mode in ['mixture-ucb', 'one-arm-ucb', 'mixture-greedy', 'one-arm-greedy', 'one-arm-eps-greedy', 'one-arm-epsilon-greedy']:
            model_idx = round_ % self.number_of_arms
            alphas = np.zeros(self.number_of_arms)
            alphas[model_idx] = 1
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        return alphas

    def run_online_evaluation(self, num_rounds, num_samples=None):
        """Run the online evaluation loop."""
        if num_samples is None:
            num_samples = self.config.MINI_BATCH
        
        print_interval = max(1, min(num_rounds // 20, 100))
        
        for round_ in range(num_rounds):
            self.current_round = round_
            
            # Select alpha using EGD with autograd
            alphas = self._select_alpha_egd(round_)
            self.alpha_history.append(alphas.copy())

            if self.mode == 'one-arm-greedy':
                chosen_idx = self._select_one_arm_greedy()
                chosen_arm = self.model_names[chosen_idx]
            elif self.mode in ['one-arm-eps-greedy', 'one-arm-epsilon-greedy']:
                chosen_idx = self._select_one_arm_epsilon_greedy()
                chosen_arm = self.model_names[chosen_idx]
            else:
                chosen_arm = np.random.choice(self.model_names, size=1, p=alphas)[0]

            new_samples = self._get_samples(chosen_arm, num_samples)
            for sample in new_samples:
                self._add_sample(chosen_arm, sample)

            vne = self._compute_current_vne()
            vendi_score = np.exp(vne) if not np.isnan(vne) else float('nan')
            self.vne_scores.append(vendi_score)
            
            if round_ % print_interval == 0 or round_ == num_rounds - 1:
                alpha_str = ', '.join([f'{a:.3f}' for a in alphas])
                sample_str = ', '.join([f'{self.model_names[i]}:{len(self.samples[self.model_names[i]])}' 
                                       for i in range(len(self.model_names))])
                print(f"Round {round_:5d}/{num_rounds} | Alpha: [{alpha_str}] | Samples: [{sample_str}] | Vendi Score: {vendi_score:.4f}")

    def _select_alpha_egd(self, round_):
        """Select mixture weights for the current round."""
        if self.mode == 'mixture-oracle':
            return self._mixture_oracle_alphas()

        if round_ < self.config.INITIAL_SAMPLE_COUNT * self.number_of_arms:
            return self.initial_alphas(round_)

        if len(self.all_features) < self.config.MIN_VNE_SAMPLES:
            return self._uniform_alphas()

        alpha0 = self._sample_proportions()
        eta = self.config.EG_ETA
        if self.config.EG_SQRT_DECAY:
            eta = eta / np.sqrt(max(round_, 1))

        alpha_current = alpha0.copy()
        for _ in range(self.config.EG_STEPS_PER_ROUND):
            gradient = self._compute_vne_gradient(alpha_current)
            alpha_current = self._exponentiated_gradient_step(
                alpha_current,
                gradient,
                eta,
            )

        self.alphas = alpha_current
        return alpha_current

    def _uniform_alphas(self):
        return np.ones(self.number_of_arms, dtype=np.float64) / self.number_of_arms

    def _mixture_oracle_alphas(self):
        if self.oracle_alphas is not None:
            return np.array(self.oracle_alphas, dtype=np.float64)
        if self.OfflineEvaluator is not None:
            return np.array(self.OfflineEvaluator.optimal_alphas, dtype=np.float64)
        raise ValueError(
            "mixture-oracle mode requires either oracle_alphas or "
            "OfflineEvaluator with optimal_alphas"
        )

    def _sample_proportions(self):
        sample_sizes = np.array(self._get_sample_sizes(), dtype=np.float64)
        total = sample_sizes.sum()
        if total > 0:
            alpha0 = sample_sizes / total
        else:
            alpha0 = self._uniform_alphas()
        alpha0 = np.clip(alpha0, 1e-12, 1.0)
        return alpha0 / alpha0.sum()

    def _exponentiated_gradient_step(self, alphas, gradient, eta):
        """Apply one entropy-maximizing exponentiated-gradient step."""
        eps = self.config.VNE_EPS
        if np.std(gradient) > eps:
            gradient = (gradient - gradient.mean()) / (gradient.std() + eps)

        log_weights = np.log(alphas + 1e-12) + eta * gradient
        log_weights = log_weights - log_weights.max()
        updated_alphas = np.exp(log_weights)
        updated_alphas = updated_alphas / updated_alphas.sum()

        alpha_floor = self.config.ALPHA_FLOOR
        if alpha_floor > 0:
            updated_alphas = np.maximum(updated_alphas, alpha_floor)
            updated_alphas = updated_alphas / updated_alphas.sum()
        return updated_alphas

    def _compute_vne_gradient(self, alpha0):
        """
        Compute gradient of VNE with respect to alpha using torch.autograd.
        
        VNE(alpha) = -tr(rho_mixture * log(rho_mixture))
        where rho_mixture = sum_i (alpha_i * rho_i)
        and rho_i = Ci / Tr(Ci) with Ci = (1/(N-1)) * Xi^T @ Xi
        """
        eps = self.config.VNE_EPS
        
        # Get density matrices for all arms
        rho_list = self._get_all_rhos()
        if rho_list is None:
            # Not enough samples, return uniform gradient
            return np.zeros(self.number_of_arms)
        
        # Create alpha tensor with gradient tracking
        alpha_torch = torch.tensor(alpha0, dtype=torch.float64, requires_grad=True)
        
        # Compute VNE as function of alpha
        vne = compute_mixture_vne(rho_list, alpha_torch, eps=eps)
        
        # Backward pass to compute gradients
        vne.backward()
        
        # Get gradient
        grad = alpha_torch.grad.detach().numpy()
        
        return grad

    def _compute_current_vne(self):
        """Compute VNE of all collected samples using cosine similarity kernel.
        
        This computes the actual sample-level VNE (not alpha-weighted) for monitoring.
        Uses cosine similarity: K[i,j] = (x_i · x_j) / (||x_i|| * ||x_j||)
        For speed, subsample at most 2000 samples if we have more.
        """
        N = len(self.all_features)
        if N < self.config.MIN_VNE_SAMPLES:
            return float('nan')
        
        eps = self.config.VNE_EPS
        max_samples = 2000
        
        # Subsample if needed
        if N <= max_samples:
            features = np.array(self.all_features)
        else:
            indices = np.random.choice(N, size=max_samples, replace=False)
            features = np.array([self.all_features[i] for i in indices])
        
        # Compute cosine similarity kernel matrix
        X = torch.tensor(features, dtype=torch.float64)
        
        # Normalize each row to unit length
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + eps)
        
        # Cosine similarity: K = X_norm @ X_norm^T
        K = X_norm @ X_norm.T
        
        # Standard VNE: rho = K / tr(K), then VNE = -sum(lambda * log(lambda))
        trace_K = torch.trace(K)
        if trace_K < eps:
            return 0.0
        
        rho = K / trace_K
        rho = 0.5 * (rho + rho.T)  # Symmetrize for numerical stability
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # VNE = -sum(lambda * log(lambda))
        vne = -torch.sum(eigenvalues * torch.log(eigenvalues + eps))
        
        return vne.item()


def plot_vne_results(vne_scores=None, load_path=None, save_path=None, title='Vendi Score over Rounds',
                     annotate_interval=200, per_model_vne=None):
    """Plot Vendi Scores over rounds with annotations."""
    import matplotlib.pyplot as plt
    
    if load_path is not None:
        data = np.load(load_path)
        vne_scores = data['vendi_scores']
        print(f"Loaded Vendi scores from: {load_path}")
        print(f"  Shape: {vne_scores.shape}, Rounds: {len(vne_scores)}")
    
    if vne_scores is None:
        raise ValueError("Either vne_scores or load_path must be provided")
    
    vne_scores = np.array(vne_scores)
    valid_mask = ~np.isnan(vne_scores)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    
    rounds = np.arange(len(vne_scores))
    ax.plot(rounds[valid_mask], vne_scores[valid_mask], linewidth=2, color='green', label='Vendi-EG')
    
    if annotate_interval > 0:
        annotation_rounds = list(range(0, len(vne_scores), annotate_interval))
        last_round = len(vne_scores) - 1
        if last_round not in annotation_rounds:
            if annotation_rounds and (last_round - annotation_rounds[-1]) > annotate_interval // 2:
                annotation_rounds.append(last_round)
        
        for r in annotation_rounds:
            if valid_mask[r]:
                vne_val = vne_scores[r]
                is_last = (r == annotation_rounds[-1])
                
                marker_size = 80 if is_last else 50
                ax.scatter([r], [vne_val], color='darkgreen', s=marker_size, zorder=5)
                
                fontsize = 11 if is_last else 9
                fontweight = 'bold' if is_last else 'normal'
                arrow_lw = 1.5 if is_last else 0.5
                ax.annotate(f'{vne_val:.2f}', 
                           xy=(r, vne_val), 
                           xytext=(10, 15),
                           textcoords='offset points',
                           fontsize=fontsize,
                           fontweight=fontweight,
                           color='darkgreen',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9 if is_last else 0.7),
                           arrowprops=dict(arrowstyle='->', color='darkgreen' if is_last else 'gray', lw=arrow_lw))
    
    if per_model_vne is not None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(per_model_vne)))
        for i, (model_name, vne_val) in enumerate(per_model_vne.items()):
            ax.axhline(y=vne_val, color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'{model_name}: {vne_val:.2f}')
    
    ax.set_xlabel("Round", fontsize=18)
    ax.set_ylabel("Vendi Score", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_alpha_history(alpha_history, model_names, save_path=None, title='Alpha Weights over Rounds'):
    """Plot alpha weights for each model over rounds."""
    import matplotlib.pyplot as plt
    
    alpha_history = np.array(alpha_history)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    rounds = np.arange(len(alpha_history))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for i, name in enumerate(model_names):
        ax.plot(rounds, alpha_history[:, i], linewidth=2, color=colors[i], label=name)
    
    ax.set_xlabel("Round", fontsize=18)
    ax.set_ylabel("Alpha Weight", fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


def compute_per_model_vne(model_names, dataset_name, num_samples=10000, 
                          feature_extractor='dino'):
    """Compute Vendi Score for each model individually using cosine similarity.
    
    Args:
        model_names: List of model names
        dataset_name: Dataset name for loading model features
        num_samples: Number of samples to use from each model
        feature_extractor: Feature extractor key
    
    Returns:
        Dictionary mapping model name to Vendi Score (exp(VNE))
    """
    print(f"\n{'='*60}")
    print(f"Computing Per-Model Vendi Scores (Cosine Similarity)")
    print(f"Models: {model_names}")
    print(f"Samples per model: {num_samples}")
    print(f"{'='*60}\n")
    
    vne_scores = {}
    eps = 1e-10
    
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        
        if dataset_name in ['imagenet', 'ffhq', 'ffhq256', 'cifar10', 'lsun', 'toy', 
                           'ffhq_truncated', 'quality_ffhq', 'afhq_truncated', 'FFHQ', 
                           'FFHQ256', 'Imagenet256']:
            # model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')

            model_path = os.path.join(f'/research/d7/rshr/bahar/data/dgm/{dataset_name}/features/CLIP', f'{model_name}.npz')
        if dataset_name == 't2i_cars':
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/sdxl/', f'{model_name}_cars.npz')
        if dataset_name == 't2i_dog':
            model_path = os.path.join('/research/d7/rshr/bahar/data/sdxl', f'{model_name}_features.npz')
        elif dataset_name in ['city', 'celeb', 'city_s', 'city_us']:
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/text_data/{dataset_name}/', f'{model_name}_{dataset_name}_sbert_features.npz')
        elif dataset_name == 'red_birds':
            model_path = os.path.join('/research/d7/rshr/bahar/red_birds/embeddings', f'{model_name}.npz')

        elif dataset_name == 'FFHQ_truncated':
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/FFHQ/dino', f'{model_name}.npz')
        model_data = np.load(model_path)[f'{feature_extractor}_features']
        print(f"  Loaded {model_name}: {model_data.shape}")
        
        n = min(num_samples, len(model_data))
        if n < num_samples:
            print(f"  Warning: Only {n} samples available (requested {num_samples})")
        
        samples = model_data[:n]
        
        # Compute cosine similarity kernel matrix
        X = torch.tensor(samples, dtype=torch.float64)
        
        # Normalize each row to unit length
        X_norm = X / (torch.norm(X, dim=1, keepdim=True) + eps)
        
        # Cosine similarity: K = X_norm @ X_norm^T
        K = X_norm @ X_norm.T
        
        # Standard VNE: rho = K / tr(K)
        trace_K = torch.trace(K)
        rho = K / trace_K
        rho = 0.5 * (rho + rho.T)
        
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        vne = -torch.sum(eigenvalues * torch.log(eigenvalues + eps))
        vendi_score = torch.exp(vne).item()
        vne_scores[model_name] = vendi_score
        
        print(f"  Vendi({model_name}): {vendi_score:.4f}")
    
    print(f"\n{'='*60}")
    print("Summary: Per-Model Vendi Scores")
    print(f"{'='*60}")
    for name, vs in sorted(vne_scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {vs:.4f}")
    print(f"{'='*60}\n")
    
    return vne_scores


def compute_vne_from_alpha_history(alpha_history, model_names, dataset_name, 
                                    num_samples=10000, feature_extractor='dino',
                                    save_path=None, plot=True,
                                    mixture_oracle_alphas=None, one_arm_oracle_alphas=None,
                                    feature_space='cosine', rff_n=256, rff_sigma=40, rff_seed=42):
    """Compute VNE scores for each alpha in the history using precomputed rho matrices.
    feature_space: 'cosine' (default, normalized features) or 'rff' (random Fourier features, unnormalized)
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print(f"Computing VNE from Alpha History")
    print(f"Feature space: {feature_space}")
    print(f"Models: {model_names}")
    print(f"Samples per model: {num_samples}")
    print(f"Number of alphas: {len(alpha_history)}")
    print(f"{'='*60}\n")
    
    eps = 1e-10
    alpha_history = np.array(alpha_history)
    
    # Helper function to get dataset path
    def get_path(model_name):
        if dataset_name in ['imagenet', 'ffhq', 'ffhq256', 'cifar10', 'lsun', 'toy', 
                           'ffhq_truncated', 'quality_ffhq', 'afhq_truncated', 'FFHQ', 
                           'FFHQ256', 'Imagenet256']:
            return os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')
        elif dataset_name == 't2i_cars':
            return os.path.join(f'/research/d7/rshr/bahar/data/sdxl/', f'{model_name}_cars.npz')
        elif dataset_name == 't2i_dog':
            return os.path.join('/research/d7/rshr/bahar/data/sdxl', f'{model_name}_features.npz')
        elif dataset_name == 'red_birds':
            return os.path.join('/research/d7/rshr/bahar/red_birds/embeddings', f'{model_name}.npz')
        elif dataset_name in ['city', 'celeb', 'city_s', 'city_us']:
            return os.path.join(f'/research/d7/rshr/bahar/data/text_data/{dataset_name}/', f'{model_name}_{dataset_name}_sbert_features.npz')
        elif dataset_name == 'FFHQ_truncated':
            return os.path.join(f'/research/d7/rshr/bahar/data/datasets/FFHQ/dino', f'{model_name}.npz')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load samples and compute rho_i for each model
    print("Loading samples and computing rho matrices...")
    rho_list = []
    for model_name in model_names:
        print(f"  Loading {model_name}...")
        model_path = get_path(model_name)
        model_data = np.load(model_path)[f'{feature_extractor}_features']
        
        n = min(num_samples, len(model_data))
        if n < num_samples:
            print(f"    Warning: Only {n} samples available (requested {num_samples})")
        
        samples = model_data[:n]
        X = torch.tensor(samples, dtype=torch.float64)
        N = X.shape[0]
        
        if feature_space == 'cosine':
            # Normalize features to unit length (cosine similarity)
            X = X / (torch.norm(X, dim=1, keepdim=True) + eps)
        elif feature_space == 'rff':
            # Apply RFF transform (no normalization)
            X_np = X.cpu().numpy()
            X_rff = rff_transform(X_np, n_rff=rff_n, sigma=rff_sigma, seed=rff_seed)
            X = torch.tensor(X_rff, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown feature_space: {feature_space}")
        
        # Covariance: Ci = (1/(N-1)) * X^T @ X
        Ci = (1.0 / (N - 1)) * (X.T @ X)  # d x d
        
        # Normalize: rho_i = Ci / Tr(Ci)
        trace_Ci = torch.trace(Ci)
        rho_i = Ci / trace_Ci
        rho_list.append(rho_i)
        print(f"    rho shape: {rho_i.shape}, trace: {trace_Ci.item():.4f}")
    
    print(f"\nComputing VNE for {len(alpha_history)} alphas...")
    vne_scores = []
    vendi_scores = []
    
    print_interval = max(1, len(alpha_history) // 20)
    
    for i, alpha in enumerate(alpha_history):
        alpha_torch = torch.tensor(alpha, dtype=torch.float64)
        
        # Compute mixture VNE
        vne = compute_mixture_vne(rho_list, alpha_torch, eps=eps)
        vne_val = vne.item()
        vendi_val = np.exp(vne_val)
        
        vne_scores.append(vne_val)
        vendi_scores.append(vendi_val)
        
        if i % print_interval == 0 or i == len(alpha_history) - 1:
            alpha_str = ', '.join([f'{a:.3f}' for a in alpha])
            print(f"  Round {i:5d}/{len(alpha_history)} | VNE: {vne_val:.4f} | Vendi: {vendi_val:.4f} | Alpha: [{alpha_str}]")
    
    vne_scores = np.array(vne_scores)
    vendi_scores = np.array(vendi_scores)
    
    # Compute baseline VNE values using the same rho matrices
    mixture_oracle_vne = None
    mixture_oracle_vendi = None
    one_arm_oracle_vne = None
    one_arm_oracle_vendi = None
    
    if mixture_oracle_alphas is not None:
        alpha_t = torch.tensor(mixture_oracle_alphas, dtype=torch.float64)
        vne = compute_mixture_vne(rho_list, alpha_t, eps=eps)
        mixture_oracle_vne = vne.item()
        mixture_oracle_vendi = np.exp(mixture_oracle_vne)
        print(f"\nMixture Oracle VNE: {mixture_oracle_vne:.4f}, Vendi: {mixture_oracle_vendi:.4f}")
        print(f"  Alpha: {mixture_oracle_alphas}")
    
    if one_arm_oracle_alphas is not None:
        alpha_t = torch.tensor(one_arm_oracle_alphas, dtype=torch.float64)
        vne = compute_mixture_vne(rho_list, alpha_t, eps=eps)
        one_arm_oracle_vne = vne.item()
        one_arm_oracle_vendi = np.exp(one_arm_oracle_vne)
        print(f"\nOne-Arm Oracle VNE: {one_arm_oracle_vne:.4f}, Vendi: {one_arm_oracle_vendi:.4f}")
        print(f"  Alpha: {one_arm_oracle_alphas}")
    
    print(f"\n{'='*60}")
    print(f"Completed! Final Vendi Score: {vendi_scores[-1]:.4f}")
    if mixture_oracle_vendi is not None:
        print(f"Mixture Oracle Vendi: {mixture_oracle_vendi:.4f}")
    if one_arm_oracle_vendi is not None:
        print(f"One-Arm Oracle Vendi: {one_arm_oracle_vendi:.4f}")
    print(f"{'='*60}")
    
    if save_path:
        save_dict = {
            'vne_scores': vne_scores, 
            'vendi_scores': vendi_scores, 
            'alpha_history': alpha_history, 
            'model_names': model_names
        }
        if mixture_oracle_vne is not None:
            save_dict['mixture_oracle_vne'] = mixture_oracle_vne
            save_dict['mixture_oracle_vendi'] = mixture_oracle_vendi
            save_dict['mixture_oracle_alphas'] = mixture_oracle_alphas
        if one_arm_oracle_vne is not None:
            save_dict['one_arm_oracle_vne'] = one_arm_oracle_vne
            save_dict['one_arm_oracle_vendi'] = one_arm_oracle_vendi
            save_dict['one_arm_oracle_alphas'] = one_arm_oracle_alphas
        np.savez(save_path, **save_dict)
        print(f"Saved to: {save_path}")
    
    # Plot results
    if plot:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        
        rounds = np.arange(len(vendi_scores))
        ax.plot(rounds, vendi_scores, linewidth=2, color='green', label='VNE-EG')
        
        # Add dashed lines for oracle baselines
        if mixture_oracle_vendi is not None:
            ax.axhline(y=mixture_oracle_vendi, color='blue', linestyle='--', linewidth=2, 
                      label=f'Mixture Oracle: {mixture_oracle_vendi:.2f}')
        
        if one_arm_oracle_vendi is not None:
            ax.axhline(y=one_arm_oracle_vendi, color='red', linestyle='--', linewidth=2,
                      label=f'One-Arm Oracle: {one_arm_oracle_vendi:.2f}')
        
        # Add annotations at key points
        annotate_interval = max(1, len(vendi_scores) // 5)
        annotation_rounds = list(range(0, len(vendi_scores), annotate_interval))
        if len(vendi_scores) - 1 not in annotation_rounds:
            annotation_rounds.append(len(vendi_scores) - 1)
        
        for r in annotation_rounds:
            vendi_val = vendi_scores[r]
            is_last = (r == annotation_rounds[-1])
            marker_size = 80 if is_last else 50
            ax.scatter([r], [vendi_val], color='darkgreen', s=marker_size, zorder=5)
            
            fontsize = 11 if is_last else 9
            fontweight = 'bold' if is_last else 'normal'
            ax.annotate(f'{vendi_val:.2f}', 
                       xy=(r, vendi_val), 
                       xytext=(10, 15),
                       textcoords='offset points',
                       fontsize=fontsize,
                       fontweight=fontweight,
                       color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', 
                                alpha=0.9 if is_last else 0.7),
                       arrowprops=dict(arrowstyle='->', color='darkgreen' if is_last else 'gray', 
                                      lw=1.5 if is_last else 0.5))
        
        ax.set_xlabel("Round", fontsize=18)
        ax.set_ylabel("Vendi Score", fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            plot_path = save_path.replace('.npz', '.pdf') if save_path.endswith('.npz') else f'{save_path}.pdf'
            fig.savefig(plot_path, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.show()
    
    result = {
        'vne_scores': vne_scores,
        'vendi_scores': vendi_scores,
        'alpha_history': alpha_history
    }
    if mixture_oracle_vne is not None:
        result['mixture_oracle_vne'] = mixture_oracle_vne
        result['mixture_oracle_vendi'] = mixture_oracle_vendi
    if one_arm_oracle_vne is not None:
        result['one_arm_oracle_vne'] = one_arm_oracle_vne
        result['one_arm_oracle_vendi'] = one_arm_oracle_vendi
    
    return result

def get_results(config=None, model_names=None, dataset_name=None, 
                save_dir=None, num_simulations=1, mode='mixture-greedy', oracle_alphas=None,
                mixture_oracle_alphas=None, one_arm_oracle_alphas=None):
    """Run VNE-EG online evaluation and plot results."""
    
    if config is None:
        config = ConfigVNE()
    
    if model_names is None:
        model_names = ['arm0', 'arm1']
    
    if dataset_name is None:
        dataset_name = 'Imagenet256'
    
    if save_dir is None:
        if mode == 'mixture-oracle':
            method_name = 'VNE_Oracle'
        elif mode == 'mixture-greedy':
            method_name = 'VNE_EG_10steps'
        elif mode == 'one-arm-greedy':
            method_name = 'VNE_OneArm_Greedy'
        elif mode in ['one-arm-eps-greedy', 'one-arm-epsilon-greedy']:
            method_name = 'VNE_OneArm_EpsilonGreedy'
        if config.kernel_type == 'cosine':
            save_dir = f'/research/d7/rshr/bahar/MAB/Mixture-Greedy/methods/{method_name}/{dataset_name}/cosine/{config.TOTAL_ROUNDS}'
        else:
            save_dir = f'/research/d7/rshr/bahar/MAB/Mixture-Greedy/methods/{method_name}/{dataset_name}/rff_{config.rff_n}/{config.TOTAL_ROUNDS}'

    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    if mode == 'mixture-oracle':
        print(f"Running VNE-Oracle Online Evaluation")
    elif mode == 'mixture-greedy':
        print(f"Running VNE-EG Online Evaluation")
    elif mode == 'one-arm-greedy':
        print(f"Running VNE One-Arm Greedy Online Evaluation")
    elif mode in ['one-arm-eps-greedy', 'one-arm-epsilon-greedy']:
        print(f"Running VNE One-Arm Epsilon-Greedy Online Evaluation")
        print(f"Running VNE-EG Online Evaluation")
    print(f"Models: {model_names}")
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {mode}")
    if oracle_alphas is not None:
        print(f"Oracle Alphas: {oracle_alphas}")
    print(f"Total Rounds: {config.TOTAL_ROUNDS}")
    if mode != 'mixture-oracle':
        print(f"EG Learning Rate (eta): {config.EG_ETA}")
    print(f"Kernel: {config.kernel_type}")
    print(f"Mixture: Covariance-based (Ci = Xi^T @ Xi / (N-1))")
    print(f"{'='*60}\n")
    
    all_vne_scores = []
    all_alpha_histories = []
    sim_times = []
    
    total_start_time = time.time()
    
    for sim in range(num_simulations):
        print(f"\n{'='*60}")
        print(f"Simulation {sim + 1}/{num_simulations}")
        print(f"{'='*60}")
        
        evaluator = VNEOnlineEvaluator(
            model_names=model_names,
            dataset_name=dataset_name,
            offline_evaluator=None,
            mode=mode,
            config=config,
            oracle_alphas=oracle_alphas
        )
        
        print(f"Starting online evaluation...")
        sim_start_time = time.time()
        
        evaluator.run_online_evaluation(num_rounds=config.TOTAL_ROUNDS)
        
        sim_elapsed = time.time() - sim_start_time
        sim_times.append(sim_elapsed)
        
        print(f"\nSimulation {sim + 1} completed in {sim_elapsed:.2f} seconds.")
        evaluator.report_sample_nums()
        
        all_vne_scores.append(evaluator.vne_scores)
        all_alpha_histories.append(evaluator.alpha_history)
    
    total_elapsed = time.time() - total_start_time
    avg_elapsed = total_elapsed / num_simulations
    
    mean_vne_scores = np.nanmean(np.stack(all_vne_scores), axis=0)
    final_alpha_history = all_alpha_histories[-1]
    
    np.savez(f'{save_dir}/vendi_scores.npz', vendi_scores=mean_vne_scores)
    np.savez(f'{save_dir}/alpha_history.npz', alpha_history=final_alpha_history)
    
    with open(f'{save_dir}/config.txt', 'w') as f:
        f.write(f"Models: {model_names}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Mode: {mode}\n")
        if oracle_alphas is not None:
            f.write(f"Oracle Alphas: {oracle_alphas}\n")
        f.write(f"Total Rounds: {config.TOTAL_ROUNDS}\n")
        if mode != 'mixture-oracle':
            f.write(f"EG_ETA: {config.EG_ETA}\n")
            f.write(f"EG_SQRT_DECAY: {config.EG_SQRT_DECAY}\n")
            f.write(f"ALPHA_FLOOR: {config.ALPHA_FLOOR}\n")
            f.write(f"EG_STEPS_PER_ROUND: {config.EG_STEPS_PER_ROUND}\n")
        f.write(f"Kernel: {config.kernel_type}\n")
        f.write(f"Mixture: Covariance-based\n")
        f.write(f"Final alphas: {final_alpha_history[-1]}\n")
        f.write(f"Sample sizes: {evaluator._get_sample_sizes()}\n")
        f.write(f"Simulation times (seconds): {sim_times}\n")
        f.write(f"Total running time (seconds): {total_elapsed:.2f}\n")
        f.write(f"Average running time per simulation (seconds): {avg_elapsed:.2f}\n")
    
    print(f"\n{'='*60}")
    print(f"All simulations completed!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {save_dir}")
    print(f"Final alphas: {final_alpha_history[-1]}")
    print(f"Total running time: {total_elapsed:.2f} seconds")
    print(f"Average running time per simulation: {avg_elapsed:.2f} seconds")
    
    print(f"\nComputing VNE scores from alpha history (10k samples per model)...")
    alpha_vne_results = compute_vne_from_alpha_history(
        alpha_history=final_alpha_history,
        model_names=model_names,
        dataset_name=dataset_name,
        num_samples=10000,
        feature_extractor=config.feature_extractor,
        save_path=f'{save_dir}/alpha_vne_scores.npz',
        plot=True,
        mixture_oracle_alphas=mixture_oracle_alphas,
        one_arm_oracle_alphas=one_arm_oracle_alphas
    )
    
    print(f"\nGenerating plots...")
    plot_vne_results(
        mean_vne_scores, 
        save_path=f'{save_dir}/vendi_plot.pdf',
        title=f'Vendi Score over Rounds ({dataset_name})'
    )
    
    plot_alpha_history(
        final_alpha_history,
        model_names,
        save_path=f'{save_dir}/alpha_plot.pdf',
        title=f'Alpha Weights ({dataset_name})'
    )
    
    return {
        'vendi_scores': mean_vne_scores,
        'alpha_vne_scores': alpha_vne_results['vendi_scores'],
        'alpha_history': final_alpha_history,
        'sample_sizes': evaluator._get_sample_sizes(),
        'model_names': model_names,
        'mixture_oracle_vendi': alpha_vne_results.get('mixture_oracle_vendi'),
        'one_arm_oracle_vendi': alpha_vne_results.get('one_arm_oracle_vendi'),
        'simulation_times': sim_times,
        'total_time_seconds': total_elapsed,
        'avg_time_seconds': avg_elapsed,
    }


if __name__ == "__main__":
    import sys

    from mixture_greedy.__main__ import main

    raise SystemExit(main(sys.argv[1:], default_metric="vne"))