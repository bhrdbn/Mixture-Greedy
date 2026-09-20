import numpy as np
import torch
import os
import time
from mixture_greedy.config import ConfigFID
from mixture_greedy.data.features import DEFAULT_DATA_ROOT
from mixture_greedy.metrics.fid import (
    _fid_value_and_grad_for_alpha,
    _mean_cov_np,
    _mixture_moments_torch,
    _trace_sqrtm_psd,
    fid_from_moments_torch,
)

try:
    from scipy.optimize import minimize, Bounds, LinearConstraint
except Exception:
    minimize = None
    Bounds = None
    LinearConstraint = None

default_config = ConfigFID()


class FIDOnlineEvaluator:
    """Online model-mixture evaluator that minimizes Fréchet distance."""

    def __init__(
        self,
        model_names,
        dataset_name,
        offline_evaluator,
        mode='mixture-greedy',
        use_linear=False,
        config=None,
        optimal_alphas=None,
    ):
        if config is None:
            config = default_config
        self.config = config
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.mode = mode
        self.model2idx = {name: index for index, name in enumerate(model_names)}
        self.OfflineEvaluator = offline_evaluator
        self.optimal_alphas = optimal_alphas  # For mixture-oracle mode
        
        self.real_mu = None
        self.real_cov = None
        if offline_evaluator is not None and hasattr(offline_evaluator, 'real_dataset') and offline_evaluator.real_dataset is not None:
            self.real_mu, self.real_cov = _mean_cov_np(offline_evaluator.real_dataset)
        self.use_linear = use_linear
        if self.use_linear:
            self.linears = {model_name : [] for model_name in model_names}
        self.datasets = self._load_all_datasets()
        
        # For change-point dataset, load both pre and post distributions
        if self.config.CHANGE_POINT:
            self._load_change_point_online_datasets()
        
        # For male/female adaptive experiment
        if self.config.MALE_FEMALE_ADAPTIVE:
            self._load_male_female_online_datasets()
        
        # For cats/birds/animals adaptive experiment
        if self.config.CATS_BIRDS_ADAPTIVE:
            self._load_cats_birds_online_datasets()

        self.samples = {model: [] for model in model_names}
        self.sample_indexes = {model: 0 for model in model_names}
        self.sample_indexes_post = {model: 0 for model in model_names}
        self.current_round = 0
        self.number_of_arms = len(model_names)
        self.kernel = np.zeros((len(self.model_names),len(self.model_names)))
        self.scores = []
        self.fid_scores = []
        self.mixture_fid_scores = []
        self.alpha_history = []
        
        if self.config.TRACK_MIXTURE_FID:
            self._precompute_model_moments(
                num_samples=self.config.MIXTURE_FID_SAMPLES,
                seed=self.config.MIXTURE_FID_SEED,
            )
        

    def _load_change_point_online_datasets(self):
        """Load both pre and post change-point datasets for online sampling."""
        print(f"--- Loading change-point datasets for online evaluation")
        data = np.load(self.config.CHANGE_POINT_PATH)
        
        self.datasets_pre = {}
        self.datasets_post = {}
        
        for model in self.model_names:
            if model == 'arm0':
                pre_arr = data['arm0_pre']
                post_arr = data['arm0_post']
            elif model == 'arm1':
                pre_arr = data['arm1_pre']
                post_arr = data['arm1_post']
            else:
                raise ValueError(f"Unknown arm name for change-point dataset: {model}")
            
            # Shuffle both datasets
            self.datasets_pre[model] = pre_arr[np.random.permutation(len(pre_arr))]
            self.datasets_post[model] = post_arr[np.random.permutation(len(post_arr))]
            print(f"Loaded {model}: pre={len(self.datasets_pre[model])}, post={len(self.datasets_post[model])}")

    def _load_male_female_online_datasets(self):
        """Load cats/dogs datasets for adaptive online sampling.
        cats = male, dogs = female.
        """
        print(f"--- Loading cats/dogs datasets for online evaluation")
        self.male_data = np.load('/research/d7/rshr/bahar/MAB/cats.npz')['dino_features']
        self.female_data = np.load('/research/d7/rshr/bahar/MAB/dogs.npz')['dino_features']
        
        # Shuffle both
        np.random.shuffle(self.male_data)
        np.random.shuffle(self.female_data)
        
        # Track indices for each arm's cats/dogs sampling
        self.male_idx = {'arm0': 0, 'arm1': 0}
        self.female_idx = {'arm0': 0, 'arm1': 0}
        
        print(f"Loaded cats (male): {self.male_data.shape}, dogs (female): {self.female_data.shape}")

    def _load_cats_birds_online_datasets(self):
        """Load cats/birds/animals datasets for adaptive online sampling.
        - arm0: cats (before change), animals (after change)
        - arm1: catsbirds_shuffled (constant)
        """
        print(f"--- Loading cats/birds/animals datasets for online evaluation")
        self.cats_data = np.load(os.path.join(self.config.CATS_BIRDS_DATA_PATH, 'kandinsky_giraffe.npz'))['dino_features']
        self.catsbirds_data = np.load(os.path.join(self.config.CATS_BIRDS_DATA_PATH, 'kandinsky_shark_giraffe_shuffled.npz'))['dino_features']
        self.animals_data = np.load(os.path.join(self.config.CATS_BIRDS_DATA_PATH, 'kandinsky_others.npz'))['dino_features']
        
        # Shuffle all
        np.random.shuffle(self.cats_data)
        np.random.shuffle(self.catsbirds_data)
        np.random.shuffle(self.animals_data)
        
        # Track indices for each arm's sampling
        self.cats_idx = {'arm0': 0}
        self.catsbirds_idx = {'arm1': 0}
        self.animals_idx = {'arm0': 0}
        
        print(f"Loaded cats: {self.cats_data.shape}, catsbirds: {self.catsbirds_data.shape}, animals: {self.animals_data.shape}")

    def _load_dataset(self, model_name):
        print(f"--- Loading model: {model_name}")
        dataset_path = self._get_dataset_path(model_name)
        dataset = np.load(dataset_path)[f'{self.config.feature_extractor}_features']
        shuffled_indices = np.random.permutation(dataset.shape[0])
        dataset = dataset[shuffled_indices]
        return dataset
    
    def _precompute_model_moments(self, num_samples=10000, seed=42):
        """Precompute per-model moments for efficient mixture FID computation.
        
        Uses a fixed seed for reproducibility across runs.
        """
        print(f"--- Precomputing model moments for mixture FID (using {num_samples} samples per model, seed={seed})")
        
        self.model_mus = []
        self.model_covs = []
        
        # Use fixed RNG for reproducibility
        rng = np.random.RandomState(seed)
        
        for model_name in self.model_names:
            # Load fresh copy of dataset (not the shuffled one used for online sampling)
            dataset_path = self._get_dataset_path(model_name)
            dataset = np.load(dataset_path)[f'{self.config.feature_extractor}_features']
            
            n = min(num_samples, len(dataset))
            if n < num_samples:
                print(f"Warning: {model_name} has only {n} samples (requested {num_samples})")
            
            # Use fixed random indices for reproducibility
            idx = rng.choice(len(dataset), size=n, replace=False)
            samples = dataset[idx]
            mu_i, cov_i = _mean_cov_np(samples, eps=self.config.FID_EPS)
            self.model_mus.append(mu_i)
            self.model_covs.append(cov_i)
        
        # Convert to torch tensors for efficient computation
        self.model_mus_torch = torch.tensor(np.stack(self.model_mus), dtype=torch.float64)
        self.model_covs_torch = torch.tensor(np.stack(self.model_covs), dtype=torch.float64)
        
        if self.real_mu is not None:
            self.real_mu_torch = torch.tensor(self.real_mu, dtype=torch.float64)
            self.real_cov_torch = torch.tensor(self.real_cov, dtype=torch.float64)
        
        print(f"--- Model moments precomputed for {len(self.model_names)} models")
    
    def _compute_mixture_fid_for_alphas(self, alphas):
        """Compute mixture FID for given alphas using precomputed model moments."""
        if self.real_mu is None or self.real_cov is None:
            return float('nan')
        
        alpha_t = torch.tensor(alphas, dtype=torch.float64)
        
        # Compute mixture moments
        mu_mix, cov_mix = _mixture_moments_torch(alpha_t, self.model_mus_torch, self.model_covs_torch)
        
        # Compute FID
        fid = fid_from_moments_torch(mu_mix, cov_mix, self.real_mu_torch, self.real_cov_torch, eps=self.config.FID_EPS)
        
        return fid.item()
    
    def report_sample_nums(self):
        for i in self.model_names:
            print(f"model: {i}, num: {len(self.samples[i])}")

    def _compute_per_arm_fid_scores(self):
        """Compute FID score for each arm using collected samples for that arm.

        Returns:
            List[float]: FID per arm (lower is better). If insufficient samples or real moments missing, returns np.inf for that arm.
        """
        if self.real_mu is None or self.real_cov is None:
            return [np.inf for _ in self.model_names]

        per_arm_fids = []
        eps = self.config.FID_EPS
        mu_r = torch.tensor(self.real_mu, dtype=torch.float64)
        cov_r = torch.tensor(self.real_cov, dtype=torch.float64)

        for model in self.model_names:
            X = np.asarray(self.samples[model])
            if X.ndim != 2 or X.shape[0] < max(2, self.config.MIN_FID_SAMPLES_PER_ARM):
                per_arm_fids.append(np.inf)
                continue
            mu_g, cov_g = _mean_cov_np(X, eps=eps)
            mu_g_t = torch.tensor(mu_g, dtype=torch.float64)
            cov_g_t = torch.tensor(cov_g, dtype=torch.float64)
            try:
                fid_val = fid_from_moments_torch(mu_g_t, cov_g_t, mu_r, cov_r, eps=eps).item()
            except Exception:
                fid_val = np.inf
            per_arm_fids.append(fid_val)
        return per_arm_fids

    def _select_one_arm_greedy_fid(self):
        """Select the arm with the lowest FID so far (greedy). Uses initial cycling during exploration."""
        m = self.number_of_arms
        if self.current_round < self.config.INITIAL_SAMPLE_COUNT * m:
            return int(self.current_round % m)
        fids = self._compute_per_arm_fid_scores()
        if all([np.isinf(f) for f in fids]):
            return int(self.current_round % m)
        return int(np.nanargmin(np.array(fids)))

    def _select_one_arm_epsilon_greedy_fid(self, epsilon_override=None):
        """Epsilon-greedy on per-arm FID: with prob eps choose random, else choose best (lowest FID)."""
        m = self.number_of_arms
        if self.current_round < self.config.INITIAL_SAMPLE_COUNT * m:
            return int(self.current_round % m)

        base_eps = self.config.EPSILON if epsilon_override is None else epsilon_override
        if self.config.EPSILON_DECAY and self.config.EPSILON_DECAY > 0:
            eps = base_eps * np.exp(-self.config.EPSILON_DECAY * self.current_round)
        else:
            eps = base_eps

        if np.random.rand() < eps:
            return int(np.random.randint(0, m))

        fids = self._compute_per_arm_fid_scores()
        if all([np.isinf(f) for f in fids]):
            return int(self.current_round % m)
        return int(np.nanargmin(np.array(fids)))

    def _load_all_datasets(self):
        return {model: self._load_dataset(model) for model in self.model_names}



    def _get_dataset_path(self, model_name):
        if self.dataset_name in ['imagenet','ffhq','ffhq256', 'cifar10','lsun','toy','ffhq_truncated','quality_ffhq','afhq_truncated', 'FFHQ', 
                                 'FFHQ256', 'Imagenet256']:
            return os.path.join(DEFAULT_DATA_ROOT, self.dataset_name, 'features', f'{model_name}.npz')
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
            return os.path.join('..//', f'{model_name}.npz')
        elif self.dataset_name == 't2i_birds':
            return os.path.join('/research/d7/rshr/bahar/MAB/red_cartoony_birds/embeddings', f'steps_{10}.npz')
        elif self.dataset_name == 'red_birds':
            return os.path.join('/research/d7/rshr/bahar/red_birds/embeddings', f'{model_name}.npz')
        
        
    def _get_samples(self, model_name, num_samples):
        if self.config.CHANGE_POINT:
            return self._get_samples_change_point(model_name, num_samples)
        if self.config.MALE_FEMALE_ADAPTIVE:
            return self._get_samples_male_female(model_name, num_samples)
        if self.config.CATS_BIRDS_ADAPTIVE:
            return self._get_samples_cats_birds(model_name, num_samples)
        
        dataset = self.datasets[model_name]
        n = len(dataset)
        if self.sample_indexes[model_name]+num_samples > len(dataset):
            print("ERROR: Too many samples needed")
            assert 0==1
        
        returning_sample = dataset[np.random.randint(0, n, size=num_samples)]
        self.sample_indexes[model_name] = self.sample_indexes[model_name] + num_samples
        return returning_sample
    
    def _get_samples_male_female(self, model_name, num_samples):
        """Get samples from MNIST class datasets based on current round and arm.
        class4 replaces male, class1 replaces female.
        
        Arm0 (arm0): Before MALE_FEMALE_CHANGE_ROUND: 100% class1
                     After MALE_FEMALE_CHANGE_ROUND: 50% class1, 50% class4
        Arm1 (arm1): Always 75% class1, 25% class4
        """
        samples = []
        
        for _ in range(num_samples):
            # Determine class4 ratio based on arm and current round
            if model_name == 'arm0':
                if self.current_round < self.config.MALE_FEMALE_CHANGE_ROUND:
                    male_ratio = self.config.ARM1_MALE_RATIO_BEFORE  # 0% class4 (100% class1)
                else:
                    male_ratio = self.config.ARM1_MALE_RATIO_AFTER   # 50% class4
            elif model_name == 'arm1':
                male_ratio = self.config.ARM2_MALE_RATIO  # 25% class4 (75% class1)
            else:
                raise ValueError(f"Unknown arm name for MNIST: {model_name}")
            
            # Sample based on ratio
            if np.random.random() < male_ratio:
                # Sample from class4 (male)
                idx = self.male_idx[model_name]
                if idx >= len(self.male_data):
                    idx = 0  # Reset if exhausted
                    self.male_idx[model_name] = 0
                sample = self.male_data[idx]
                self.male_idx[model_name] += 1
            else:
                # Sample from class1 (female)
                idx = self.female_idx[model_name]
                if idx >= len(self.female_data):
                    idx = 0  # Reset if exhausted
                    self.female_idx[model_name] = 0
                sample = self.female_data[idx]
                self.female_idx[model_name] += 1
            
            samples.append(sample)
        
        self.sample_indexes[model_name] += num_samples
        return np.array(samples)
    
    def _get_samples_change_point(self, model_name, num_samples):
        """Get samples from pre or post dataset based on current round.
        Before T/2: sample from pre-change distribution
        After T/2: sample from post-change distribution (arms swapped)
        """
        change_point = self.config.TOTAL_ROUNDS // 2
        
        if self.current_round < change_point:
            # Pre-change regime
            dataset = self.datasets_pre[model_name]
            idx = self.sample_indexes[model_name]
            if idx + num_samples > len(dataset):
                print(f"ERROR: Too many pre-change samples needed for {model_name}")
                assert 0 == 1
            returning_sample = dataset[idx:idx + num_samples]
            self.sample_indexes[model_name] += num_samples
        else:
            # Post-change regime
            dataset = self.datasets_post[model_name]
            idx = self.sample_indexes_post[model_name]
            if idx + num_samples > len(dataset):
                print(f"ERROR: Too many post-change samples needed for {model_name}")
                assert 0 == 1
            returning_sample = dataset[idx:idx + num_samples]
            self.sample_indexes_post[model_name] += num_samples
        
        return returning_sample

    def _get_samples_cats_birds(self, model_name, num_samples):
        """Get samples from cats/birds/animals datasets based on current round and arm.
        
        Arm0 (arm0): Before CATS_BIRDS_CHANGE_ROUND: 100% cats
                     After CATS_BIRDS_CHANGE_ROUND: 100% animals
        Arm1 (arm1): Always catsbirds_shuffled
        """
        samples = []
        
        for _ in range(num_samples):
            if model_name == 'arm0':
                if self.current_round < self.config.CATS_BIRDS_CHANGE_ROUND:
                    # Before change: sample from cats
                    idx = self.cats_idx['arm0']
                    if idx >= len(self.cats_data):
                        idx = 0  # Reset if exhausted
                        self.cats_idx['arm0'] = 0
                    sample = self.cats_data[idx]
                    self.cats_idx['arm0'] += 1
                else:
                    # After change: sample from animals
                    idx = self.animals_idx['arm0']
                    if idx >= len(self.animals_data):
                        idx = 0  # Reset if exhausted
                        self.animals_idx['arm0'] = 0
                    sample = self.animals_data[idx]
                    self.animals_idx['arm0'] += 1
            elif model_name == 'arm1':
                # Always sample from catsbirds_shuffled
                idx = self.catsbirds_idx['arm1']
                if idx >= len(self.catsbirds_data):
                    idx = 0  # Reset if exhausted
                    self.catsbirds_idx['arm1'] = 0
                sample = self.catsbirds_data[idx]
                self.catsbirds_idx['arm1'] += 1
            else:
                raise ValueError(f"Unknown arm name for cats/birds: {model_name}")
            
            samples.append(sample)
        
        self.sample_indexes[model_name] += num_samples
        return np.array(samples)

    def _extend_samples(self, current_samples, new_samples):
        if len(current_samples)==0:
            return new_samples
        return np.concatenate((current_samples, new_samples), axis=0)


    def initial_alphas(self,round_):
        if self.mode == 'one-arm-oracle':
            model_idx = self.OfflineEvaluator.optimal_model
            alphas = np.array([0 for _ in range(self.number_of_arms)])
            alphas[model_idx] = 1
        elif self.mode == 'mixture-oracle':
            alphas = self.OfflineEvaluator.optimal_alphas
        elif self.mode in ['mixture-ucb','one-arm-ucb','mixture-greedy']:
            model_idx = round_ % self.number_of_arms
            alphas = np.array([0 for _ in range(self.number_of_arms)])
            alphas[model_idx] = 1
        else:
            print('INVALID MODE')
            assert 0==1
        return alphas

    def run_online_evaluation(self, num_rounds, num_samples=None):
        if num_samples is None:
            num_samples = self.config.MINI_BATCH
        
        print_interval = max(1, min(num_rounds // 20, 10))
        
        for round_ in range(0,num_rounds):
            self.current_round = round_

            
            # Log change-point transition
            if self.config.CHANGE_POINT and round_ == self.config.TOTAL_ROUNDS // 2:
                print(f"=== CHANGE POINT at round {round_} ===")
            
            # Log male/female change point
            if self.config.MALE_FEMALE_ADAPTIVE and round_ == self.config.MALE_FEMALE_CHANGE_ROUND:
                print(f"=== MALE/FEMALE CHANGE POINT at round {round_} (UCB) ===")
            
            # Debug: print male ratio at key points
            if self.config.MALE_FEMALE_ADAPTIVE and round_ in [0, 999, 1000, 1001, 1010]:
                print(f"[UCB] round={round_}, arm0 male_ratio={self.config.ARM1_MALE_RATIO_BEFORE if round_ < self.config.MALE_FEMALE_CHANGE_ROUND else self.config.ARM1_MALE_RATIO_AFTER}")
            
            alphas = self._select_best_model_fid_eg(round_)
            self.alpha_history.append(alphas)

            chosen_models = np.random.choice(self.model_names, size=num_samples, p=alphas)
            
            for model_name in chosen_models:
                new_sample = self._get_samples(model_name, 1)  
                self.samples[model_name] = self._extend_samples(self.samples[model_name], new_sample)

            fid = self._cal_fid_sample_based()
            self.fid_scores.append(fid)
            
            if self.config.TRACK_MIXTURE_FID:
                mixture_fid = self._compute_mixture_fid_for_alphas(alphas)
                self.mixture_fid_scores.append(mixture_fid)
            
            if round_ % print_interval == 0 or round_ == num_rounds - 1:
                alpha_str = ', '.join([f'{a:.3f}' for a in alphas])
                sample_str = ', '.join([f'{self.model_names[i]}:{len(self.samples[self.model_names[i]])}' for i in range(len(self.model_names))])
                print(f"Round {round_:5d}/{num_rounds} | FID: {fid:8.2f} | Alpha: [{alpha_str}] | Samples: [{sample_str}]")
            
            
    def _select_best_model_fid_eg(self, round_):
        """Select mixture weights for the current round."""
        if self.mode == 'mixture-oracle':
            return self._mixture_oracle_alphas()

        if (
            round_ < self.config.INITIAL_SAMPLE_COUNT * self.number_of_arms
            and not str(self.mode).startswith('one-arm')
        ):
            return self.initial_alphas(round_)

        if str(self.mode) == 'one-arm-greedy':
            return self._one_hot(self._select_one_arm_greedy_fid())
        if str(self.mode) in ['one-arm-eps-greedy', 'one-arm-epsilon-greedy']:
            return self._one_hot(self._select_one_arm_epsilon_greedy_fid())

        if self.real_mu is None or self.real_cov is None:
            return self._uniform_alphas()

        eps = self.config.FID_EPS
        mus, covs = self._estimated_arm_moments(eps)
        mu_r = torch.tensor(self.real_mu, dtype=torch.float64)
        cov_r = torch.tensor(self.real_cov, dtype=torch.float64)
        alpha0 = self._sample_proportions()

        if str(getattr(self.config, 'OPTIMIZER', 'eg')).lower() == 'scipy':
            alphas = self._solve_alpha_scipy(alpha0, mus, covs, mu_r, cov_r, eps)
        else:
            alphas = self._run_exponentiated_gradient(
                alpha0,
                mus,
                covs,
                mu_r,
                cov_r,
                eps,
            )

        self.alphas = alphas
        return alphas

    def _one_hot(self, arm_index):
        alphas = np.zeros(self.number_of_arms, dtype=np.float64)
        alphas[arm_index] = 1.0
        return alphas

    def _uniform_alphas(self):
        return np.ones(self.number_of_arms, dtype=np.float64) / self.number_of_arms

    def _mixture_oracle_alphas(self):
        if self.optimal_alphas is not None:
            return self.optimal_alphas
        if (
            hasattr(self.OfflineEvaluator, 'optimal_alphas')
            and self.OfflineEvaluator.optimal_alphas is not None
        ):
            return self.OfflineEvaluator.optimal_alphas
        print("Warning: mixture-oracle mode but no optimal_alphas provided, using uniform")
        return self._uniform_alphas()

    def _estimated_arm_moments(self, eps):
        """Estimate one mean and covariance matrix for each sampled arm."""
        feature_dimension = self.real_mu.shape[0]

        mus = []
        covs = []
        minimum_samples = max(2, self.config.MIN_FID_SAMPLES_PER_ARM)
        for model in self.model_names:
            samples = np.asarray(self.samples[model])
            if samples.ndim != 2 or samples.shape[0] < minimum_samples:
                mean = np.zeros((feature_dimension,), dtype=np.float64)
                covariance = np.eye(feature_dimension, dtype=np.float64) * 1e3
            else:
                mean, covariance = _mean_cov_np(samples, eps=eps)
            mus.append(mean)
            covs.append(covariance)

        return (
            torch.tensor(np.stack(mus), dtype=torch.float64),
            torch.tensor(np.stack(covs), dtype=torch.float64),
        )

    def _sample_proportions(self):
        sample_sizes = np.array(self._get_sample_sizes(), dtype=np.float64)
        total_samples = sample_sizes.sum()
        if total_samples > 0:
            alpha0 = sample_sizes / total_samples
        else:
            alpha0 = self._uniform_alphas()
        alpha0 = np.clip(alpha0, 1e-12, 1.0)
        return alpha0 / alpha0.sum()

    def _run_exponentiated_gradient(self, alpha0, mus, covs, mu_r, cov_r, eps):
        """Minimize FID on the simplex with exponentiated-gradient steps."""
        alpha_current = alpha0.copy()
        for eg_step in range(self.config.EG_STEPS):
            alpha = torch.tensor(alpha_current, dtype=torch.float64, requires_grad=True)
            mu_mix, cov_mix = _mixture_moments_torch(alpha, mus, covs)
            fid_obj = fid_from_moments_torch(mu_mix, cov_mix, mu_r, cov_r, eps=eps)
            fid_obj.backward()
            grad = alpha.grad.detach().cpu().numpy()

            eta = self.config.EG_ETA
            if self.config.EG_SQRT_DECAY:
                eta = eta / np.sqrt(eg_step + 1)

            logw = np.log(alpha_current + 1e-12) - eta * grad
            logw = logw - logw.max()
            alpha_new = np.exp(logw)
            alpha_new = alpha_new / alpha_new.sum()

            eps_floor = self.config.ALPHA_FLOOR
            if eps_floor > 0:
                alpha_new = np.maximum(alpha_new, eps_floor)
                alpha_new = alpha_new / alpha_new.sum()
            alpha_current = alpha_new

        return alpha_current

    def _solve_alpha_scipy(self, alpha0, mus, covs, mu_r, cov_r, eps):
        """High-precision simplex-constrained optimization of FID(alpha)."""
        m = self.number_of_arms
        if minimize is None:
            print("Warning: scipy is not available; falling back to EG optimizer.")
            return alpha0

        eps_floor = float(getattr(self.config, 'ALPHA_FLOOR', 0.0))
        if eps_floor * m >= 1.0:
            print("Warning: ALPHA_FLOOR is too large for simplex; using uniform alphas.")
            return np.ones(m, dtype=np.float64) / m

        lower = np.full(m, eps_floor, dtype=np.float64)
        upper = np.ones(m, dtype=np.float64)
        bounds = Bounds(lower, upper)
        linear_constraint = LinearConstraint(np.ones((1, m), dtype=np.float64), [1.0], [1.0])

        method = str(getattr(self.config, 'SCIPY_METHOD', 'trust-constr'))

        def _obj(x):
            v, _ = _fid_value_and_grad_for_alpha(x, mus, covs, mu_r, cov_r, eps=eps)
            if np.isnan(v) or np.isinf(v):
                return 1e30
            return v

        def _jac(x):
            _, g = _fid_value_and_grad_for_alpha(x, mus, covs, mu_r, cov_r, eps=eps)
            return g

        x0_candidates = [alpha0.copy()]
        x0_candidates.append(np.ones(m, dtype=np.float64) / m)
        restarts = int(max(0, getattr(self.config, 'SCIPY_RESTARTS', 0)))
        for _ in range(restarts):
            x = np.random.dirichlet(np.ones(m, dtype=np.float64))
            if eps_floor > 0:
                x = np.maximum(x, eps_floor)
                x = x / x.sum()
            x0_candidates.append(x)

        options = {'maxiter': int(getattr(self.config, 'SCIPY_MAXITER', 200))}
        if method == 'trust-constr':
            options.update({
                'gtol': float(getattr(self.config, 'SCIPY_GTOL', 1e-10)),
                'xtol': float(getattr(self.config, 'SCIPY_XTOL', 1e-12)),
                'barrier_tol': float(getattr(self.config, 'SCIPY_BARRIER_TOL', 1e-12)),
                'verbose': 0,
            })
        elif method == 'SLSQP':
            options.update({
                'ftol': float(getattr(self.config, 'SCIPY_FTOL', 1e-12)),
                'disp': False,
            })

        best_x = alpha0.copy()
        best_f = np.inf

        for x0 in x0_candidates:
            result = minimize(
                _obj,
                x0,
                method=method,
                jac=_jac,
                bounds=bounds,
                constraints=[linear_constraint],
                options=options,
            )
            x = np.asarray(result.x, dtype=np.float64)
            x = np.maximum(x, eps_floor)
            s = x.sum()
            if s <= 0:
                continue
            x = x / s
            fval = _obj(x)
            if np.isfinite(fval) and fval < best_f:
                best_f = fval
                best_x = x

        return best_x
        

    def _collect_all_samples(self):
        # Filter out empty sample arrays to avoid concatenation errors
        non_empty_samples = [self.samples[model] for model in self.model_names 
                            if len(self.samples[model]) > 0]
        if len(non_empty_samples) == 0:
            return np.array([])
        return np.concatenate(non_empty_samples, axis=0)

    def _get_sample_sizes(self):
        return [len(self.samples[model]) for model in self.model_names]
    
    def _cal_fid_sample_based(self):
        """
        Compute FID between all collected samples and the real dataset.
        Uses fid_from_moments_torch for differentiable FID computation.
        """
        if self.real_mu is None or self.real_cov is None:
            return float('nan')
        
        # Collect all samples
        all_samples = self._collect_all_samples()
        if len(all_samples) < 2:
            return float('nan')
        
        eps = self.config.FID_EPS
        mu_g, cov_g = _mean_cov_np(all_samples, eps=eps)
        
        mu_g = torch.tensor(mu_g, dtype=torch.float64)
        cov_g = torch.tensor(cov_g, dtype=torch.float64)
        mu_r = torch.tensor(self.real_mu, dtype=torch.float64)
        cov_r = torch.tensor(self.real_cov, dtype=torch.float64)
        
        fid = fid_from_moments_torch(mu_g, cov_g, mu_r, cov_r, eps=eps)
        
        return fid.item()       


def plot_fid_results(fid_scores=None, load_path=None, save_path=None, title='FID over Rounds',
                     annotate_interval=1000, per_model_fids=None):
    """Plot FID scores over rounds with annotations.
    
    Args:
        fid_scores: List of FID scores (optional if load_path is provided)
        load_path: Path to .npz file to load fid_scores from (optional)
        save_path: Path to save the figure (optional)
        title: Plot title
        annotate_interval: Add FID value annotations every N rounds (default 1000)
        per_model_fids: Dictionary of per-model FID scores to show as horizontal lines (optional)
    
    Returns:
        fig: The matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Load fid_scores from file if path provided
    if load_path is not None:
        data = np.load(load_path)
        fid_scores = data['fid_scores']
        print(f"Loaded FID scores from: {load_path}")
        print(f"  Shape: {fid_scores.shape}, Rounds: {len(fid_scores)}")
    
    if fid_scores is None:
        raise ValueError("Either fid_scores or load_path must be provided")
    
    fid_scores = np.array(fid_scores)[200::2]
    
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    
    rounds = np.arange(len(fid_scores))
    ax.plot(rounds, fid_scores, linewidth=2, color='blue', label='FID-EG')
    
    # Add annotations at specified intervals
    if annotate_interval > 0:
        annotation_rounds = list(range(0, len(fid_scores), annotate_interval))
        # Only add the last round if it's far enough from the previous annotation
        last_round = len(fid_scores) - 1
        if last_round not in annotation_rounds:
            # Only add if at least half an interval away from last annotation
            if annotation_rounds and (last_round - annotation_rounds[-1]) > annotate_interval // 2:
                annotation_rounds.append(last_round)
        
        for r in annotation_rounds:
            fid_val = fid_scores[r]
            is_last = (r == annotation_rounds[-1])
            
            # Add marker - larger and bolder for last point
            marker_size = 80 if is_last else 50
            ax.scatter([r], [fid_val], color='red', s=marker_size, zorder=5)
            
            # Add text annotation - bolder for last point
            fontsize = 11 if is_last else 9
            fontweight = 'bold' if is_last else 'normal'
            arrow_lw = 1.5 if is_last else 0.5
            ax.annotate(f'{fid_val:.2f}', 
                       xy=(r, fid_val), 
                       xytext=(10, 15),
                       textcoords='offset points',
                       fontsize=fontsize,
                       fontweight=fontweight,
                       color='darkred',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9 if is_last else 0.7),
                       arrowprops=dict(arrowstyle='->', color='darkred' if is_last else 'gray', lw=arrow_lw))
    
    # Add horizontal lines for per-model FID scores
    if per_model_fids is not None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(per_model_fids)))
        for i, (model_name, fid_val) in enumerate(per_model_fids.items()):
            ax.axhline(y=fid_val, color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'{model_name}: {fid_val:.2f}')
    
    ax.set_xlabel("Round", fontsize=18)
    ax.set_ylabel("FD", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    # ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add summary stats in text box
    final_fid = fid_scores[-1]
    min_fid = np.min(fid_scores)
    min_round = np.argmin(fid_scores)

    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    return fig


def plot_alpha_history(alpha_history, model_names, save_path=None, title='Alpha Weights over Rounds'):
    """Plot alpha weights for each model over rounds.
    
    Args:
        alpha_history: List of alpha arrays, shape (num_rounds, num_arms)
        model_names: List of model names
        save_path: Path to save the figure (optional)
        title: Plot title
    """
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


def compute_mixture_fid_from_alpha_history(alpha_history_path, model_names, dataset_name, 
                                           real_dataset_path, save_path=None, plot=True,
                                           num_samples=10000, feature_extractor='clip', seed=42,
                                           mixture_oracle_alphas=None, one_arm_oracle_alphas=None):
    """Compute mixture FID for each alpha in the history using fixed model moments.
    
    This is a post-processing function that:
    1. Loads alpha history from file
    2. Computes per-model moments using fixed samples (with seed for reproducibility)
    3. Computes mixture FID for each alpha
    4. Optionally plots and saves the results
    
    Args:
        alpha_history_path: Path to .npz file containing alpha_history
        model_names: List of model names
        dataset_name: Dataset name for loading model features
        real_dataset_path: Path to real dataset .npz file
        save_path: Path to save results (optional, will save .npz and .pdf)
        plot: Whether to generate plot (default True)
        num_samples: Number of samples per model for moment computation (default 10000)
        feature_extractor: Feature extractor key (default 'clip')
        seed: Random seed for reproducibility (default 42)
        mixture_oracle_alphas: Alpha weights for mixture oracle baseline (optional, plotted as dashed line)
        one_arm_oracle_alphas: Alpha weights for one-arm oracle baseline (optional, plotted as dashed line)
    
    Returns:
        Dictionary with alpha_history, mixture_fid_scores, and baseline FIDs if provided
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print(f"Computing Mixture FID from Alpha History")
    print(f"Alpha history: {alpha_history_path}")
    print(f"Models: {model_names}")
    print(f"Samples per model: {num_samples}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Load alpha history
    alpha_history = np.load(alpha_history_path)['alpha_history']
    print(f"Loaded alpha history: shape {alpha_history.shape}")
    
    # Load real dataset moments
    real_data = np.load(real_dataset_path)[f'{feature_extractor}_features']
    real_mu, real_cov = _mean_cov_np(real_data)
    mu_r = torch.tensor(real_mu, dtype=torch.float64)
    cov_r = torch.tensor(real_cov, dtype=torch.float64)
    print(f"Loaded real dataset: {real_data.shape}")
    
    # Compute per-model moments with fixed seed
    rng = np.random.RandomState(seed)
    model_mus = []
    model_covs = []
    
    for model_name in model_names:
        # Load model dataset
        if dataset_name in ['imagenet','ffhq','ffhq256', 'cifar10','lsun','toy','ffhq_truncated',
                           'quality_ffhq','afhq_truncated', 'FFHQ', 'FFHQ256', 'Imagenet256']:
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/dgm/{dataset_name}/features/', f'{model_name}.npz')
            # model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')

        if dataset_name == 'lsun':
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')
        
        model_data = np.load(model_path)[f'{feature_extractor}_features']
        
        n = min(num_samples, len(model_data))
        if n < num_samples:
            print(f"Warning: {model_name} has only {n} samples (requested {num_samples})")
        
        # Use fixed random indices
        idx = rng.choice(len(model_data), size=n, replace=False)
        samples = model_data[idx]
        
        mu_i, cov_i = _mean_cov_np(samples)
        model_mus.append(mu_i)
        model_covs.append(cov_i)
        print(f"  {model_name}: loaded {n} samples")
    
    # Convert to torch tensors
    model_mus_torch = torch.tensor(np.stack(model_mus), dtype=torch.float64)
    model_covs_torch = torch.tensor(np.stack(model_covs), dtype=torch.float64)
    
    # Compute mixture FID for each alpha
    print(f"\nComputing mixture FID for {len(alpha_history)} rounds...")
    mixture_fid_scores = []
    
    for i, alphas in enumerate(alpha_history):
        alpha_t = torch.tensor(alphas, dtype=torch.float64)
        mu_mix, cov_mix = _mixture_moments_torch(alpha_t, model_mus_torch, model_covs_torch)
        fid = fid_from_moments_torch(mu_mix, cov_mix, mu_r, cov_r, eps=1e-6)
        mixture_fid_scores.append(fid.item())
        
        if (i + 1) % 1000 == 0 or i == len(alpha_history) - 1:
            print(f"  Round {i+1}/{len(alpha_history)}: FID = {fid.item():.4f}, alpha = {alphas}")
        if (i+1) == 6000:
            break
    
    mixture_fid_scores = np.array(mixture_fid_scores)
    
    # Compute baseline FIDs using the same fixed model moments
    mixture_oracle_fid = None
    one_arm_oracle_fid = None
    
    if mixture_oracle_alphas is not None:
        alpha_t = torch.tensor(np.array(mixture_oracle_alphas), dtype=torch.float64)
        mu_mix, cov_mix = _mixture_moments_torch(alpha_t, model_mus_torch, model_covs_torch)
        mixture_oracle_fid = fid_from_moments_torch(mu_mix, cov_mix, mu_r, cov_r, eps=1e-6).item()
        print(f"Mixture Oracle FID (alphas={mixture_oracle_alphas}): {mixture_oracle_fid:.4f}")
    
    if one_arm_oracle_alphas is not None:
        alpha_t = torch.tensor(np.array(one_arm_oracle_alphas), dtype=torch.float64)
        mu_mix, cov_mix = _mixture_moments_torch(alpha_t, model_mus_torch, model_covs_torch)
        one_arm_oracle_fid = fid_from_moments_torch(mu_mix, cov_mix, mu_r, cov_r, eps=1e-6).item()
        print(f"One-Arm Oracle FID (alphas={one_arm_oracle_alphas}): {one_arm_oracle_fid:.4f}")
    
    # Save results
    if save_path is not None:
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FID scores
        base_name = os.path.splitext(save_path)[0]
        print(f"\nSaved mixture FID scores to: {base_name}.npz")
    
    # Plot results
    if plot:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        
        rounds = np.arange(len(mixture_fid_scores))[100::]
        ax.plot(rounds, mixture_fid_scores[100::], linewidth=2, color='green', label='FID-EG')
        
        # Add baseline dashed lines
        if mixture_oracle_fid is not None:
            ax.axhline(y=mixture_oracle_fid, color='red', linestyle='--', linewidth=2, 
                      label=f'Mixture Oracle: {mixture_oracle_fid:.2f}')
        if one_arm_oracle_fid is not None:
            ax.axhline(y=one_arm_oracle_fid, color='blue', linestyle='--', linewidth=2, 
                      label=f'One-Arm Oracle: {one_arm_oracle_fid:.2f}')
        
        ax.set_xlabel("Round", fontsize=18)
        ax.set_ylabel("Mixture FID", fontsize=18)
        ax.set_title(f'Mixture FID from Alpha History ({dataset_name})', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        ax.scatter([len(mixture_fid_scores)-1], [mixture_fid_scores[-1]], color='red', s=80, zorder=5)
        ax.annotate(f'{mixture_fid_scores[-1]:.2f}', 
                   xy=(len(mixture_fid_scores)-1, mixture_fid_scores[-1]), 
                   xytext=(10, 15),
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   color='darkred',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
        
        fig.tight_layout()
        
        if save_path is not None:
            fig.savefig(f'{base_name}_6000.pdf', bbox_inches='tight')
            print(f"Saved plot to: {base_name}.pdf")
        
        plt.show()
    
    print(f"\n{'='*60}")
    print(f"Final Mixture FID: {mixture_fid_scores[-1]:.4f}")
    print(f"Min Mixture FID: {np.min(mixture_fid_scores):.4f} (round {np.argmin(mixture_fid_scores)})")
    if mixture_oracle_fid is not None:
        print(f"Mixture Oracle FID: {mixture_oracle_fid:.4f}")
    if one_arm_oracle_fid is not None:
        print(f"One-Arm Oracle FID: {one_arm_oracle_fid:.4f}")
    print(f"{'='*60}\n")
    
    result = {
        'alpha_history': alpha_history,
        'mixture_fid_scores': mixture_fid_scores
    }
    if mixture_oracle_fid is not None:
        result['mixture_oracle_fid'] = mixture_oracle_fid
    if one_arm_oracle_fid is not None:
        result['one_arm_oracle_fid'] = one_arm_oracle_fid
    
    return result


class SimpleOfflineEvaluator:
    """Simple offline evaluator that just holds the real dataset for FID computation."""
    
    def __init__(self, real_dataset_path, feature_key='dino_features', optimal_alphas=None):
        """Load real dataset for FID computation.
        
        Args:
            real_dataset_path: Path to .npz file containing real dataset features
            feature_key: Key in the .npz file for the features
            optimal_alphas: Optional pre-computed optimal mixture weights (for mixture-oracle mode)
        """
        data = np.load(real_dataset_path)
        self.real_dataset = data[feature_key]
        print(f"Loaded real dataset: {self.real_dataset.shape}")
        
        # Store optimal alphas (used for mixture-oracle mode)
        self.optimal_alphas = optimal_alphas
        if optimal_alphas is not None:
            print(f"Loaded optimal alphas: {optimal_alphas}")
        self.optimal_model = None


def compute_fid_for_alphas(alphas, model_names, dataset_name, real_dataset_path,
                           num_samples=10000, feature_extractor='dino'):
    """Compute FID for a given mixture of models with specified alpha weights.
    
    This function computes the mixture FID by:
    1. Loading/subsampling num_samples from each model's dataset
    2. Computing per-model moments (mean, covariance)
    3. Computing mixture moments using the given alphas
    4. Computing FID between the mixture distribution and real distribution
    
    Args:
        alphas: Array of mixture weights (must sum to 1), shape (num_models,)
        model_names: List of model names
        dataset_name: Dataset name for loading model features
        real_dataset_path: Path to real dataset .npz file
        num_samples: Number of samples to use from each model (default 10000)
        feature_extractor: Feature extractor key (default 'dino')
    
    Returns:
        fid_score: The mixture FID score
    """
    alphas = np.array(alphas, dtype=np.float64)
    assert len(alphas) == len(model_names), "alphas must match number of models"
    assert np.abs(alphas.sum() - 1.0) < 1e-6, f"alphas must sum to 1, got {alphas.sum()}"
    
    # Load real dataset
    real_data = np.load(real_dataset_path)[f'{feature_extractor}_features']
    real_mu, real_cov = _mean_cov_np(real_data)
    
    # Convert real moments to torch
    mu_r = torch.tensor(real_mu, dtype=torch.float64)
    cov_r = torch.tensor(real_cov, dtype=torch.float64)
    
    # Compute per-model moments
    mus = []
    covs = []
    
    for model_name in model_names:
        # Load model dataset
        if dataset_name in ['imagenet','ffhq','ffhq256', 'cifar10','lsun','toy','ffhq_truncated',
                           'quality_ffhq','afhq_truncated', 'FFHQ', 'FFHQ256', 'Imagenet256']:
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/dgm/{dataset_name}/features/', f'{model_name}.npz')
        else:
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')
        
        model_data = np.load(model_path)[f'{feature_extractor}_features']
        
        # Subsample if needed
        n = min(num_samples, len(model_data))
        if n < num_samples:
            print(f"Warning: {model_name} has only {n} samples (requested {num_samples})")
        idx = np.random.choice(len(model_data), size=n, replace=False)
        samples = model_data[idx]
        
        # Compute moments
        mu_i, cov_i = _mean_cov_np(samples)
        mus.append(mu_i)
        covs.append(cov_i)
    
    # Convert to torch tensors
    mus = torch.tensor(np.stack(mus), dtype=torch.float64)
    covs = torch.tensor(np.stack(covs), dtype=torch.float64)
    alpha_t = torch.tensor(alphas, dtype=torch.float64)
    
    # Compute mixture moments
    mu_mix, cov_mix = _mixture_moments_torch(alpha_t, mus, covs)
    
    # Compute FID
    fid = fid_from_moments_torch(mu_mix, cov_mix, mu_r, cov_r, eps=1e-6)
    
    return fid.item()


def compute_per_model_fid(model_names, dataset_name, real_dataset_path, 
                          num_samples=10000, feature_extractor='dino'):
    """Compute FID for each model against the real dataset.
    
    Args:
        model_names: List of model names
        dataset_name: Dataset name for loading model features
        real_dataset_path: Path to real dataset .npz file
        num_samples: Number of samples to use from each model
        feature_extractor: Feature extractor key (default 'dino')
    
    Returns:
        Dictionary mapping model name to FID score
    """
    print(f"\n{'='*60}")
    print(f"Computing Per-Model FID Scores")
    print(f"Models: {model_names}")
    print(f"Samples per model: {num_samples}")
    print(f"{'='*60}\n")
    
    # Load real dataset
    real_data = np.load(real_dataset_path)[f'{feature_extractor}_features']
    print(f"Loaded real dataset: {real_data.shape}")
    real_mu, real_cov = _mean_cov_np(real_data)
    
    # Convert real moments to torch
    mu_r = torch.tensor(real_mu, dtype=torch.float64)
    cov_r = torch.tensor(real_cov, dtype=torch.float64)
    
    fid_scores = {}
    
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        
        # Load model dataset
        if dataset_name in ['imagenet','ffhq','ffhq256', 'cifar10','lsun','toy','ffhq_truncated',
                           'quality_ffhq','afhq_truncated', 'FFHQ', 'FFHQ256', 'Imagenet256']:
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/dgm/{dataset_name}/features', f'{model_name}.npz')
            # model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')

        if dataset_name == "lsun":
            model_path = os.path.join(f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/dino/', f'{model_name}.npz')
        
        model_data = np.load(model_path)[f'{feature_extractor}_features']
        print(f"  Loaded {model_name}: {model_data.shape}")
        
        # Use up to num_samples (or all if fewer available)
        n = min(num_samples, len(model_data))
        if n < num_samples:
            print(f"  Warning: Only {n} samples available (requested {num_samples})")

        # Pick n random samples without replacement from the model data
        idx = np.random.choice(len(model_data), size=n, replace=False)
        samples = model_data[idx]
        
        # Compute moments
        mu_g, cov_g = _mean_cov_np(samples)
        mu_g = torch.tensor(mu_g, dtype=torch.float64)
        cov_g = torch.tensor(cov_g, dtype=torch.float64)
        
        # Compute FID
        fid = fid_from_moments_torch(mu_g, cov_g, mu_r, cov_r, eps=1e-6)
        fid_scores[model_name] = fid.item()
        
        print(f"  FID({model_name}): {fid.item():.4f}")
    
    print(f"\n{'='*60}")
    print("Summary: Per-Model FID Scores")
    print(f"{'='*60}")
    for name, fid in sorted(fid_scores.items(), key=lambda x: x[1]):
        print(f"  {name}: {fid:.4f}")
    print(f"{'='*60}\n")
    return fid_scores


def get_results(config=None, model_names=None, dataset_name=None, real_dataset_path=None, 
                save_dir=None, num_simulations=1, mode='mixture-greedy', optimal_alphas=None,
                mixture_oracle_alphas=None, one_arm_oracle_alphas=None):
    import os
    
    if config is None:
        config = ConfigFID()
    
    if model_names is None:
        model_names = ['arm0', 'arm1']
    
    if dataset_name is None:
        dataset_name = 'Imagenet256'
    
    if save_dir is None:
        save_dir = f'/research/d7/rshr/bahar/MAB/Mixture-Greedy/methods/FID_Scipy/{mode}/{dataset_name}/{config.TOTAL_ROUNDS}'
    
    os.makedirs(save_dir, exist_ok=True)
    
    if real_dataset_path is not None:
        offline_evaluator = SimpleOfflineEvaluator(real_dataset_path, optimal_alphas=optimal_alphas)
    else:
        default_real_path = f'/research/d7/rshr/bahar/data/datasets/{dataset_name}/clip/real.npz'
        if os.path.exists(default_real_path):
            offline_evaluator = SimpleOfflineEvaluator(default_real_path, optimal_alphas=optimal_alphas)
        else:
            print(f"Warning: No real dataset found at {default_real_path}. FID will be nan.")
            offline_evaluator = None
    
    print(f"\n{'='*60}")
    print(f"Running Online Evaluation - Mode: {mode}")
    print(f"Models: {model_names}")
    print(f"Dataset: {dataset_name}")
    print(f"Total Rounds: {config.TOTAL_ROUNDS}")
    if mode == 'mixture-oracle':
        print(f"Optimal Alphas: {optimal_alphas}")
    else:
        print(f"Optimizer: {config.OPTIMIZER}")
        print(f"EG Learning Rate (eta): {config.EG_ETA}")
    print(f"{'='*60}\n")
    
    all_fid_scores = []
    all_mixture_fid_scores = []
    all_alpha_histories = []
    sim_times = []
    
    total_start_time = time.time()
    
    for sim in range(num_simulations):
        print(f"\n{'='*60}")
        print(f"Simulation {sim + 1}/{num_simulations}")
        print(f"{'='*60}")
        
        evaluator = FIDOnlineEvaluator(
            model_names=model_names,
            dataset_name=dataset_name,
            offline_evaluator=offline_evaluator,
            mode=mode,
            config=config,
            optimal_alphas=optimal_alphas
        )
        
        print("Starting online evaluation...")
        sim_start_time = time.time()
        
        evaluator.run_online_evaluation(num_rounds=config.TOTAL_ROUNDS)
        
        sim_elapsed = time.time() - sim_start_time
        sim_times.append(sim_elapsed)
        
        print(f"\nSimulation {sim + 1} completed in {sim_elapsed:.2f} seconds.")
        evaluator.report_sample_nums()
        
        all_fid_scores.append(evaluator.fid_scores)
        all_mixture_fid_scores.append(evaluator.mixture_fid_scores)
        all_alpha_histories.append(evaluator.alpha_history)
    
    total_elapsed = time.time() - total_start_time
    avg_elapsed = total_elapsed / num_simulations
    
    mean_fid_scores = np.mean(np.stack(all_fid_scores), axis=0)
    mean_mixture_fid_scores = np.mean(np.stack(all_mixture_fid_scores), axis=0)
    final_alpha_history = all_alpha_histories[-1]
    
    np.savez(f'{save_dir}/fid_scores.npz', fid_scores=mean_fid_scores)
    np.savez(f'{save_dir}/alpha_history.npz', alpha_history=final_alpha_history)
    
    with open(f'{save_dir}/config.txt', 'w') as f:
        f.write(f"Mode: {mode}\n")
        f.write(f"Models: {model_names}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Rounds: {config.TOTAL_ROUNDS}\n")
        if mode == 'mixture-oracle':
            f.write(f"Optimal Alphas (input): {optimal_alphas}\n")
        else:
            f.write(f"OPTIMIZER: {config.OPTIMIZER}\n")
            f.write(f"EG_ETA: {config.EG_ETA}\n")
            f.write(f"EG_SQRT_DECAY: {config.EG_SQRT_DECAY}\n")
            f.write(f"ALPHA_FLOOR: {config.ALPHA_FLOOR}\n")
            f.write(f"EG_STEPS: {config.EG_STEPS}\n")
            if str(config.OPTIMIZER).lower() == 'scipy':
                f.write(f"SCIPY_METHOD: {config.SCIPY_METHOD}\n")
                f.write(f"SCIPY_MAXITER: {config.SCIPY_MAXITER}\n")
                f.write(f"SCIPY_GTOL: {config.SCIPY_GTOL}\n")
                f.write(f"SCIPY_XTOL: {config.SCIPY_XTOL}\n")
                f.write(f"SCIPY_BARRIER_TOL: {config.SCIPY_BARRIER_TOL}\n")
                f.write(f"SCIPY_FTOL: {config.SCIPY_FTOL}\n")
                f.write(f"SCIPY_RESTARTS: {config.SCIPY_RESTARTS}\n")
        f.write(f"Final alphas: {final_alpha_history[-1]}\n")
        f.write(f"Sample sizes: {evaluator._get_sample_sizes()}\n")
        f.write(f"Simulation times (seconds): {sim_times}\n")
        f.write(f"Total running time (seconds): {total_elapsed:.2f}\n")
        f.write(f"Average running time per simulation (seconds): {avg_elapsed:.2f}\n")
    
    print(f"\n{'='*60}")
    print("All simulations completed!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {save_dir}")
    print(f"Final alphas: {final_alpha_history[-1]}")
    print(f"Final FID (samples): {mean_fid_scores[-1]:.4f}")
    print(f"Total running time: {total_elapsed:.2f} seconds")
    print(f"Average running time per simulation: {avg_elapsed:.2f} seconds")
    
    print("\nGenerating plots...")
    try:
        import matplotlib  # noqa: F401
        plot_fid_results(
            mean_fid_scores, 
            save_path=f'{save_dir}/fid_plot.pdf',
            title=f'FID over Rounds ({dataset_name})'
        )
        plot_alpha_history(
            final_alpha_history,
            model_names,
            save_path=f'{save_dir}/alpha_plot.pdf',
            title=f'Alpha Weights ({dataset_name})'
        )
    except Exception as e:
        print(f"Skipping plotting because matplotlib is not available or failed: {e}")
    
    return {
        'fid_scores': mean_fid_scores,
        'mixture_fid_scores': mean_mixture_fid_scores,
        'mixture_fid_fixed': None,
        'alpha_history': final_alpha_history,
        'sample_sizes': evaluator._get_sample_sizes(),
        'model_names': model_names,
        'simulation_times': sim_times,
        'total_time_seconds': total_elapsed,
        'avg_time_seconds': avg_elapsed,
    }


if __name__ == "__main__":
    import sys

    from mixture_greedy.__main__ import main

    raise SystemExit(main(sys.argv[1:], default_metric="fid"))
    config.EG_SQRT_DECAY = False

    config.EG_STEPS = 1000

    config.INITIAL_SAMPLE_COUNT = 5

