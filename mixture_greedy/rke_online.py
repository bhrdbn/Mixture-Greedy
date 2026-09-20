"""Online sampling and arm-selection policies for RKE/KID experiments."""

import os
import sys

import cvxpy as cp
import numpy as np

from mixture_greedy.data.features import (
    _candidate_feature_npz_paths,
    _load_features_from_npz,
    _resolve_existing_path,
)
from mixture_greedy.metrics.fid import compute_fid
from mixture_greedy.metrics.rke import (
    KernelUtils,
    _linear_coeff,
    default_config,
)


def _legacy_cluster_name():
    """Read the intentionally module-global cluster name from the legacy facade."""
    legacy_module = sys.modules.get("online_reproducible")
    if legacy_module is not None and "cluster_name" in vars(legacy_module):
        return vars(legacy_module)["cluster_name"]
    raise NameError("name 'cluster_name' is not defined")


class RKEOnlineEvaluator:
    def __init__(
        self,
        model_names,
        dataset_name,
        offline_evaluator,
        mode="mix_ucb",
        use_linear=False,
        config=None,
    ):
        if config is None:
            config = default_config
        self.config = config
        self.dataset_name = dataset_name
        self.OfflineEvaluator = offline_evaluator
        self.model_names = model_names
        self.mode = mode
        self.model2idx = {}
        self.use_linear = use_linear
        if self.use_linear:
            self.linears = {model_name: [] for model_name in model_names}
        self.alphas = None
        for i in range(len(model_names)):
            self.model2idx[model_names[i]] = i
        self.datasets = self.OfflineEvaluator.datasets

        if self.config.CHANGE_POINT:
            self._load_change_point_online_datasets()
        if self.config.MALE_FEMALE_ADAPTIVE:
            self._load_male_female_online_datasets()
        if self.config.CATS_BIRDS_ADAPTIVE:
            self._load_cats_birds_online_datasets()

        self.samples = {model: [] for model in model_names}
        self.sample_indexes = {model: 0 for model in model_names}
        self.sample_indexes_post = {model: 0 for model in model_names}
        self.current_round = 0
        self.number_of_arms = len(model_names)
        self.kernel = np.zeros((len(self.model_names), len(self.model_names)))
        self.scores = []
        self.fid_scores = []
        self.alpha_history = []

    def _load_change_point_online_datasets(self):
        """Load both pre- and post-change datasets for online sampling."""
        print("--- Loading change-point datasets for online evaluation")
        data = np.load(self.config.CHANGE_POINT_PATH)

        self.datasets_pre = {}
        self.datasets_post = {}

        for model in self.model_names:
            if model == "arm0":
                pre_arr = data["arm0_pre"]
                post_arr = data["arm0_post"]
            elif model == "arm1":
                pre_arr = data["arm1_pre"]
                post_arr = data["arm1_post"]
            else:
                raise ValueError(
                    f"Unknown arm name for change-point dataset: {model}"
                )

            self.datasets_pre[model] = pre_arr[
                np.random.permutation(len(pre_arr))
            ]
            self.datasets_post[model] = post_arr[
                np.random.permutation(len(post_arr))
            ]
            print(
                f"Loaded {model}: pre={len(self.datasets_pre[model])}, "
                f"post={len(self.datasets_post[model])}"
            )

    def _load_male_female_online_datasets(self):
        """Load cats/dogs data for adaptive online sampling."""
        print("--- Loading cats/dogs datasets for online evaluation")
        self.male_data = np.load("/research/d7/rshr/bahar/MAB/cats.npz")[
            "dino_features"
        ]
        self.female_data = np.load("/research/d7/rshr/bahar/MAB/dogs.npz")[
            "dino_features"
        ]

        np.random.shuffle(self.male_data)
        np.random.shuffle(self.female_data)

        self.male_idx = {"arm0": 0, "arm1": 0}
        self.female_idx = {"arm0": 0, "arm1": 0}

        print(
            f"Loaded cats (male): {self.male_data.shape}, "
            f"dogs (female): {self.female_data.shape}"
        )

    def _load_cats_birds_online_datasets(self):
        """Load cats/birds/animals data for adaptive online sampling."""
        print("--- Loading cats/birds/animals datasets for online evaluation")
        self.cats_data = np.load(
            os.path.join(
                self.config.CATS_BIRDS_DATA_PATH, "kandinsky_giraffe.npz"
            )
        )["dino_features"]
        self.catsbirds_data = np.load(
            os.path.join(
                self.config.CATS_BIRDS_DATA_PATH,
                "kandinsky_shark_giraffe_shuffled.npz",
            )
        )["dino_features"]
        self.animals_data = np.load(
            os.path.join(
                self.config.CATS_BIRDS_DATA_PATH, "kandinsky_others.npz"
            )
        )["dino_features"]

        np.random.shuffle(self.cats_data)
        np.random.shuffle(self.catsbirds_data)
        np.random.shuffle(self.animals_data)

        self.cats_idx = {"arm0": 0}
        self.catsbirds_idx = {"arm1": 0}
        self.animals_idx = {"arm0": 0}

        print(
            f"Loaded cats: {self.cats_data.shape}, "
            f"catsbirds: {self.catsbirds_data.shape}, "
            f"animals: {self.animals_data.shape}"
        )

    def _load_dataset(self, model_name):
        print(f"--- Loading model: {model_name}")
        dataset_path = self._get_dataset_path(model_name)
        if self.config.DEBUG:
            dataset, key_used = _load_features_from_npz(
                dataset_path, self.config.feature_extractor, return_key=True
            )
            print(
                f"[ONLINE] feature_extractor={self.config.feature_extractor} "
                f"path={dataset_path} key={key_used}"
            )
        else:
            dataset = _load_features_from_npz(
                dataset_path, self.config.feature_extractor
            )
        shuffled_indices = np.random.permutation(dataset.shape[0])
        dataset = dataset[shuffled_indices]
        return dataset

    def report_sample_nums(self):
        for i in self.model_names:
            print(f"model: {i}, num: {len(self.samples[i])}")

    def _load_all_datasets(self):
        return {model: self._load_dataset(model) for model in self.model_names}

    def _get_dataset_path(self, model_name):
        if self.dataset_name in [
            "imagenet",
            "ffhq",
            "ffhq256",
            "cifar10",
            "lsun",
            "toy",
            "ffhq_truncated",
            "quality_ffhq",
            "afhq_truncated",
            "FFHQ",
            "FFHQ256",
            "Imagenet256",
        ]:
            candidates = _candidate_feature_npz_paths(
                self.dataset_name, model_name, self.config.feature_extractor
            )
            return _resolve_existing_path(candidates)
        if self.dataset_name == "t2i_cluster":
            return os.path.join("..", f"{model_name}/{_legacy_cluster_name()}.npz")
        if self.dataset_name == "t2i_coco":
            return os.path.join("../", f"{model_name}.npz")
        if self.dataset_name == "t2t":
            return os.path.join("../", f"{model_name}_features.npz")
        if self.dataset_name == "imagewoof":
            return os.path.join("../", f"{model_name}.npz")
        if self.dataset_name == "styles":
            return os.path.join("../", f"{model_name}.npz")
        if self.dataset_name == "t2i_dog":
            return os.path.join("..//", f"{model_name}.npz")
        if self.dataset_name == "t2i_cars":
            return os.path.join(
                "/research/d7/rshr/bahar/data/sdxl", f"{model_name}_cars.npz"
            )
        if self.dataset_name == "t2i_birds":
            return os.path.join(
                "/research/d7/rshr/bahar/MAB/red_cartoony_birds/embeddings",
                f"steps_{10}.npz",
            )
        if self.dataset_name == "red_birds":
            return os.path.join(
                "/research/d7/rshr/bahar/red_birds/embeddings",
                f"{model_name}.npz",
            )
        return None

    def _get_samples(self, model_name, num_samples):
        if self.config.CHANGE_POINT:
            return self._get_samples_change_point(model_name, num_samples)
        if self.config.MALE_FEMALE_ADAPTIVE:
            return self._get_samples_male_female(model_name, num_samples)
        if self.config.CATS_BIRDS_ADAPTIVE:
            return self._get_samples_cats_birds(model_name, num_samples)

        dataset = self.datasets[model_name]
        n = len(dataset)

        if self.sample_indexes[model_name] + num_samples > n:
            returning_sample = dataset[np.random.randint(0, n, size=num_samples)]
        else:
            start_idx = self.sample_indexes[model_name]
            returning_sample = dataset[start_idx : start_idx + num_samples]

        self.sample_indexes[model_name] = (
            self.sample_indexes[model_name] + num_samples
        )
        return returning_sample

    def _get_samples_male_female(self, model_name, num_samples):
        """Sample the adaptive cats/dogs mixture for the current round."""
        samples = []

        for _ in range(num_samples):
            if model_name == "arm0":
                if self.current_round < self.config.MALE_FEMALE_CHANGE_ROUND:
                    male_ratio = self.config.ARM1_MALE_RATIO_BEFORE
                else:
                    male_ratio = self.config.ARM1_MALE_RATIO_AFTER
            elif model_name == "arm1":
                male_ratio = self.config.ARM2_MALE_RATIO
            else:
                raise ValueError(f"Unknown arm name for MNIST: {model_name}")

            if np.random.random() < male_ratio:
                idx = self.male_idx[model_name]
                if idx >= len(self.male_data):
                    idx = 0
                    self.male_idx[model_name] = 0
                sample = self.male_data[idx]
                self.male_idx[model_name] += 1
            else:
                idx = self.female_idx[model_name]
                if idx >= len(self.female_data):
                    idx = 0
                    self.female_idx[model_name] = 0
                sample = self.female_data[idx]
                self.female_idx[model_name] += 1

            samples.append(sample)

        self.sample_indexes[model_name] += num_samples
        return np.array(samples)

    def _get_samples_change_point(self, model_name, num_samples):
        """Sample from the pre- or post-change data for the current round."""
        change_point = self.config.TOTAL_ROUNDS // 2

        if self.current_round < change_point:
            dataset = self.datasets_pre[model_name]
            idx = self.sample_indexes[model_name]
            if idx + num_samples > len(dataset):
                print(f"ERROR: Too many pre-change samples needed for {model_name}")
                assert 0 == 1
            returning_sample = dataset[idx : idx + num_samples]
            self.sample_indexes[model_name] += num_samples
        else:
            dataset = self.datasets_post[model_name]
            idx = self.sample_indexes_post[model_name]
            if idx + num_samples > len(dataset):
                print(f"ERROR: Too many post-change samples needed for {model_name}")
                assert 0 == 1
            returning_sample = dataset[idx : idx + num_samples]
            self.sample_indexes_post[model_name] += num_samples

        return returning_sample

    def _get_samples_cats_birds(self, model_name, num_samples):
        """Sample the adaptive cats/birds/animals arm for the current round."""
        samples = []

        for _ in range(num_samples):
            if model_name == "arm0":
                if self.current_round < self.config.CATS_BIRDS_CHANGE_ROUND:
                    idx = self.cats_idx["arm0"]
                    if idx >= len(self.cats_data):
                        idx = 0
                        self.cats_idx["arm0"] = 0
                    sample = self.cats_data[idx]
                    self.cats_idx["arm0"] += 1
                else:
                    idx = self.animals_idx["arm0"]
                    if idx >= len(self.animals_data):
                        idx = 0
                        self.animals_idx["arm0"] = 0
                    sample = self.animals_data[idx]
                    self.animals_idx["arm0"] += 1
            elif model_name == "arm1":
                idx = self.catsbirds_idx["arm1"]
                if idx >= len(self.catsbirds_data):
                    idx = 0
                    self.catsbirds_idx["arm1"] = 0
                sample = self.catsbirds_data[idx]
                self.catsbirds_idx["arm1"] += 1
            else:
                raise ValueError(f"Unknown arm name for cats/birds: {model_name}")

            samples.append(sample)

        self.sample_indexes[model_name] += num_samples
        return np.array(samples)

    def _extend_samples(self, current_samples, new_samples):
        if len(current_samples) == 0:
            return new_samples
        return np.concatenate((current_samples, new_samples), axis=0)

    def initial_alphas(self, round_):
        if self.mode == "one-arm-oracle":
            model_idx = self.OfflineEvaluator.optimal_model
            alphas = np.array([0 for _ in range(self.number_of_arms)])
            alphas[model_idx] = 1
        elif self.mode == "mixture-oracle":
            alphas = self.OfflineEvaluator.optimal_alphas
        elif self.mode in [
            "mixture-ucb",
            "one-arm-ucb",
            "mixture-greedy",
            "mixture-greedy-eg",
        ]:
            model_idx = round_ % self.number_of_arms
            alphas = np.array([0 for _ in range(self.number_of_arms)])
            alphas[model_idx] = 1
        else:
            print("INVALID MODE")
            assert 0 == 1
        return alphas

    def run_online_evaluation(self, num_rounds, num_samples=None):
        if num_samples is None:
            num_samples = self.config.MINI_BATCH
        for round_ in range(0, num_rounds):
            self.current_round = round_

            if (
                self.config.CHANGE_POINT
                and round_ == self.config.TOTAL_ROUNDS // 2
            ):
                print(f"=== CHANGE POINT at round {round_} ===")

            if (
                self.config.MALE_FEMALE_ADAPTIVE
                and round_ == self.config.MALE_FEMALE_CHANGE_ROUND
            ):
                print(f"=== MALE/FEMALE CHANGE POINT at round {round_} (UCB) ===")

            if self.config.MALE_FEMALE_ADAPTIVE and round_ in [
                0,
                999,
                1000,
                1001,
                1010,
            ]:
                male_ratio = (
                    self.config.ARM1_MALE_RATIO_BEFORE
                    if round_ < self.config.MALE_FEMALE_CHANGE_ROUND
                    else self.config.ARM1_MALE_RATIO_AFTER
                )
                print(
                    f"[UCB] round={round_}, arm0 male_ratio="
                    f"{male_ratio}"
                )

            alphas = self._select_best_model_ucb(round_)
            self.alpha_history.append(alphas)

            chosen_models = np.random.choice(
                self.model_names, size=num_samples, p=alphas
            )

            for model_name in chosen_models:
                new_sample = self._get_samples(model_name, 1)
                self.samples[model_name] = self._extend_samples(
                    self.samples[model_name], new_sample
                )
                self._update_the_kernel(new_sample, model_name)

            score = self._cal_metric_sample_based()
            self.scores.append(score)

            if self.use_linear and self.OfflineEvaluator.real_dataset is not None:
                if round_ % 200 == 0 or round_ == num_rounds - 1:
                    fid = self._cal_fid_sample_based()
                    self.fid_scores.append(fid)

    def best_rke_model(self, round_):
        if self.mode == "one-arm-oracle":
            model_idx = self.OfflineEvaluator.optimal_model
        else:
            kernel_matrix = self.kernel
            kernel_matrix = KernelUtils.scaled_kernel(
                kernel_matrix, self._get_sample_sizes(), config=self.config
            )
            arm_values = [
                kernel_matrix[i, i] for i in range(len(self.model_names))
            ]
            if self.use_linear:
                coeff = _linear_coeff(self.config)
                precision_array = np.array(
                    [np.mean(self.linears[mod]) for mod in self.model_names]
                )
                precision_array = coeff * precision_array
                arm_values -= precision_array
            ucb_values = [
                self.config.DELTA_L
                * np.sqrt(
                    self.config.BETA
                    * np.log(round_)
                    / (2 * len(self.samples[m]) / self.config.MINI_BATCH)
                )
                for m in self.model_names
            ]
            if round_ % self.config.ALPHA_PRINT == 0:
                print("arm val")
                print(arm_values)
                print([len(self.samples[n]) for n in self.model_names])
                print(ucb_values)
                print("done val")
            arm_values_ucb = [
                arm_values[i] - ucb_values[i] for i in range(len(arm_values))
            ]
            if self.use_linear:
                coeff = _linear_coeff(self.config)
                precision_ucb_values = coeff * np.array(
                    [
                        self.config.DELTA_G
                        / len(self.samples[m])
                        * self.config.MINI_BATCH
                        for m in self.model_names
                    ]
                )
                arm_values_ucb = [
                    arm_values_ucb[i] - precision_ucb_values[i]
                    for i in range(len(arm_values_ucb))
                ]
            model_idx = np.argmin(arm_values_ucb)

        alphas = np.array([0 for _ in range(len(self.model_names))])
        alphas[model_idx] = 1
        return alphas

    def _cal_metric_sample_based(self):
        sample_count = np.sum(self._get_sample_sizes())
        score = np.sum(self.kernel) / (sample_count * sample_count)

        if self.config.QUADRATIC_METRIC == "rke" and not self.use_linear:
            return 1 / score

        if self.config.QUADRATIC_METRIC == "kid":
            score += self.OfflineEvaluator.kid_self_kernel
        elif self.config.QUADRATIC_METRIC != "rke":
            return None

        linear_sum = sum(np.sum(self.linears[model]) for model in self.model_names)
        linear_count = sum(len(self.linears[model]) for model in self.model_names)
        precision = np.mean(linear_sum / linear_count)
        return score - _linear_coeff(self.config) * precision

    def _cal_fid_sample_based(self):
        """Compute FID between all collected samples and the real dataset."""
        all_samples = self._collect_all_samples()

        if len(all_samples) < 2:
            return float("inf")

        real_features = self.OfflineEvaluator.real_dataset

        if real_features is None:
            return float("inf")

        fid = compute_fid(real_features, all_samples)
        return fid

    def _select_best_model_ucb(self, round_):
        if round_ < self.config.INITIAL_SAMPLE_COUNT * self.number_of_arms:
            return self.initial_alphas(round_)
        if self.mode in ["one-arm-oracle", "one-arm-ucb"]:
            return self.best_rke_model(round_)
        if self.mode == "mixture-oracle":
            return self.OfflineEvaluator.optimal_alphas
        if self.mode == "mixture-greedy-eg":
            return self._select_best_model_mixture_greedy_eg(round_)

        if self.config.OGD:
            optimal_alphas = self._select_ogd_alphas(round_)
        else:
            optimal_alphas = self._solve_mixture_program(round_)

        optimal_alphas /= np.sum(optimal_alphas)
        return optimal_alphas

    def _scaled_kernel(self):
        return KernelUtils.scaled_kernel(
            self.kernel,
            self._get_sample_sizes(),
            config=self.config,
        )

    def _diversity_ucb_weights(self, round_):
        return np.array(
            [
                self.config.DELTA_L
                * np.sqrt(
                    self.config.BETA
                    * np.log(round_)
                    / len(self.samples[model])
                    * self.config.MINI_BATCH
                )
                for model in self.model_names
            ]
        )

    def _precision_values(self):
        coefficient = _linear_coeff(self.config)
        return coefficient * np.array(
            [np.mean(self.linears[model]) for model in self.model_names]
        )

    def _precision_ucb_weights(self):
        coefficient = _linear_coeff(self.config)
        return coefficient * np.array(
            [
                self.config.DELTA_G
                / len(self.samples[model])
                * self.config.MINI_BATCH
                for model in self.model_names
            ]
        )

    def _epsilon_constraints(self, round_):
        if self.mode == "mixture-greedy" and self.config.DYNAMIC_EPSILON:
            epsilon = min(
                1.0 / self.number_of_arms,
                1.0 / max(1, round_) ** (1 / 3),
            )
            return np.full(self.number_of_arms, epsilon)
        if len(self.config.EPSILON) == 0 or self.mode == "mixture-ucb":
            return np.zeros(self.number_of_arms)
        if len(self.config.EPSILON) == self.number_of_arms:
            return np.array(self.config.EPSILON)
        raise ValueError(
            f"EPSILON list length ({len(self.config.EPSILON)}) must be 0 "
            f"or equal to number of arms ({self.number_of_arms})"
        )

    def _solve_mixture_program(self, round_):
        is_ucb = self.mode == "mixture-ucb"
        alphas = cp.Variable(self.number_of_arms)
        kernel_matrix = self._scaled_kernel()

        if self.config.DEBUG:
            print(f"kernel size -> {kernel_matrix.shape}")
        kernel_matrix += np.eye(self.number_of_arms) * self.config.small_amount

        alpha_array = 0
        for arm_index in range(self.number_of_arms):
            arm_section = np.zeros(self.number_of_arms)
            arm_section[arm_index] = 1
            alpha_array += alphas[arm_index] * arm_section
        if self.config.DEBUG:
            print(f"alphas shape -> {alpha_array.shape}")

        objective = cp.quad_form(alpha_array, kernel_matrix)
        if is_ucb:
            objective -= cp.sum(
                cp.multiply(self._diversity_ucb_weights(round_), alphas)
            )

        if self.use_linear:
            precision_term = cp.sum(cp.multiply(self._precision_values(), alphas))
            if is_ucb:
                precision_term += cp.sum(
                    cp.multiply(self._precision_ucb_weights(), alphas)
                )
            objective -= precision_term

        constraints = [
            cp.sum(alphas) == 1,
            alphas >= self._epsilon_constraints(round_),
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
        return np.clip(alphas.value, 0, None)

    def _select_ogd_alphas(self, round_):
        sample_sizes = self._get_sample_sizes()
        gradient = self._scaled_kernel() @ np.array(sample_sizes).reshape(
            len(sample_sizes), 1
        )
        gradient = 2 / round_ * gradient
        diversity_ucb = self._diversity_ucb_weights(round_)

        if self.use_linear:
            precision_values = self._precision_values()
            precision_ucb = self._precision_ucb_weights()
            adjusted_gradient = [
                gradient[i] - precision_values[i]
                for i in range(len(gradient))
            ]
            adjusted_ucb = [
                diversity_ucb[i] - precision_ucb[i]
                for i in range(len(diversity_ucb))
            ]
            arm_values = [
                adjusted_gradient[i] - adjusted_ucb[i]
                for i in range(len(gradient))
            ]
        else:
            arm_values = [
                gradient[i] - diversity_ucb[i]
                for i in range(len(diversity_ucb))
            ]

        best_arm = np.argmin(arm_values)
        alphas = [0 for _ in range(len(self.model_names))]
        alphas[best_arm] = 1
        return alphas

    def _select_best_model_mixture_greedy_eg(self, round_):
        kernel_matrix = self.kernel
        kernel_matrix = KernelUtils.scaled_kernel(
            kernel_matrix, self._get_sample_sizes(), config=self.config
        )
        kernel_matrix = np.array(kernel_matrix, dtype=np.float64)
        kernel_matrix += np.eye(self.number_of_arms) * self.config.small_amount

        if self.use_linear:
            precision_array = np.array(
                [
                    np.mean(self.linears[mod])
                    if len(self.linears[mod]) > 0
                    else 0.0
                    for mod in self.model_names
                ],
                dtype=np.float64,
            )
            precision_array = _linear_coeff(self.config) * precision_array
        else:
            precision_array = np.zeros(
                self.number_of_arms, dtype=np.float64
            )

        sizes = np.array(self._get_sample_sizes(), dtype=np.float64)
        total = sizes.sum()
        if total > 0:
            alpha_current = sizes / total
        else:
            alpha_current = (
                np.ones(self.number_of_arms, dtype=np.float64)
                / self.number_of_arms
            )
        alpha_current = np.clip(alpha_current, 1e-12, 1.0)
        alpha_current /= np.sum(alpha_current)

        for eg_step in range(self.config.EG_STEPS):
            grad = 2.0 * (kernel_matrix @ alpha_current) - precision_array
            eta = self.config.EG_ETA
            if self.config.EG_SQRT_DECAY:
                eta = eta / np.sqrt(eg_step + 1)

            logw = (
                np.log(np.clip(alpha_current, 1e-12, 1.0)) - eta * grad
            )
            logw -= np.max(logw)
            alpha_new = np.exp(logw)
            alpha_new /= np.sum(alpha_new)

            if self.config.ALPHA_FLOOR > 0:
                alpha_new = np.maximum(alpha_new, self.config.ALPHA_FLOOR)
                alpha_new /= np.sum(alpha_new)

            alpha_current = alpha_new

        self.alphas = alpha_current
        return alpha_current

    def _collect_all_samples(self):
        non_empty_samples = [
            self.samples[model]
            for model in self.model_names
            if len(self.samples[model]) > 0
        ]
        if len(non_empty_samples) == 0:
            return np.array([])
        return np.concatenate(non_empty_samples, axis=0)

    def _get_sample_sizes(self):
        return [len(self.samples[model]) for model in self.model_names]

    def _update_the_kernel(self, samples, samples_model, sigma=None):
        if sigma is None:
            sigma = self.config.SIGMA
        for model in self.model_names:
            k = KernelUtils.frobenius_norm(
                samples, self.samples[model], config=self.config
            )
            self.kernel[
                self.model2idx[samples_model], self.model2idx[model]
            ] += k
            self.kernel[
                self.model2idx[model], self.model2idx[samples_model]
            ] += k
            if self.use_linear:
                prec = self.OfflineEvaluator.compute_sample_precision(samples)
                self.linears[samples_model].append(prec)

    def initiate_kernel_matrix(self):
        for i in range(len(self.model_names)):
            xi = self.samples[self.model_names[i]]
            for j in range(len(self.model_names)):
                xj = self.samples[self.model_names[j]]
                self.kernel[i, j] = KernelUtils.frobenius_norm(
                    xi, xj, config=self.config
                )
        if self.use_linear:
            for model in self.model_names:
                for sample in self.samples[model]:
                    self.linears[model].append(
                        self.OfflineEvaluator.compute_sample_precision(sample)
                    )


__all__ = ["RKEOnlineEvaluator", "default_config"]
