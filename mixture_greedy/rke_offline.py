"""Offline dataset preparation and optimization for RKE/KID experiments."""

import os
import sys

import cvxpy as cp
import numpy as np

from mixture_greedy.data.features import (
    _candidate_feature_npz_paths,
    _load_features_from_npz,
    _resolve_existing_path,
)
from mixture_greedy.metrics.rke import (
    KernelUtils,
    _linear_coeff,
    compute_linear_term,
    compute_reals_nearest_neighbour_distances,
    default_config,
)


def _legacy_cluster_name():
    """Read the intentionally module-global cluster name from the legacy facade."""
    legacy_module = sys.modules.get("online_reproducible")
    if legacy_module is not None and "cluster_name" in vars(legacy_module):
        return vars(legacy_module)["cluster_name"]
    raise NameError("name 'cluster_name' is not defined")


class RKEOfflineEvaluator:
    def __init__(self, model_names, dataset_name, has_reference=False, config=None):
        if config is None:
            config = default_config
        self.config = config
        self.dataset_name = dataset_name
        self.model_names = model_names
        self.datasets = self._load_all_datasets()
        self.real_dataset = None
        self.use_linear = has_reference
        if has_reference:
            self._prepare_real_dataset()
        self.number_of_arms = len(model_names)
        self.kernel = np.zeros((self.number_of_arms, self.number_of_arms))
        self.build_kernel_matrix()
        (
            self.optimal_alphas,
            self.optimal_ans,
            self.true_linears,
            self.true_kernel,
        ) = self.calculate_optimal_alphas(use_linear=has_reference)
        self.optimal_model = self.cal_optimal_model()
        self.true_model_scores = {}
        self.optimal_rke = self.cal_score_with_alphas(self.optimal_alphas, save=True)
        print("offline is done :) -> optimal metric is: ", self.optimal_rke)

    def _prepare_real_dataset(self):
        if self.config.CHANGE_POINT:
            data = np.load(self.config.CHANGE_POINT_PATH)
            real = data["real"]
            size = self.config.DATASET_REAL_CUTOFF
            self.real_dataset = real[
                np.random.choice(len(real), size=min(size, len(real)), replace=False)
            ]
            print(
                f"Loaded real dataset for change-point: {len(self.real_dataset)} samples"
            )
        elif self.config.MALE_FEMALE_ADAPTIVE:
            male_data = np.load(
                os.path.join(self.config.MALE_FEMALE_DATA_PATH, "male.npz")
            )["dino_features"]
            female_data = np.load(
                os.path.join(self.config.MALE_FEMALE_DATA_PATH, "female.npz")
            )["dino_features"]
            size = self.config.DATASET_REAL_CUTOFF
            n_each = size // 2
            real = np.concatenate([male_data[:n_each], female_data[:n_each]], axis=0)
            np.random.shuffle(real)
            self.real_dataset = real[:size]
            print(
                f"Loaded real dataset for male/female: {len(self.real_dataset)} "
                "samples (50/50 mix)"
            )
        else:
            self.real_dataset = self._load_dataset(
                self.dataset_name, self.config.DATASET_REAL_CUTOFF
            )
        self.reals_nnd = None
        if self.config.LINEAR_METRIC != "kid":
            self.reals_nnd = compute_reals_nearest_neighbour_distances(
                self.real_dataset, self.config.KNN
            )
        self.linears = self._compute_linears()

    def _get_dataset_sizes(self):
        return [len(self.datasets[model]) for model in self.model_names]

    def _load_dataset(self, model_name, size=None):
        if size is None:
            size = self.config.OFFLINE_CUTOFF
        print(f"--- Loading model: {model_name}")
        dataset_path = self._get_dataset_path(model_name)
        if self.dataset_name in ["t2t", "city", "celeb", "city_s", "city_us"]:
            n = np.load(dataset_path)["sbert_features"]
        elif self.config.DEBUG:
            n, key_used = _load_features_from_npz(
                dataset_path, self.config.feature_extractor, return_key=True
            )
            print(
                f"[OFFLINE] feature_extractor={self.config.feature_extractor} "
                f"path={dataset_path} key={key_used}"
            )
        else:
            n = _load_features_from_npz(dataset_path, self.config.feature_extractor)
        repetition = False
        print("Dataset Size ->", len(n))
        return n[
            np.random.choice(
                len(n), size=min(size, len(n)), replace=repetition
            )
        ]

    def _load_all_datasets(self):
        if self.config.CHANGE_POINT:
            return self._load_change_point_datasets()
        if self.config.MALE_FEMALE_ADAPTIVE:
            return self._load_male_female_datasets()
        if self.config.CATS_BIRDS_ADAPTIVE:
            return self._load_cats_birds_datasets()
        return {model: self._load_dataset(model) for model in self.model_names}

    def _load_male_female_datasets(self):
        """Load cats/dogs data using each arm's initial configuration."""
        print("--- Loading cats/dogs datasets")
        male_data = np.load("/research/d7/rshr/bahar/MAB/cats.npz")[
            "sbert_features"
        ]
        female_data = np.load("/research/d7/rshr/bahar/MAB/dogs.npz")[
            "sbert_features"
        ]

        np.random.shuffle(male_data)
        np.random.shuffle(female_data)

        print(
            f"Loaded cats (male): {male_data.shape}, "
            f"dogs (female): {female_data.shape}"
        )

        size = self.config.OFFLINE_CUTOFF
        datasets = {}

        for model in self.model_names:
            if model == "arm0":
                male_ratio = self.config.ARM1_MALE_RATIO_BEFORE
            elif model == "arm1":
                male_ratio = self.config.ARM2_MALE_RATIO
            else:
                raise ValueError(
                    f"Unknown arm name for male/female dataset: {model}"
                )

            n_male = int(size * male_ratio)
            n_female = size - n_male

            mixed_data = np.concatenate(
                [male_data[:n_male], female_data[:n_female]], axis=0
            )
            np.random.shuffle(mixed_data)

            datasets[model] = mixed_data[:size]
            print(
                f"Dataset Size for {model} -> {len(datasets[model])} "
                f"(male_ratio={male_ratio})"
            )

        return datasets

    def _load_change_point_datasets(self):
        """Load the pre-regime synthetic change-point data."""
        print(
            f"--- Loading change-point dataset from {self.config.CHANGE_POINT_PATH}"
        )
        data = np.load(self.config.CHANGE_POINT_PATH)
        size = self.config.OFFLINE_CUTOFF
        datasets = {}
        for model in self.model_names:
            if model == "arm0":
                arr = data["arm0_pre"]
            elif model == "arm1":
                arr = data["arm1_pre"]
            else:
                raise ValueError(
                    f"Unknown arm name for change-point dataset: {model}"
                )
            datasets[model] = arr[
                np.random.choice(len(arr), size=min(size, len(arr)), replace=False)
            ]
            print(f"Dataset Size for {model} -> {len(datasets[model])}")
        return datasets

    def _load_cats_birds_datasets(self):
        """Load the initial cats/birds adaptive arm distributions."""
        print(
            "--- Loading cats/birds/animals datasets from "
            f"{self.config.CATS_BIRDS_DATA_PATH}"
        )
        cats_data = np.load(
            os.path.join(
                self.config.CATS_BIRDS_DATA_PATH, "kandinsky_giraffe.npz"
            )
        )["dino_features"]
        catsbirds_data = np.load(
            os.path.join(
                self.config.CATS_BIRDS_DATA_PATH,
                "kandinsky_shark_giraffe_shuffled.npz",
            )
        )["dino_features"]
        animals_data = np.load(  # noqa: F841 - the legacy load is intentional
            os.path.join(
                self.config.CATS_BIRDS_DATA_PATH, "kandinsky_others.npz"
            )
        )["dino_features"]
        np.random.shuffle(cats_data)
        np.random.shuffle(catsbirds_data)

        print(
            f"Loaded cats: {cats_data.shape}, "
            f"catsbirds_shuffled: {catsbirds_data.shape}"
        )

        size = self.config.OFFLINE_CUTOFF
        datasets = {}

        for model in self.model_names:
            if model == "arm0":
                datasets[model] = cats_data[: min(size, len(cats_data))]
            elif model == "arm1":
                datasets[model] = catsbirds_data[: min(size, len(catsbirds_data))]
            else:
                raise ValueError(f"Unknown arm name for cats/birds dataset: {model}")
            print(f"Dataset Size for {model} -> {len(datasets[model])}")

        return datasets

    def _get_dataset_path(self, model_name):
        if self.dataset_name in [
            "imagenet",
            "ffhq",
            "ffhq256",
            "cifar10",
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
        if self.dataset_name == "lsun":
            return os.path.join(
                f"/research/d7/rshr/bahar/data/datasets/{self.dataset_name}/dino/",
                f"{model_name}.npz",
            )
        if self.dataset_name == "FAHQ":
            return os.path.join(
                "/research/d7/rshr/bahar/data/dgm/FFHQ256/features/",
                f"{model_name}.npz",
            )
        if self.dataset_name == "t2i_cluster":
            return os.path.join("../", f"{model_name}/{_legacy_cluster_name()}.npz")
        if self.dataset_name == "t2i_coco":
            return os.path.join("../", f"{model_name}.npz")
        if self.dataset_name == "t2t":
            return os.path.join("../", f"{model_name}_features.npz")
        if self.dataset_name == "imagewoof":
            return os.path.join("", f"{model_name}.npz")
        if self.dataset_name == "styles":
            return os.path.join("", f"{model_name}.npz")
        if self.dataset_name == "t2i_dog":
            return os.path.join(
                "/research/d7/rshr/bahar/data/sdxl", f"{model_name}_features.npz"
            )
        if self.dataset_name == "t2i_birds":
            return os.path.join(
                "/research/d7/rshr/bahar/MAB/red_cartoony_birds/embeddings",
                f"steps_{10}.npz",
            )
        if self.dataset_name == "t2i_cars":
            return os.path.join(
                "/research/d7/rshr/bahar/data/sdxl", f"{model_name}_cars.npz"
            )
        if self.dataset_name == "red_birds":
            return os.path.join(
                "/research/d7/rshr/bahar/red_birds/embeddings",
                f"{model_name}.npz",
            )
        if self.dataset_name in ["city", "celeb", "city_s", "city_us"]:
            return os.path.join(
                f"/research/d7/rshr/bahar/data/text_data/{self.dataset_name}/",
                f"{model_name}_{self.dataset_name}_sbert_features.npz",
            )
        return None

    def _compute_linears(self, samples=None):
        linears = {}
        if samples is None:
            for model in self.model_names:
                linears[model] = compute_linear_term(
                    self.real_dataset,
                    self.datasets[model],
                    self.config.KNN,
                    self.reals_nnd,
                    config=self.config,
                )
        else:
            for model in self.model_names:
                linears[model] = compute_linear_term(
                    self.real_dataset,
                    samples[model],
                    self.config.KNN,
                    self.reals_nnd,
                    config=self.config,
                )
        return linears

    def compute_sample_precision(self, sample):
        if len(sample.shape) == 1:
            sample = sample.reshape(1, sample.shape[0])
        return compute_linear_term(
            self.real_dataset,
            sample,
            self.config.KNN,
            self.reals_nnd,
            config=self.config,
        )

    def cal_optimal_model(self):
        if self.config.QUADRATIC_METRIC == "rke":
            val = [1 / self.kernel[i, i] for i in range(len(self.model_names))]
            if self.use_linear:
                val = [
                    self.kernel[i, i]
                    / (len(self.datasets[self.model_names[i]]) ** 2)
                    - self.config.LAMBDA * self.linears[self.model_names[i]]
                    for i in range(len(self.model_names))
                ]
                return np.argmin(val)
            return np.argmax(val)
        if self.config.QUADRATIC_METRIC == "kid":
            val = [
                self.kernel[i, i]
                / (len(self.datasets[self.model_names[i]]) ** 2)
                - self.linears[self.model_names[i]]
                for i in range(len(self.model_names))
            ]
            return np.argmin(val)
        return None

    def build_kernel_matrix(self):
        for i in range(self.number_of_arms):
            xi = self.datasets[self.model_names[i]]
            for j in range(self.number_of_arms):
                xj = self.datasets[self.model_names[j]]
                self.kernel[i, j] = KernelUtils.frobenius_norm(
                    xi, xj, config=self.config
                )
        if self.config.QUADRATIC_METRIC == "kid":
            self_kernel = KernelUtils.frobenius_norm(
                self.real_dataset, self.real_dataset, config=self.config
            )
            self_kernel /= len(self.real_dataset) ** 2
            self.kid_self_kernel = self_kernel

    def cal_score_with_alphas(self, alphas, save=False):
        quadratic = (
            alphas.reshape(1, alphas.shape[-1])
            @ KernelUtils.scaled_kernel(
                self.kernel, self._get_dataset_sizes(), config=self.config
            )
            @ alphas.reshape(alphas.shape[-1], 1)
        )
        quadratic = quadratic[0, -1]
        if self.config.DEBUG:
            print(alphas)
            print("pre, -> ", quadratic)
            print("rke-mc = ", 1 / quadratic)

        if self.use_linear:
            coeff = _linear_coeff(self.config)
            quadratic -= coeff * np.dot(
                np.array([self.linears[mod] for mod in self.model_names]), alphas
            )

        if self.config.QUADRATIC_METRIC == "kid":
            quadratic += self.kid_self_kernel
        self.true_model_scores["optimal"] = quadratic
        if not self.use_linear:
            quadratic = 1 / quadratic

        return quadratic

    def print_model_scores(self):
        for i in self.model_names:
            datas = self.datasets[i]
            ker = KernelUtils.frobenius_norm(
                datas[: self.config.OFFLINE_CUTOFF],
                datas[: self.config.OFFLINE_CUTOFF],
                sigma=self.config.SIGMA,
                config=self.config,
            )
            self.true_model_scores[i] = ker
            scaled = ker / (len(datas) ** 2)

            if self.config.QUADRATIC_METRIC == "kid":
                if self.config.KID_NEW:
                    ker = ker - len(datas)
                    scaled = ker / (len(datas) * (len(datas) - 1))
                scaled += self.kid_self_kernel
                scaled -= self.linears[i]
                print(scaled)
            elif self.use_linear:
                scaled = scaled - self.config.LAMBDA * self.linears[i]

            if not self.use_linear:
                print(f"{i} ------>", 1 / scaled)

    def calculate_optimal_alphas(self, use_linear=False):
        n_arms = self.number_of_arms
        alphas = cp.Variable(n_arms, nonneg=True)
        kernel = self.kernel
        kernel = KernelUtils.scaled_kernel(
            kernel, self._get_dataset_sizes(), config=self.config
        )
        if self.config.DEBUG:
            print("the offline kernel -> ")
            print("kernel ->", kernel)
        kernel_size = kernel.shape[0]
        alpha_array = 0
        for j in range(n_arms):
            arm_section = np.zeros(n_arms)
            arm_section[j] = 1
            alpha_array += alphas[j] * arm_section

        epsilon_eye = np.eye(kernel_size) * 1e-7
        kernel += epsilon_eye
        probabilistic_kernel = cp.quad_form(alpha_array, kernel)

        linear_array = np.array([0 for _ in range(len(self.model_names))])
        if use_linear:
            linear_array = np.array(
                [self.linears[mod] for mod in self.model_names]
            )
            linear_array = _linear_coeff(self.config) * linear_array
            linear_term = cp.sum(cp.multiply(linear_array, alphas))
            probabilistic_kernel = probabilistic_kernel - linear_term

        objective = cp.Minimize(probabilistic_kernel)
        constraints = [cp.sum(alphas) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(eps_abs=1e-8, eps_rel=1e-8, max_iter=200000)
        optimal_alphas = alphas.value
        return (
            optimal_alphas / np.sum(optimal_alphas),
            problem.value,
            linear_array,
            kernel,
        )


__all__ = ["RKEOfflineEvaluator", "default_config"]
